"""
generate_rag_answers_v7.py

generate_rag_answers_v6.py からの修正。変更点：
  - サブクエリ生成の max_tokens を 128 → 512 に増加
      openai/gpt-oss-120b は推論モデルのため、128 トークンでは
      <thinking> フェーズだけで使い切り content が出力されなかった
  - サブクエリ生成プロンプトを簡潔化して推論トークン消費を抑制
  - message に 'content' がなく 'reasoning' のみの場合のフォールバック追加
  - デバッグ出力を削除（原因が判明したため）
  - max_tokens は 768 を維持

テストケース(test_cases.json)をもとに：
  1. Milvus からコンテキストを検索（サブクエリ展開）
  2. watsonx.ai (openai/gpt-oss-120b) で RAG 回答を生成
  3. 評価データ (eval_data.json) として保存

使い方:
  python generate_rag_answers_v7.py
"""

import json
import os

from dotenv import load_dotenv
from pymilvus import connections, Collection
from ibm_watsonx_ai import Credentials
from ibm_watsonx_ai.foundation_models import Embeddings, ModelInference

# ==========================================
# 設定読み込み
# ==========================================

load_dotenv()

WATSONX_APIKEY = os.getenv("WATSONX_APIKEY")
WATSONX_PROJECT_ID = os.getenv("WATSONX_PROJECT_ID")
WATSONX_URL = os.getenv("WATSONX_URL")

MILVUS_URI = os.getenv("MILVUS_URI")
MILVUS_TOKEN = os.getenv("MILVUS_TOKEN", "")
COLLECTION_NAME = os.getenv("MILVUS_COLLECTION", "it_rag")

SEARCH_TOP_K = int(os.getenv("SEARCH_TOP_K", "3"))
SEARCH_MIN_SCORE = float(os.getenv("SEARCH_MIN_SCORE", "0.75"))
SEARCH_FINAL_K = int(os.getenv("SEARCH_FINAL_K", "5"))   # マージ後に使用する上位件数
TEST_CASES_JSON = os.getenv("TEST_CASES_JSON", "test_cases.json")
EVAL_DATA_JSON = os.getenv("EVAL_DATA_JSON", "eval_data_v7.json")

LLM_MODEL_ID = "openai/gpt-oss-120b"
EMBEDDING_MODEL_ID = "intfloat/multilingual-e5-large"

# ==========================================
# watsonx.ai クライアント初期化
# ==========================================

credentials = Credentials(
    api_key=WATSONX_APIKEY,
    url=WATSONX_URL,
)

embedding_model = Embeddings(
    model_id=EMBEDDING_MODEL_ID,
    credentials=credentials,
    project_id=WATSONX_PROJECT_ID,
)

# chat() API 用パラメータ（TextChatParameters に対応するキー）
# max_new_tokens は generate_text() 用。chat() では max_tokens を使う。
LLM_CHAT_PARAMS = {
    "max_tokens": 768,
    "temperature": 0.0,
}

llm = ModelInference(
    model_id=LLM_MODEL_ID,
    credentials=credentials,
    project_id=WATSONX_PROJECT_ID,
)

# ==========================================
# Milvus 接続
# ==========================================

connections.connect(
    alias="default",
    uri=MILVUS_URI,
    token=MILVUS_TOKEN if MILVUS_TOKEN else None,
)

collection = Collection(name=COLLECTION_NAME)
collection.load()


# ==========================================
# RAG 検索関数
# ==========================================

def search_milvus(query: str, top_k: int = SEARCH_TOP_K,
                  min_score: float = SEARCH_MIN_SCORE) -> list[dict]:
    """クエリを Milvus で検索し、上位チャンクのリストを返す。

    min_score 未満のチャンクはフィルタリングして除外する。
    """
    query_text = f"query: {query}"
    vector = embedding_model.embed_documents(texts=[query_text])[0]

    results = collection.search(
        data=[vector],
        anns_field="embedding",
        param={"metric_type": "COSINE", "params": {"ef": 64}},
        limit=top_k,
        output_fields=["id", "system", "title", "content"],
    )

    chunks = []
    for hit in results[0]:
        score = round(hit.distance, 4)
        if score < min_score:
            continue
        chunks.append({
            "doc_id": hit.entity.get("id"),
            "system": hit.entity.get("system"),
            "title": hit.entity.get("title"),
            "content": hit.entity.get("content"),
            "score": score,
        })
    return chunks


# openai/gpt-oss-120b は推論モデルのため、システムプロンプトは極力簡潔にする。
# 長いプロンプトは <thinking> トークン消費を増やし、max_tokens 超過の原因になる。
SUBQUERY_SYSTEM_PROMPT = """与えられた質問に対して、異なるキーワードで検索するためのクエリを最大2個、1行ずつ日本語で出力してください。前置きや番号は不要です。"""


def generate_subqueries(question: str) -> list[str]:
    """元の質問からサブクエリを最大2個生成する。生成失敗時は空リストを返す。

    openai/gpt-oss-120b は推論モデルのため message に 'content' ではなく
    'reasoning' のみが含まれる場合がある。max_tokens=512 で content まで
    出力させる。
    """
    messages = [
        {"role": "system", "content": SUBQUERY_SYSTEM_PROMPT},
        {"role": "user",   "content": question},
    ]
    try:
        # max_tokens=512: 推論モデルの <thinking> フェーズ後に content を出力させるため
        response = llm.chat(messages=messages, params={"max_tokens": 512, "temperature": 0.0})
        choice = response.get("choices", [{}])[0]
        message = choice.get("message", {})

        # 推論モデルは content / reasoning どちらかに回答を入れることがある
        text = message.get("content") or message.get("reasoning") or ""
        if not text:
            print(f"    ⚠ サブクエリ: 応答テキスト空 (finish_reason={choice.get('finish_reason')})", flush=True)
            return []

        text = text.strip()
        # 空行を除いた各行をサブクエリとして取得（最大2個）
        subqueries = [line.strip() for line in text.splitlines() if line.strip()][:2]
        return subqueries
    except Exception as e:
        print(f"    ⚠ サブクエリ生成失敗: {e}", flush=True)
        return []


def search_with_subqueries(question: str, final_k: int = SEARCH_FINAL_K) -> list[dict]:
    """元クエリ＋サブクエリで検索し、重複排除・スコア順で上位 final_k 件を返す。"""
    # 元クエリで検索
    all_chunks: dict[str, dict] = {}
    for chunk in search_milvus(question):
        if chunk["doc_id"] not in all_chunks:
            all_chunks[chunk["doc_id"]] = chunk

    # サブクエリ生成＆検索
    subqueries = generate_subqueries(question)
    print(f"    サブクエリ: {subqueries}", flush=True)
    for sq in subqueries:
        for chunk in search_milvus(sq):
            if chunk["doc_id"] not in all_chunks:
                all_chunks[chunk["doc_id"]] = chunk
            else:
                # すでにある場合はスコアが高い方を保持
                if chunk["score"] > all_chunks[chunk["doc_id"]]["score"]:
                    all_chunks[chunk["doc_id"]] = chunk

    # スコア降順でソートして上位 final_k 件を返す
    merged = sorted(all_chunks.values(), key=lambda c: c["score"], reverse=True)
    return merged[:final_k]


def build_context_text(chunks: list[dict]) -> str:
    """検索チャンクを LLM プロンプト用のコンテキスト文字列に変換する。"""
    parts = []
    for i, chunk in enumerate(chunks, 1):
        parts.append(
            f"[参考文書{i}] {chunk['system']} / {chunk['title']}\n{chunk['content']}"
        )
    return "\n\n".join(parts)


# ==========================================
# RAG 回答生成関数
# ==========================================

RAG_SYSTEM_PROMPT = """あなたは社内ITツール規程・マニュアルを参照して回答するアシスタントです。

【回答の原則】
1. まず【参考文書】の内容をすべて確認し、質問に関連する情報を探してください。
2. 関連する情報が見つかった場合は、必ずその内容を根拠として回答してください。
3. 【参考文書】のどこにも関連する情報が見当たらない場合は、「対象の文書には関連情報が見当たりません。管理者または社内IT部門にご確認ください。」と回答してください。
4. 質問が曖昧で特定のシステムや対象が不明な場合は、確認質問を返してください。ただし【参考文書】に一般的な回答が含まれている場合は、確認質問と合わせてその内容も回答してください。
5. 回答は簡潔かつ正確に、箇条書きを適宜活用してください。

【出力の制約】
- 回答は日本語のみで記述してください。
- 英語のテキスト・思考過程・作業メモは一切含めないでください。
- 前置きや「回答:」などのラベルは不要です。回答本文のみを出力してください。"""


def generate_rag_answer(question: str, context: str) -> str:
    """LLM に RAG 回答を生成させる（chat API 使用）。"""
    messages = [
        {"role": "system", "content": RAG_SYSTEM_PROMPT},
        {"role": "user",   "content": f"【参考文書】\n{context}\n\n【質問】\n{question}"},
    ]
    response = llm.chat(messages=messages, params=LLM_CHAT_PARAMS)
    return response["choices"][0]["message"]["content"].strip()


# ==========================================
# メイン処理
# ==========================================

def main():
    with open(TEST_CASES_JSON, "r", encoding="utf-8") as f:
        test_cases = json.load(f)

    eval_data = []

    for i, tc in enumerate(test_cases, 1):
        tc_id = tc["id"]
        question = tc["question"]
        expected_answer = tc["expected_answer"]
        reference_doc_ids = tc.get("reference_doc_ids", [])

        print(f"[{i}/{len(test_cases)}] {tc_id}: {question[:40]}...", flush=True)

        # Milvus 検索（サブクエリ展開）
        print(f"  → Milvus 検索中（サブクエリ展開）...", flush=True)
        chunks = search_with_subqueries(question)
        print(f"  → {len(chunks)} 件取得 (doc_ids: {[c['doc_id'] for c in chunks]})", flush=True)

        context_text = build_context_text(chunks)

        # RAG 回答生成
        print(f"  → LLM 回答生成中...", flush=True)
        rag_answer = generate_rag_answer(question, context_text)
        print(f"  → 完了 (回答 {len(rag_answer)} 文字)", flush=True)

        # 評価データエントリ構築
        entry = {
            "id": tc_id,
            "question": question,
            "reference_doc_ids": reference_doc_ids,
            "expected_answer": expected_answer,
            "search_context": [
                {
                    "doc_id": c["doc_id"],
                    "system": c["system"],
                    "title": c["title"],
                    "content": c["content"],
                    "score": c["score"],
                }
                for c in chunks
            ],
            "rag_answer": rag_answer,
            # 人間評価・LLM評価は後工程で記入
            "human_eval": {
                "accuracy": None,       # 1〜5
                "groundedness": None,   # 1〜5
                "relevance": None,      # 1〜5
                "comment": "",
            },
            "llm_eval": {
                "accuracy": None,
                "groundedness": None,
                "relevance": None,
                "comment": "",
                "judge_model": "",
            },
        }
        eval_data.append(entry)

    with open(EVAL_DATA_JSON, "w", encoding="utf-8") as f:
        json.dump(eval_data, f, ensure_ascii=False, indent=2)

    print(f"\n完了: {len(eval_data)} 件を {EVAL_DATA_JSON} に保存しました。")


if __name__ == "__main__":
    main()
