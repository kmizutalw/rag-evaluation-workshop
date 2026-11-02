"""
generate_rag_answers.py

テストケース(test_cases.json)をもとに：
  1. Milvus からコンテキストを検索
  2. watsonx.ai (openai/gpt-oss-120b) で RAG 回答を生成
  3. 評価データ (eval_data.json) として保存

使い方:
  python generate_rag_answers.py
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
TEST_CASES_JSON = os.getenv("TEST_CASES_JSON", "test_cases.json")
EVAL_DATA_JSON = os.getenv("EVAL_DATA_JSON", "eval_data.json")

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

llm = ModelInference(
    model_id=LLM_MODEL_ID,
    credentials=credentials,
    project_id=WATSONX_PROJECT_ID,
    params={
        "max_new_tokens": 1024,
        "temperature": 0.0,
    },
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

def search_milvus(query: str, top_k: int = SEARCH_TOP_K) -> list[dict]:
    """クエリを Milvus で検索し、上位チャンクのリストを返す。"""
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
        chunks.append({
            "doc_id": hit.entity.get("id"),
            "system": hit.entity.get("system"),
            "title": hit.entity.get("title"),
            "content": hit.entity.get("content"),
            "score": round(hit.distance, 4),
        })
    return chunks


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
以下の【参考文書】のみを根拠として、質問に回答してください。
参考文書に情報がない場合は、推測せずに「提供された資料には記載がありません」と明示してください。
回答は簡潔かつ正確に、箇条書きを適宜活用してください。"""


def generate_rag_answer(question: str, context: str) -> str:
    """LLM に RAG 回答を生成させる。"""
    prompt = f"""{RAG_SYSTEM_PROMPT}

【参考文書】
{context}

【質問】
{question}

【回答】"""

    response = llm.generate_text(prompt=prompt)
    return response.strip()


# ==========================================
# メイン処理
# ==========================================

def main():
    with open(TEST_CASES_JSON, "r", encoding="utf-8") as f:
        test_cases = json.load(f)

    eval_data = []

    for tc in test_cases:
        tc_id = tc["id"]
        question = tc["question"]
        expected_answer = tc["expected_answer"]
        reference_doc_ids = tc.get("reference_doc_ids", [])

        print(f"処理中: {tc_id} - {question[:40]}...")

        # Milvus 検索
        chunks = search_milvus(question)
        context_text = build_context_text(chunks)

        # RAG 回答生成
        rag_answer = generate_rag_answer(question, context_text)

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
