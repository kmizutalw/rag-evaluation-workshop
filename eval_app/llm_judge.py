"""
llm_judge.py

eval_data.json に保存された RAG 回答を、OpenRouter 経由のフロンティアモデルで評価し、
llm_eval フィールドに結果を書き込む。

評価軸（各 1〜5 点）:
  accuracy    : 回答が正確か（期待する回答と一致しているか）
  groundedness: 回答が参照文書に基づいているか（幻覚がないか）
  relevance   : 回答が質問に対して関連性があるか

使い方:
  python llm_judge.py
"""

import json
import os
import re
import time

import requests
from dotenv import load_dotenv

# ==========================================
# 設定読み込み
# ==========================================

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "anthropic/claude-3.5-sonnet")
EVAL_DATA_JSON = os.getenv("EVAL_DATA_JSON", "eval_data.json")

OPENROUTER_API_URL = "https://openrouter.ai/api/v1/responses"

# API コール間のウェイト（レートリミット対策）
REQUEST_INTERVAL_SEC = 2.0

# ==========================================
# LLM-as-a-Judge プロンプト
# ==========================================

JUDGE_SYSTEM_PROMPT = """あなたはRAGシステムの回答品質を評価する専門家です。
以下の評価軸でRAG回答を評価し、必ず指定のJSON形式のみで回答してください。

【評価軸と採点基準】

■ accuracy（正確性）
2点：正確 — 必要な事実と条件が正しい
1点：一部正確 — 主な回答は正しいが、条件の不足や軽微な誤りがある
0点：不正確 — 主要な事実が間違っている

■ groundedness（根拠性）
2点：根拠あり — 回答全体を検索コンテキストから確認できる
1点：一部根拠なし — 一部に、文書から確認できない表現がある
0点：根拠なし — 重要部分が文書に基づいていない

■ relevance（回答関連性）
2点：関連性が高い — 質問へ直接、簡潔に回答している
1点：一部不十分 — 回答にはなっているが、情報不足または冗長
0点：関連性が低い — 質問に答えていない

【出力形式】
- 返答は JSON オブジェクト **のみ** にしてください。
- 前置き・説明文・思考過程・英語テキストは一切含めないでください。
- 必ず日本語で comment を記述してください。

{"accuracy":<0|1|2>,"groundedness":<0|1|2>,"relevance":<0|1|2>,"comment":"<評価理由を2〜4文で記述>"}"""


def build_judge_prompt(question: str, expected_answer: str,
                       context: str, rag_answer: str) -> str:
    return f"""【質問】
{question}

【期待する回答（正解例）】
{expected_answer}

【検索コンテキスト（RAGが参照した文書）】
{context}

【RAG回答（評価対象）】
{rag_answer}

上記を評価してください。JSONオブジェクトのみを返してください。前置きや説明は不要です。"""


# ==========================================
# OpenRouter API 呼び出し
# ==========================================

def call_openrouter(system_prompt: str, user_prompt: str) -> str:
    """OpenRouter Responses API を呼び出してテキストを返す。"""
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/local/rag-eval",
        "X-Title": "RAG Evaluation Tool",
    }
    # Responses API: input にメッセージ配列、instructions にシステムプロンプト
    payload = {
        "model": JUDGE_MODEL,
        "instructions": system_prompt,
        "input": user_prompt,
        "temperature": 0.0,
        "max_output_tokens": 512,
    }
    response = requests.post(OPENROUTER_API_URL, headers=headers, json=payload, timeout=60)
    if not response.ok:
        raise requests.HTTPError(
            f"{response.status_code} {response.reason}: {response.text}",
            response=response,
        )
    data = response.json()
    # Responses API のレスポンス: output[0].content[0].text
    try:
        return data["output"][0]["content"][0]["text"].strip()
    except (KeyError, IndexError, TypeError) as e:
        raise ValueError(f"レスポンス形式が想定外です: {data}") from e


# ==========================================
# JSON 抽出ヘルパー
# ==========================================

def extract_json(text: str) -> dict:
    """LLM の出力からJSONブロックを抽出してパースする。"""
    # コードブロック内のJSONを優先的に取り出す
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        json_str = match.group(1)
    else:
        # コードブロックなしの場合は { } を探す
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            json_str = match.group(0)
        else:
            raise ValueError(f"JSONが見つかりません: {text}")
    return json.loads(json_str)


# ==========================================
# コンテキスト文字列の再構築
# ==========================================

def format_context(search_context: list[dict]) -> str:
    parts = []
    for i, chunk in enumerate(search_context, 1):
        parts.append(
            f"[参考文書{i}] {chunk['system']} / {chunk['title']}\n{chunk['content']}"
        )
    return "\n\n".join(parts)


# ==========================================
# メイン処理
# ==========================================

def main():
    with open(EVAL_DATA_JSON, "r", encoding="utf-8") as f:
        eval_data = json.load(f)

    updated = 0

    for entry in eval_data:
        tc_id = entry["id"]

        # すでに評価済みの場合はスキップ（エラー終了のものは再評価）
        llm = entry["llm_eval"]
        is_error = llm.get("accuracy") is None and "評価エラー" in llm.get("comment", "")
        if llm.get("accuracy") is not None:
            print(f"スキップ（評価済み）: {tc_id}")
            continue
        if not is_error and llm.get("comment", "") and llm.get("accuracy") is None:
            # comment だけ入っていて accuracy=null → エラー扱いで再評価
            pass

        print(f"評価中: {tc_id} - {entry['question'][:40]}...")

        context_text = format_context(entry.get("search_context", []))
        user_prompt = build_judge_prompt(
            question=entry["question"],
            expected_answer=entry["expected_answer"],
            context=context_text,
            rag_answer=entry.get("rag_answer", ""),
        )

        try:
            raw_output = call_openrouter(JUDGE_SYSTEM_PROMPT, user_prompt)
            try:
                result = extract_json(raw_output)
            except ValueError:
                # JSON が取れなかった場合、JSON のみを要求して1回リトライ
                print(f"  ⚠ JSON取得失敗。リトライ中...")
                time.sleep(REQUEST_INTERVAL_SEC)
                retry_prompt = (
                    "次のテキストから評価結果だけを取り出し、"
                    "JSONオブジェクトのみ（前置き・説明なし）で返してください。\n\n"
                    f"{raw_output}\n\n"
                    '形式: {"accuracy":<0|1|2>,"groundedness":<0|1|2>,"relevance":<0|1|2>,"comment":"..."}'
                )
                raw_output = call_openrouter(JUDGE_SYSTEM_PROMPT, retry_prompt)
                result = extract_json(raw_output)

            entry["llm_eval"] = {
                "accuracy": int(result.get("accuracy", 0)),
                "groundedness": int(result.get("groundedness", 0)),
                "relevance": int(result.get("relevance", 0)),
                "comment": result.get("comment", ""),
                "judge_model": JUDGE_MODEL,
            }
            updated += 1
            print(f"  → accuracy={entry['llm_eval']['accuracy']} "
                  f"groundedness={entry['llm_eval']['groundedness']} "
                  f"relevance={entry['llm_eval']['relevance']}")

        except Exception as e:
            print(f"  ✗ エラー ({tc_id}): {e}")
            entry["llm_eval"]["comment"] = f"評価エラー: {e}"

        time.sleep(REQUEST_INTERVAL_SEC)

    # 上書き保存
    with open(EVAL_DATA_JSON, "w", encoding="utf-8") as f:
        json.dump(eval_data, f, ensure_ascii=False, indent=2)

    print(f"\n完了: {updated} 件を評価し、{EVAL_DATA_JSON} を更新しました。")


if __name__ == "__main__":
    main()
