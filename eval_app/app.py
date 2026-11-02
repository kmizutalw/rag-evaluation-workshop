"""
app.py  — RAG 評価ワークショップ Streamlit アプリ

機能:
  - 起動時に評価者名を入力し、個人ファイル (eval_data_{名前}.json) で保存
  - eval_data.json に保存された 20 件の RAG 回答を表示
  - 各回答に対して人間が accuracy / groundedness / relevance（0〜2）とコメントを入力・保存
  - LLM-as-a-Judge の評価結果と並べて比較
  - 評価軸ごとの集計グラフ（人間 vs LLM）を表示
  - Excel エクスポート機能

使い方:
  streamlit run app.py
"""

import io
import json
import os
import re
import shutil
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# マスターデータ（RAG回答・LLM評価が入った元ファイル）
EVAL_DATA_JSON = os.getenv("EVAL_DATA_JSON", "eval_data.json")

# ==========================================
# ページ設定
# ==========================================

st.set_page_config(
    page_title="RAG 評価ワークショップ",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==========================================
# 評価者ファイルパス
# ==========================================

def evaluator_file_path(name: str) -> str:
    """評価者名からパーソナルファイルパスを返す（英数字・ひらがな・漢字などを許容）。"""
    # ファイル名に使えない文字をアンダースコアに置換
    safe = re.sub(r'[\\/:*?"<>|\s]', "_", name).strip("_")
    base = Path(EVAL_DATA_JSON)
    return str(base.parent / f"{base.stem}_{safe}{base.suffix}")


def ensure_evaluator_file(name: str) -> str:
    """評価者ファイルがなければマスターからコピーして作成し、パスを返す。"""
    path = evaluator_file_path(name)
    if not Path(path).exists():
        shutil.copy(EVAL_DATA_JSON, path)
    return path


# ==========================================
# データ読み込み / 保存
# ==========================================

@st.cache_data(show_spinner=False)
def load_eval_data(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_eval_data(data: list[dict], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def reload_data():
    """キャッシュをクリアしてデータを再読み込みする。"""
    st.cache_data.clear()


# ==========================================
# 名前入力画面
# ==========================================

def render_name_input() -> None:
    """評価者名が未設定のときに表示する入力画面。"""
    st.title("🔍 RAG 評価ワークショップ")
    st.markdown("---")
    st.subheader("はじめに、お名前を入力してください")
    st.caption("入力した名前ごとに評価データを保存します。同じ名前で再入力すると続きから再開できます。")

    with st.form("name_form"):
        name = st.text_input(
            "お名前",
            placeholder="例：山田 太郎",
            max_chars=40,
        )
        submitted = st.form_submit_button("評価を開始する →", use_container_width=True)

    if submitted:
        name = name.strip()
        if not name:
            st.error("お名前を入力してください。")
            st.stop()
        if not Path(EVAL_DATA_JSON).exists():
            st.error(
                f"`{EVAL_DATA_JSON}` が見つかりません。\n\n"
                "先に `python generate_rag_answers.py` を実行してください。"
            )
            st.stop()
        st.session_state["evaluator_name"] = name
        st.session_state["eval_file"] = ensure_evaluator_file(name)
        reload_data()
        st.rerun()


# ==========================================
# ヘルパー
# ==========================================

SCORE_OPTIONS = [0, 1, 2]
SCORE_LABELS = {
    0: "0 – 不正確 / 根拠なし / 関連性が低い",
    1: "1 – 一部正確 / 一部根拠なし / 一部不十分",
    2: "2 – 正確 / 根拠あり / 関連性が高い",
}
# 評価軸ごとの選択肢ラベル（スライダーのツールチップ用）
SCORE_LABELS_BY_AXIS = {
    "accuracy":     {0: "0 – 不正確", 1: "1 – 一部正確", 2: "2 – 正確"},
    "groundedness": {0: "0 – 根拠なし", 1: "1 – 一部根拠なし", 2: "2 – 根拠あり"},
    "relevance":    {0: "0 – 関連性が低い", 1: "1 – 一部不十分", 2: "2 – 関連性が高い"},
}

AXES = ["accuracy", "groundedness", "relevance"]
AXES_JP = {"accuracy": "正確性", "groundedness": "根拠性", "relevance": "回答関連性"}


def score_badge(score) -> str:
    if score is None:
        return "–"
    color_map = {0: "#ef4444", 1: "#eab308", 2: "#22c55e"}
    color = color_map.get(int(score), "#94a3b8")
    return f'<span style="background:{color};color:#fff;padding:2px 8px;border-radius:4px;font-weight:bold">{score}</span>'


def format_context(chunks: list[dict]) -> str:
    parts = []
    for i, chunk in enumerate(chunks, 1):
        parts.append(
            f"**[{i}] {chunk.get('system','')} / {chunk.get('title','')}**"
            f"（スコア: {chunk.get('score', '–')}）\n\n{chunk.get('content','')}"
        )
    return "\n\n---\n\n".join(parts)


# ==========================================
# サイドバー
# ==========================================

def render_sidebar(eval_data: list[dict]) -> int:
    st.sidebar.title("🔍 RAG 評価ワークショップ")
    st.sidebar.markdown("---")

    # 評価者名と切り替えボタン
    name = st.session_state.get("evaluator_name", "")
    st.sidebar.markdown(f"**👤 評価者：{name}**")
    if st.sidebar.button("名前を変更する", use_container_width=True):
        del st.session_state["evaluator_name"]
        del st.session_state["eval_file"]
        reload_data()
        st.rerun()
    st.sidebar.markdown("---")

    # 進捗サマリー
    human_done = sum(
        1 for e in eval_data
        if e["human_eval"].get("accuracy") is not None
    )
    llm_done = sum(
        1 for e in eval_data
        if e["llm_eval"].get("accuracy") is not None
    )
    total = len(eval_data)

    st.sidebar.metric("人間評価 完了", f"{human_done} / {total}")
    st.sidebar.metric("LLM評価 完了", f"{llm_done} / {total}")
    st.sidebar.progress(human_done / total if total else 0)

    st.sidebar.markdown("---")
    st.sidebar.markdown("**テストケースを選択**")

    options = [
        f"{e['id']}: {e['question'][:28]}{'…' if len(e['question']) > 28 else ''}"
        for e in eval_data
    ]
    selected_idx = st.sidebar.radio(
        "ケース一覧",
        range(len(eval_data)),
        format_func=lambda i: options[i],
        label_visibility="collapsed",
    )

    st.sidebar.markdown("---")
    if st.sidebar.button("📊 集計・比較ビューへ", use_container_width=True):
        st.session_state["view"] = "summary"

    if st.sidebar.button("📝 評価入力ビューへ", use_container_width=True):
        st.session_state["view"] = "detail"

    return selected_idx


# ==========================================
# 評価入力ビュー（詳細）
# ==========================================

def render_detail_view(entry: dict, entry_idx: int, eval_data: list[dict], eval_file: str) -> None:
    st.header(f"{entry['id']} — 評価入力")

    # 質問・期待回答・RAG回答 を上段に並べる
    col_q, col_exp = st.columns(2)
    with col_q:
        st.subheader("❓ 質問")
        st.info(entry["question"])
    with col_exp:
        st.subheader("✅ 期待する回答")
        st.success(entry["expected_answer"])

    st.subheader("🤖 RAG 回答（評価対象）")
    rag_answer = entry.get("rag_answer", "（未生成）")
    st.warning(rag_answer)

    # 検索コンテキスト（折りたたみ）
    with st.expander("🔎 検索コンテキスト（Milvus 検索結果）", expanded=False):
        chunks = entry.get("search_context", [])
        if chunks:
            st.markdown(format_context(chunks))
        else:
            st.write("コンテキストなし")

    st.markdown("---")
    col_human, col_llm = st.columns(2)

    # ==================
    # 人間評価入力
    # ==================
    with col_human:
        st.subheader("👤 あなたの評価")

        h_eval = entry["human_eval"]

        with st.form(key=f"human_form_{entry['id']}"):
            h_accuracy = st.select_slider(
                "正確性（accuracy）",
                options=SCORE_OPTIONS,
                value=h_eval.get("accuracy") if h_eval.get("accuracy") is not None else 1,
                format_func=lambda x: SCORE_LABELS_BY_AXIS["accuracy"][x],
            )
            h_groundedness = st.select_slider(
                "根拠性（groundedness）",
                options=SCORE_OPTIONS,
                value=h_eval.get("groundedness") if h_eval.get("groundedness") is not None else 1,
                format_func=lambda x: SCORE_LABELS_BY_AXIS["groundedness"][x],
            )
            h_relevance = st.select_slider(
                "回答関連性（relevance）",
                options=SCORE_OPTIONS,
                value=h_eval.get("relevance") if h_eval.get("relevance") is not None else 1,
                format_func=lambda x: SCORE_LABELS_BY_AXIS["relevance"][x],
            )
            h_comment = st.text_area(
                "評価理由・コメント",
                value=h_eval.get("comment", ""),
                height=100,
                placeholder="判断根拠や気になった点を記入してください",
            )
            submitted = st.form_submit_button("💾 評価を保存", use_container_width=True)

        if submitted:
            eval_data[entry_idx]["human_eval"] = {
                "accuracy": h_accuracy,
                "groundedness": h_groundedness,
                "relevance": h_relevance,
                "comment": h_comment,
            }
            save_eval_data(eval_data, eval_file)
            reload_data()
            st.success("保存しました！")
            st.rerun()

    # ==================
    # LLM 評価表示
    # ==================
    with col_llm:
        st.subheader("🧠 LLM 評価（LLM-as-a-Judge）")
        llm_eval = entry.get("llm_eval", {})
        model_name = llm_eval.get("judge_model", "–")
        st.caption(f"モデル: `{model_name}`")

        if llm_eval.get("accuracy") is not None:
            cols = st.columns(3)
            for col, axis in zip(cols, AXES):
                with col:
                    st.metric(
                        label=AXES_JP[axis],
                        value=f"{llm_eval.get(axis, '–')} / 2",
                    )

            st.markdown("**評価コメント**")
            st.markdown(
                f"> {llm_eval.get('comment', '（コメントなし）')}"
            )
        else:
            st.info(
                "LLM評価はまだ実行されていません。\n\n"
                "`python llm_judge.py` を実行してください。"
            )

    # スコア比較バー（両方入力済みの場合）
    if (
        entry["human_eval"].get("accuracy") is not None
        and llm_eval.get("accuracy") is not None
    ):
        st.markdown("---")
        st.subheader("📊 スコア比較")
        compare_df = pd.DataFrame(
            {
                "評価軸": [AXES_JP[a] for a in AXES],
                "人間": [entry["human_eval"].get(a, 0) for a in AXES],
                "LLM": [llm_eval.get(a, 0) for a in AXES],
            }
        ).set_index("評価軸")
        st.bar_chart(compare_df, color=["#3b82d4", "#f97316"])


# ==========================================
# 集計・比較ビュー
# ==========================================

def render_summary_view(eval_data: list[dict]) -> None:
    st.header("📊 集計・比較ビュー")

    # データフレームに変換
    rows = []
    for e in eval_data:
        h = e["human_eval"]
        l = e["llm_eval"]
        rows.append({
            "ID": e["id"],
            "質問": e["question"][:40] + "…" if len(e["question"]) > 40 else e["question"],
            "人間_正確性": h.get("accuracy"),
            "人間_根拠性": h.get("groundedness"),
            "人間_関連性": h.get("relevance"),
            "LLM_正確性": l.get("accuracy"),
            "LLM_根拠性": l.get("groundedness"),
            "LLM_関連性": l.get("relevance"),
        })

    df = pd.DataFrame(rows)

    # ---- 全件スコア一覧テーブル
    st.subheader("全件スコア一覧")
    st.dataframe(df, use_container_width=True, hide_index=True)

    # ---- 軸別平均比較
    st.subheader("評価軸別 平均スコア（人間 vs LLM）")
    avg_data = {}
    for axis, label in AXES_JP.items():
        h_col = f"人間_{label}"
        l_col = f"LLM_{label}"
        avg_data[label] = {
            "人間": round(df[h_col].dropna().mean(), 2) if df[h_col].dropna().any() else None,
            "LLM": round(df[l_col].dropna().mean(), 2) if df[l_col].dropna().any() else None,
        }
    avg_df = pd.DataFrame(avg_data).T
    if not avg_df.empty:
        st.bar_chart(avg_df, color=["#3b82d4", "#f97316"])

    # ---- 差分（人間スコア – LLM スコア）
    st.subheader("スコア差分（人間 – LLM）")
    diff_rows = []
    for _, row in df.iterrows():
        diff_row = {"ID": row["ID"], "質問": row["質問"]}
        for axis, label in AXES_JP.items():
            h_val = row[f"人間_{label}"]
            l_val = row[f"LLM_{label}"]
            if h_val is not None and l_val is not None:
                diff_row[f"差分_{label}"] = h_val - l_val
            else:
                diff_row[f"差分_{label}"] = None
        diff_rows.append(diff_row)
    diff_df = pd.DataFrame(diff_rows)
    st.dataframe(diff_df, use_container_width=True, hide_index=True)

    # ---- Excel エクスポート
    st.subheader("📥 Excel エクスポート")

    name = st.session_state.get("evaluator_name", "unknown")
    export_rows = []
    for e in eval_data:
        h = e["human_eval"]
        l = e["llm_eval"]
        ctx_titles = " / ".join(
            c.get("title", "") for c in e.get("search_context", [])
        )
        export_rows.append({
            "評価者": name,
            "ID": e["id"],
            "質問": e["question"],
            "参照文書ID": ", ".join(e.get("reference_doc_ids", [])),
            "期待する回答": e["expected_answer"],
            "検索コンテキスト（タイトル）": ctx_titles,
            "RAG回答": e.get("rag_answer", ""),
            "人間_正確性": h.get("accuracy"),
            "人間_根拠性": h.get("groundedness"),
            "人間_回答関連性": h.get("relevance"),
            "人間_コメント": h.get("comment", ""),
            "LLM_正確性": l.get("accuracy"),
            "LLM_根拠性": l.get("groundedness"),
            "LLM_回答関連性": l.get("relevance"),
            "LLM_コメント": l.get("comment", ""),
            "LLM_モデル": l.get("judge_model", ""),
        })

    export_df = pd.DataFrame(export_rows)

    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        export_df.to_excel(writer, index=False, sheet_name="評価データ")
    buffer.seek(0)

    safe_name = re.sub(r'[\\/:*?"<>|\s]', "_", name).strip("_")
    st.download_button(
        label="📊 Excelファイルをダウンロード",
        data=buffer,
        file_name=f"rag_evaluation_{safe_name}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
    )


# ==========================================
# メインルーティング
# ==========================================

def main() -> None:
    # 名前が未入力なら入力画面を表示して止める
    if "evaluator_name" not in st.session_state:
        render_name_input()
        st.stop()

    eval_file = st.session_state["eval_file"]
    eval_data = load_eval_data(eval_file)

    # ビュー状態の初期化
    if "view" not in st.session_state:
        st.session_state["view"] = "detail"

    selected_idx = render_sidebar(eval_data)

    if st.session_state["view"] == "summary":
        render_summary_view(eval_data)
    else:
        render_detail_view(eval_data[selected_idx], selected_idx, eval_data, eval_file)


if __name__ == "__main__":
    main()
