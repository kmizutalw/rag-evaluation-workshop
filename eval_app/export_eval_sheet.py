"""
export_eval_sheet.py

eval_data.json から社外評価者向け Excel ファイルを生成する。

シート構成:
  1. 【評価ガイド】  評価方法・観点の説明
  2. 【評価シート】  各テストケースの評価入力欄（記入用）
  3. 【参考：LLM評価】 LLM-as-a-Judge の評価結果（参照用・保護）

使い方:
  python export_eval_sheet.py
  python export_eval_sheet.py --input eval_data.json --output rag_eval_sheet.xlsx
"""

import argparse
import json
from pathlib import Path

from openpyxl import Workbook
from openpyxl.styles import (
    Alignment, Border, Font, PatternFill, Side
)
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.datavalidation import DataValidation

# ==========================================
# スタイル定数
# ==========================================

# カラーパレット
C_HEADER_BG   = "1F4E79"   # 濃紺（ヘッダー背景）
C_HEADER_FG   = "FFFFFF"   # 白（ヘッダー文字）
C_GUIDE_BG    = "DEEAF1"   # 薄青（ガイドセクション背景）
C_SECTION_BG  = "2E75B6"   # 中青（セクション見出し）
C_ROW_ODD     = "FFFFFF"   # 白（奇数行）
C_ROW_EVEN    = "F2F7FB"   # 極薄青（偶数行）
C_INPUT_BG    = "FFFDE7"   # 薄黄（入力セル）
C_LOCKED_BG   = "F5F5F5"   # 薄灰（読み取り専用セル）
C_SCORE_2     = "C6EFCE"   # 緑（スコア2）
C_SCORE_1     = "FFEB9C"   # 黄（スコア1）
C_SCORE_0     = "FFC7CE"   # 赤（スコア0）

def _font(bold=False, size=10, color="000000"):
    return Font(name="游ゴシック", bold=bold, size=size, color=color)

def _fill(hex_color):
    return PatternFill("solid", fgColor=hex_color)

def _border_thin():
    s = Side(style="thin", color="BFBFBF")
    return Border(left=s, right=s, top=s, bottom=s)

def _align(h="left", v="top", wrap=True):
    return Alignment(horizontal=h, vertical=v, wrap_text=wrap)


# ==========================================
# シート 1：評価ガイド
# ==========================================

def build_guide_sheet(ws):
    ws.title = "評価ガイド"
    ws.sheet_view.showGridLines = False
    ws.column_dimensions["A"].width = 3
    ws.column_dimensions["B"].width = 22
    ws.column_dimensions["C"].width = 60
    ws.column_dimensions["D"].width = 38

    def write(row, col, text, bold=False, size=10, fg="000000", bg=None,
              h="left", v="top", wrap=True, merge_to=None):
        cell = ws.cell(row=row, column=col, value=text)
        cell.font = _font(bold=bold, size=size, color=fg)
        cell.alignment = _align(h=h, v=v, wrap=wrap)
        if bg:
            cell.fill = _fill(bg)
        if merge_to:
            ws.merge_cells(
                start_row=row, start_column=col,
                end_row=row, end_column=merge_to
            )
        return cell

    r = 2
    write(r, 2, "RAG 評価ワークショップ — 評価ガイド", bold=True, size=16,
          fg=C_HEADER_FG, bg=C_HEADER_BG, h="center", merge_to=4)
    ws.row_dimensions[r].height = 32
    r += 2

    # ---- このシートの目的
    write(r, 2, "このシートの目的", bold=True, size=12,
          fg=C_HEADER_FG, bg=C_SECTION_BG, merge_to=4)
    ws.row_dimensions[r].height = 22
    r += 1
    purpose = (
        "このワークショップでは、社内規程・マニュアルを対象とした RAG（検索拡張生成）システムの回答品質を、\n"
        "人間の目線で評価していただきます。\n"
        "「評価シート」タブに移動し、各テストケースのRAG回答に対してスコアとコメントを記入してください。"
    )
    write(r, 2, purpose, bg=C_GUIDE_BG, merge_to=4)
    ws.row_dimensions[r].height = 60
    r += 2

    # ---- 記入方法
    write(r, 2, "記入方法", bold=True, size=12,
          fg=C_HEADER_FG, bg=C_SECTION_BG, merge_to=4)
    ws.row_dimensions[r].height = 22
    r += 1
    steps = [
        ("①", "「評価シート」タブを開く",
         "このガイドシートを参照しながら、隣の「評価シート」タブで評価を行います。"),
        ("②", "RAG回答を読む",
         "「質問」「参照文書」「期待する回答」「検索コンテキスト」を確認した上でRAG回答を読んでください。"),
        ("③", "3つの評価軸でスコアを入力",
         "正確性・根拠性・回答関連性の各セルに、ドロップダウンから 0・1・2 を選んで入力してください。"),
        ("④", "評価コメントを記入",
         "スコアの根拠や気になった点を自由記述で記入してください（任意ですが、比較分析に役立ちます）。"),
        ("⑤", "ファイルを保存して返送",
         "記入が完了したら、ファイルを上書き保存してご返送ください。"),
    ]
    for num, step, desc in steps:
        write(r, 2, num, bold=True, h="center", bg=C_GUIDE_BG)
        write(r, 3, step, bold=True, bg=C_GUIDE_BG)
        write(r, 4, desc, bg=C_GUIDE_BG)
        ws.row_dimensions[r].height = 30
        r += 1
    r += 1

    # ---- 評価軸の説明
    write(r, 2, "評価軸の説明と採点基準", bold=True, size=12,
          fg=C_HEADER_FG, bg=C_SECTION_BG, merge_to=4)
    ws.row_dimensions[r].height = 22
    r += 1

    # ヘッダー行
    for col, label in [(2, "評価軸"), (3, "採点基準"), (4, "判断のヒント")]:
        c = ws.cell(row=r, column=col, value=label)
        c.font = _font(bold=True, color=C_HEADER_FG)
        c.fill = _fill("2E75B6")
        c.alignment = _align(h="center", v="center")
        c.border = _border_thin()
    ws.row_dimensions[r].height = 18
    r += 1

    criteria = [
        (
            "正確性\n(accuracy)",
            "2点：正確\n必要な事実と条件が正しい\n\n1点：一部正確\n主な回答は正しいが、条件の不足や軽微な誤りがある\n\n0点：不正確\n主要な事実が間違っている",
            "・「期待する回答」と照らし合わせて、事実・手順・条件が合っているかを確認\n・細かい言い回しの違いは減点対象外。内容の正確さを見る\n・重要なステップが抜けている場合は1点以下"
        ),
        (
            "根拠性\n(groundedness)",
            "2点：根拠あり\n回答全体を検索コンテキストから確認できる\n\n1点：一部根拠なし\n一部に、文書から確認できない表現がある\n\n0点：根拠なし\n重要部分が文書に基づいていない",
            "・「検索コンテキスト」欄の文章にない情報をRAGが勝手に付け加えていないか確認\n・LLMが一般知識で答えている場合は根拠なし（0点）\n・「記載がありません」と正直に回答しているのは根拠あり（2点）"
        ),
        (
            "回答関連性\n(relevance)",
            "2点：関連性が高い\n質問へ直接、簡潔に回答している\n\n1点：一部不十分\n回答にはなっているが、情報不足または冗長\n\n0点：関連性が低い\n質問に答えていない",
            "・質問の意図に対して的外れな回答をしていないか確認\n・必要なことが書かれているが余分な情報が多い → 1点\n・質問の種類（手順確認 / 書類なし / 曖昧な質問）に応じた回答ができているか"
        ),
    ]

    for axis, score_text, hint_text in criteria:
        ws.cell(row=r, column=2, value=axis).font = _font(bold=True)
        ws.cell(row=r, column=2).alignment = _align(h="center", v="top")
        ws.cell(row=r, column=2).fill = _fill(C_GUIDE_BG)
        ws.cell(row=r, column=2).border = _border_thin()
        ws.cell(row=r, column=3, value=score_text).alignment = _align(v="top")
        ws.cell(row=r, column=3).fill = _fill(C_GUIDE_BG)
        ws.cell(row=r, column=3).border = _border_thin()
        ws.cell(row=r, column=4, value=hint_text).alignment = _align(v="top")
        ws.cell(row=r, column=4).fill = _fill(C_GUIDE_BG)
        ws.cell(row=r, column=4).border = _border_thin()
        ws.row_dimensions[r].height = 80
        r += 1
    r += 1

    # ---- 評価カテゴリの補足
    write(r, 2, "テストケースのカテゴリ補足", bold=True, size=12,
          fg=C_HEADER_FG, bg=C_SECTION_BG, merge_to=4)
    ws.row_dimensions[r].height = 22
    r += 1
    categories = [
        ("T01〜T06", "直接回答できる質問",
         "1つの文書を参照すれば回答できる質問です。根拠文書が正しく検索され、事実が正確に回答されているかを見ます。"),
        ("T07〜T11", "複数情報を統合する質問",
         "複数の文書を横断して回答が必要な質問です。必要な情報をすべて拾えているかを確認します。"),
        ("T12〜T15", "条件・手順を確認する質問",
         "手順の順序や条件が正確かどうかが重要です。「はい/いいえ」の判断が求められるものも含みます。"),
        ("T16〜T18", "文書に答えがない質問",
         "登録資料に答えが存在しない質問です。正解は「記載がない」と伝えること。LLMが勝手に推測・でっちあげをしていないかを確認します。"),
        ("T19〜T20", "曖昧な質問",
         "質問が曖昧で、どのシステムについて聞いているのか不明な場合です。確認質問を返せているかが評価ポイントです。"),
    ]
    for ids, cat, desc in categories:
        write(r, 2, ids, bold=True, bg=C_GUIDE_BG, h="center")
        write(r, 3, cat, bold=True, bg=C_GUIDE_BG)
        write(r, 4, desc, bg=C_GUIDE_BG)
        ws.row_dimensions[r].height = 40
        r += 1


# ==========================================
# シート 2：評価シート（記入用）
# ==========================================

def build_eval_sheet(ws, records: list[dict]):
    ws.title = "評価シート"
    ws.sheet_view.showGridLines = False
    ws.freeze_panes = "A3"  # 2行目まで固定

    # 列幅
    col_widths = {
        "A": 6,   # ID
        "B": 36,  # 質問
        "C": 42,  # 期待する回答
        "D": 64,  # 検索コンテキスト（タイトル＋本文）
        "E": 50,  # RAG回答
        "F": 14,  # 正確性
        "G": 14,  # 根拠性
        "H": 14,  # 回答関連性
        "I": 46,  # 評価コメント
    }
    for col_letter, width in col_widths.items():
        ws.column_dimensions[col_letter].width = width

    # ---- 1行目：タイトルバー
    ws.merge_cells("A1:I1")
    title_cell = ws["A1"]
    title_cell.value = "RAG 評価ワークショップ  ―  評価シート　　※黄色のセルに 0 / 1 / 2 を入力してください"
    title_cell.font = _font(bold=True, size=12, color=C_HEADER_FG)
    title_cell.fill = _fill(C_HEADER_BG)
    title_cell.alignment = _align(h="left", v="center", wrap=False)
    ws.row_dimensions[1].height = 24

    # ---- 2行目：ヘッダー
    headers = [
        ("A2", "ID"),
        ("B2", "質問"),
        ("C2", "期待する回答"),
        ("D2", "検索コンテキスト\n（RAGが参照した文書）"),
        ("E2", "RAG回答（評価対象）"),
        ("F2", "正確性\n(accuracy)\n0〜2"),
        ("G2", "根拠性\n(groundedness)\n0〜2"),
        ("H2", "回答関連性\n(relevance)\n0〜2"),
        ("I2", "評価コメント\n（根拠・気づき・問題点など）"),
    ]
    for addr, label in headers:
        c = ws[addr]
        c.value = label
        c.font = _font(bold=True, size=9, color=C_HEADER_FG)
        c.fill = _fill(C_HEADER_BG)
        c.alignment = _align(h="center", v="center")
        c.border = _border_thin()
    ws.row_dimensions[2].height = 48

    # ---- スコアのドロップダウン検証
    dv = DataValidation(type="list", formula1='"0,1,2"', allow_blank=True,
                        showErrorMessage=True, errorTitle="入力エラー",
                        error="0・1・2 のいずれかを選択してください。")
    ws.add_data_validation(dv)

    # ---- データ行
    for i, rec in enumerate(records):
        row = i + 3
        bg = C_ROW_ODD if i % 2 == 0 else C_ROW_EVEN

        ctx_full = "\n\n".join(
            f"[{j+1}] {c.get('system','')} / {c.get('title','')}\n{c.get('content','')}"
            for j, c in enumerate(rec.get("search_context", []))
        )

        values = {
            "A": rec["id"],
            "B": rec["question"],
            "C": rec["expected_answer"],
            "D": ctx_full,
            "E": rec.get("rag_answer", "（未生成）"),
        }
        for col_letter, val in values.items():
            c = ws[f"{col_letter}{row}"]
            c.value = val
            c.font = _font(size=9)
            c.fill = _fill(bg)
            c.alignment = _align(v="top")
            c.border = _border_thin()

        # スコア入力セル（黄色・ドロップダウン）
        for col_letter in ("F", "G", "H"):
            c = ws[f"{col_letter}{row}"]
            c.fill = _fill(C_INPUT_BG)
            c.alignment = _align(h="center", v="center", wrap=False)
            c.font = _font(bold=True, size=11)
            c.border = _border_thin()
            dv.add(c)

        # コメント入力セル
        c = ws[f"I{row}"]
        c.fill = _fill(C_INPUT_BG)
        c.alignment = _align(v="top")
        c.font = _font(size=9)
        c.border = _border_thin()

        ws.row_dimensions[row].height = 120

    # 印刷設定
    ws.print_title_rows = "1:2"
    ws.page_setup.orientation = "landscape"
    ws.page_setup.fitToPage = True
    ws.page_setup.fitToWidth = 1


# ==========================================
# シート 3：参考 LLM評価
# ==========================================

def build_llm_sheet(ws, records: list[dict]):
    ws.title = "参考：LLM評価"
    ws.sheet_view.showGridLines = False

    col_widths = {"A": 6, "B": 36, "C": 12, "D": 12, "E": 12, "F": 60}
    for col_letter, width in col_widths.items():
        ws.column_dimensions[col_letter].width = width

    headers = [("A", "ID"), ("B", "質問"), ("C", "正確性"), ("D", "根拠性"),
               ("E", "回答関連性"), ("F", "LLMコメント")]
    for col_letter, label in headers:
        c = ws[f"{col_letter}1"]
        c.value = label
        c.font = _font(bold=True, color=C_HEADER_FG)
        c.fill = _fill(C_HEADER_BG)
        c.alignment = _align(h="center", v="center")
        c.border = _border_thin()
    ws.row_dimensions[1].height = 20

    score_fill = {2: _fill(C_SCORE_2), 1: _fill(C_SCORE_1), 0: _fill(C_SCORE_0)}

    for i, rec in enumerate(records):
        row = i + 2
        bg = C_ROW_ODD if i % 2 == 0 else C_ROW_EVEN
        llm = rec.get("llm_eval", {})

        for col_letter, val in [("A", rec["id"]), ("B", rec["question"]),
                                  ("F", llm.get("comment", ""))]:
            c = ws[f"{col_letter}{row}"]
            c.value = val
            c.font = _font(size=9)
            c.fill = _fill(bg)
            c.alignment = _align(v="top")
            c.border = _border_thin()

        for col_letter, key in [("C", "accuracy"), ("D", "groundedness"), ("E", "relevance")]:
            score = llm.get(key)
            c = ws[f"{col_letter}{row}"]
            c.value = score
            c.font = _font(bold=True, size=11)
            c.fill = score_fill.get(score, _fill(bg))
            c.alignment = _align(h="center", v="center", wrap=False)
            c.border = _border_thin()

        ws.row_dimensions[row].height = 50


# ==========================================
# メイン
# ==========================================

def main():
    parser = argparse.ArgumentParser(description="eval_data.json から評価用 Excel を生成")
    parser.add_argument("--input",  default="eval_data.json",      help="入力JSONファイル")
    parser.add_argument("--output", default="rag_eval_sheet.xlsx", help="出力Excelファイル")
    args = parser.parse_args()

    if not Path(args.input).exists():
        print(f"エラー: {args.input} が見つかりません。")
        return

    with open(args.input, "r", encoding="utf-8") as f:
        records = json.load(f)

    wb = Workbook()
    # デフォルトシートを削除
    wb.remove(wb.active)

    build_guide_sheet(wb.create_sheet())
    build_eval_sheet(wb.create_sheet(), records)
    build_llm_sheet(wb.create_sheet(), records)

    wb.save(args.output)
    print(f"出力完了: {args.output}  （{len(records)} 件）")
    print("  シート1: 評価ガイド")
    print("  シート2: 評価シート（記入用）")
    print("  シート3: 参考：LLM評価")


if __name__ == "__main__":
    main()
