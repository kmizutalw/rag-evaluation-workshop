# RAG 評価ワークショップ

社内ITツール規程・マニュアルを対象とした RAG（Retrieval-Augmented Generation）回答の品質を、  
**人間**と**LLM-as-a-Judge**の両方で評価・比較するワークショップ用アプリです。

---

## 検証する問い

> **LLM-as-a-Judge は、RAG 回答の正確性・根拠性・回答関連性を人間と同じように評価できるか？**

---

## 全体の流れ

```
規程・マニュアル (IT_RAG.json)
        ↓  ingest_milvus.py（既存）
    Milvus に Embedding 登録
        ↓
generate_rag_answers.py
  ├─ Milvus でコンテキスト検索
  ├─ watsonx.ai (openai/gpt-oss-120b) で RAG 回答生成
  └─ eval_data.json に保存
        ↓
llm_judge.py
  ├─ OpenRouter 経由フロンティアモデルで LLM 評価
  ├─ JSON 取得失敗時は自動リトライ（1回）
  └─ eval_data.json に追記
        ↓
   ┌──────────────────────┬───────────────────────┐
   │  streamlit run app.py │  export_eval_sheet.py  │
   │  （社内向け評価 UI）    │  （社外向け Excel 出力） │
   └──────────────────────┴───────────────────────┘
```

---

## ファイル構成

```
eval_app/
├── .env.example              # 環境変数テンプレート
├── requirements.txt          # Python 依存ライブラリ
├── test_cases.json           # 評価テストケース（T01〜T20）
├── eval_data.json            # 評価データ（生成・評価後に自動作成）
├── eval_data_{名前}.json     # 評価者ごとの個人ファイル（app.py が自動作成）
├── generate_rag_answers.py   # STEP 1: RAG 回答生成
├── llm_judge.py              # STEP 2: LLM-as-a-Judge 評価
├── app.py                    # STEP 3: Streamlit 評価 UI（社内ワークショップ用）
├── export_eval_sheet.py      # STEP 4: 社外評価者向け Excel 出力
└── README.md                 # このファイル
```

（Milvus への Embedding 登録は既存の `ingest_milvus.py` を使用）

---

## セットアップ

### 1. 依存ライブラリのインストール

```bash
pip install -r eval_app/requirements.txt
```

### 2. 環境変数の設定

```bash
cp eval_app/.env.example eval_app/.env
# .env を編集して各キーを設定
```

| 変数名 | 説明 |
|---|---|
| `WATSONX_APIKEY` | IBM watsonx.ai の API キー |
| `WATSONX_PROJECT_ID` | watsonx.ai プロジェクト ID |
| `WATSONX_URL` | watsonx.ai エンドポイント URL |
| `MILVUS_URI` | Milvus サーバーの URI |
| `MILVUS_TOKEN` | Milvus 認証トークン（任意） |
| `MILVUS_COLLECTION` | コレクション名（既存と合わせる） |
| `OPENROUTER_API_KEY` | OpenRouter の API キー |
| `JUDGE_MODEL` | LLM-as-a-Judge に使うモデル（例: `anthropic/claude-3.5-sonnet`） |
| `SEARCH_TOP_K` | Milvus 検索チャンク数（既定: 3） |

### 3. Milvus への Embedding 登録（初回のみ）

```bash
python ingest_milvus.py
```

---

## 実行手順

### STEP 1: RAG 回答生成

```bash
python eval_app/generate_rag_answers.py
```

`eval_data.json` が生成されます。20 件それぞれについて、Milvus 検索コンテキストと watsonx.ai の RAG 回答が保存されます。

### STEP 2: LLM-as-a-Judge 評価

```bash
python eval_app/llm_judge.py
```

`eval_data.json` の `llm_eval` フィールドに評価スコアとコメントが追記されます。

- `accuracy` が設定済みの件はスキップ（途中再実行が可能）
- `accuracy=null` かつ `comment` にエラー文字列が残っている件は**自動的に再評価対象**になる
- LLM が JSON 形式で返答しなかった場合は**1回自動リトライ**する

### STEP 3a: 社内ワークショップ — Streamlit UI

```bash
streamlit run eval_app/app.py
```

- 起動時に**評価者名を入力**する画面が表示される
- 入力した名前ごとに `eval_data_{名前}.json` が作成され、個別に保存される
- 同じ名前で再入力すると続きから再開できる
- サイドバーの「名前を変更する」ボタンで別の評価者に切り替え可能

**UIの機能：**

| 機能 | 説明 |
|---|---|
| 評価入力ビュー | 質問・期待回答・検索コンテキスト・RAG回答を並べて表示。スライダーでスコア入力・コメント保存 |
| 集計・比較ビュー | 全件スコア一覧・人間 vs LLM の平均スコア棒グラフ・差分テーブル |
| Excel エクスポート | `rag_evaluation_{名前}.xlsx` としてダウンロード（「評価者」列付き） |

### STEP 3b: 社外評価者向け — Excel 出力

```bash
python eval_app/export_eval_sheet.py --input eval_data.json --output rag_eval_sheet.xlsx
```

引数を省略した場合は `eval_data.json` → `rag_eval_sheet.xlsx` がデフォルトです。

**出力シート構成：**

| シート名 | 内容 |
|---|---|
| 評価ガイド | 評価の目的・記入手順・採点基準・カテゴリ補足の説明 |
| 評価シート | 質問／期待する回答／検索コンテキスト（タイトル＋本文）／RAG回答を掲載。黄色セルにドロップダウン（0/1/2）で入力 |
| 参考：LLM評価 | LLM-as-a-Judge のスコアを色分け表示（参照用） |

---

## 評価データ構成（eval_data.json）

| フィールド | 内容 |
|---|---|
| `id` | テストケース ID（T01〜T20） |
| `question` | 利用者が入力する質問 |
| `reference_doc_ids` | RAG に登録した規程・マニュアルの参照文書 ID |
| `expected_answer` | 人間が事前に作成した正解例 |
| `search_context` | RAG が検索して LLM に渡した文章（doc_id, title, content, score） |
| `rag_answer` | 評価対象となる生成結果 |
| `human_eval` | メンバーが付けた評価（accuracy / groundedness / relevance / comment） |
| `llm_eval` | LLM-as-a-Judge による評価（同上 + judge_model） |

---

## 評価軸と採点基準

各軸 **0〜2 点の 3 段階**で評価します。

### 正確性（accuracy）

| 点数 | 基準 |
|---|---|
| 2 | **正確** — 必要な事実と条件が正しい |
| 1 | **一部正確** — 主な回答は正しいが、条件の不足や軽微な誤りがある |
| 0 | **不正確** — 主要な事実が間違っている |

### 根拠性（groundedness）

| 点数 | 基準 |
|---|---|
| 2 | **根拠あり** — 回答全体を検索コンテキストから確認できる |
| 1 | **一部根拠なし** — 一部に、文書から確認できない表現がある |
| 0 | **根拠なし** — 重要部分が文書に基づいていない |

### 回答関連性（relevance）

| 点数 | 基準 |
|---|---|
| 2 | **関連性が高い** — 質問へ直接、簡潔に回答している |
| 1 | **一部不十分** — 回答にはなっているが、情報不足または冗長 |
| 0 | **関連性が低い** — 質問に答えていない |

---

## テストケースのカテゴリ

| カテゴリ | ID | 評価上の狙い |
|---|---|---|
| 直接回答できる質問 | T01〜T06 | 1 つの文書で回答できる。根拠文書が正しく検索されているかを確認 |
| 複数情報を統合する質問 | T07〜T11 | 複数文書の横断検索が必要。必要情報をすべて拾えているかを確認 |
| 条件・手順の確認 | T12〜T15 | 手順の順序や条件が正確かどうか。「はい/いいえ」判断も含む |
| 文書に答えがない質問 | T16〜T18 | 幻覚せずに「記載なし」と答えられるか |
| 曖昧な質問 | T19〜T20 | 確認質問を返せるか。特定製品と決めつけないかを評価 |

---

## 使用技術

| 役割 | 技術 |
|---|---|
| Embedding | watsonx.ai `intfloat/multilingual-e5-large`（dim=1024） |
| ベクトル DB | Milvus（COSINE / HNSW） |
| RAG 回答生成 | watsonx.ai `openai/gpt-oss-120b` |
| LLM-as-a-Judge | OpenRouter 経由フロンティアモデル（Responses API） |
| 評価 UI | Streamlit |
| Excel 出力 | openpyxl |
