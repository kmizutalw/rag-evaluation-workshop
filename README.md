# RAG 評価ワークショップ

社内 IT ツール規程・マニュアルを対象とした RAG（Retrieval-Augmented Generation）システムの
回答品質を、**人間**と **LLM-as-a-Judge** の両方で評価・比較するワークショップ用リポジトリです。

> **検証する問い：**  
> LLM-as-a-Judge は、RAG 回答の正確性・根拠性・回答関連性を人間と同じように評価できるか？

---

## リポジトリ構成

```
.
├── ingest_milvus.py          # IT_RAG.json → Milvus に Embedding 登録
├── IT_RAG.json               # 知識ベース（社内 IT ツール規程・マニュアル サンプル 21件）
└── eval_app/
    ├── .env.example              # 環境変数テンプレート
    ├── requirements.txt          # Python 依存ライブラリ
    ├── test_cases.json           # 評価テストケース（T01〜T20）
    ├── generate_rag_answers.py   # STEP 1: RAG 回答生成（最新版）
    ├── generate_rag_answers_v2.py  〜  v11.py   # 過去バージョン（改善履歴）
    ├── llm_judge.py              # STEP 2: LLM-as-a-Judge 評価
    ├── app.py                    # STEP 3: Streamlit 評価 UI
    ├── export_eval_sheet.py      # STEP 4: 社外評価者向け Excel 出力
    └── README.md                 # eval_app 詳細ドキュメント
```

> **注意**: `eval_data.json`（RAG 回答・評価スコアが入った実行結果ファイル）はリポジトリに含まれていません。  
> STEP 1 を実行して生成してください。

---

## 知識ベースについて

`IT_RAG.json` は社内 IT ツール（Microsoft 365, Outlook, Kintone, Backlog, Monday, Slack）の
規程・マニュアルを想定した **LLM 生成サンプルデータ**です。実際の個人情報・機密情報は含みません。

---

## クイックスタート

### 1. 依存ライブラリのインストール

```bash
pip install -r eval_app/requirements.txt
```

### 2. 環境変数の設定

```bash
cp eval_app/.env.example eval_app/.env
# .env を編集して各キーを設定
```

必要な環境変数：

| 変数名 | 説明 |
|---|---|
| `WATSONX_APIKEY` | IBM watsonx.ai の API キー |
| `WATSONX_PROJECT_ID` | watsonx.ai プロジェクト ID |
| `WATSONX_URL` | watsonx.ai エンドポイント URL |
| `MILVUS_URI` | Milvus サーバーの URI |
| `MILVUS_TOKEN` | Milvus 認証トークン（任意） |
| `MILVUS_COLLECTION` | コレクション名 |
| `OPENROUTER_API_KEY` | OpenRouter の API キー（LLM-as-a-Judge 用） |
| `JUDGE_MODEL` | 評価モデル（例: `anthropic/claude-3.5-sonnet`） |

### 3. Milvus への Embedding 登録（初回のみ）

```bash
INPUT_JSON=IT_RAG.json python ingest_milvus.py
```

### 4. RAG 回答生成 → LLM 評価 → UI 起動

```bash
# STEP 1: RAG 回答生成
python eval_app/generate_rag_answers.py

# STEP 2: LLM-as-a-Judge 評価
python eval_app/llm_judge.py

# STEP 3a: 社内ワークショップ UI
streamlit run eval_app/app.py

# STEP 3b: 社外評価者向け Excel 出力
python eval_app/export_eval_sheet.py
```

詳細は [`eval_app/README.md`](eval_app/README.md) を参照してください。

---

## generate_rag_answers バージョン履歴

RAG プロンプトと検索戦略の改善過程を記録したバージョン群です。

| バージョン | 主な変更内容 |
|---|---|
| `v1`（`generate_rag_answers.py`） | 初期実装。`generate_text()` + シンプルな RAG プロンプト |
| `v2` | `max_new_tokens` 削減 / `chat()` へ変更 / スコア下限フィルタ追加 |
| `v3` | HyDE（仮説文書埋め込み）による検索精度向上を試験 |
| `v4` | HyDE + リランキング（スコア重み付け）を追加 |
| `v5` | サブクエリ展開（クエリ分解）による複数文書カバレッジ向上 |
| `v6` | サブクエリ展開 + 重複排除ロジック改善 |
| `v7` | プロンプト構造見直し・根拠番号明記の指示追加 |
| `v8` | 「文書外情報禁止」強化・T16〜T18（文書なし質問）対応 |
| `v9` | 曖昧質問（T19〜T20）向け確認質問指示を強化 |
| `v10` | 「すべての文書を網羅」指示追加（一部ケースで悪化） |
| `v11` | v10 の悪化要因を除去し v8 ベースに戻した最終調整版 |

---

## 評価軸

各軸 **0〜2 点の 3 段階**で評価します。

| 評価軸 | 2点 | 1点 | 0点 |
|---|---|---|---|
| **accuracy**（正確性） | 必要な事実と条件が正しい | 主な回答は正しいが不足・軽微な誤りあり | 主要な事実が間違っている |
| **groundedness**（根拠性） | 回答全体を検索コンテキストから確認できる | 一部に文書から確認できない表現がある | 重要部分が文書に基づいていない |
| **relevance**（回答関連性） | 質問へ直接・簡潔に回答している | 回答にはなっているが情報不足または冗長 | 質問に答えていない |

---

## テストケースのカテゴリ

| カテゴリ | ID | 概要 |
|---|---|---|
| 直接回答できる質問 | T01〜T06 | 単一文書で回答可能 |
| 複数情報を統合する質問 | T07〜T11 | 複数文書の横断検索が必要 |
| 条件・手順を確認する質問 | T12〜T15 | 手順・条件の正確性が重要 |
| 文書に答えがない質問 | T16〜T18 | 「記載なし」と正直に回答できるかを評価 |
| 曖昧な質問 | T19〜T20 | 確認質問を返せるかを評価 |

---

## 使用技術

| 役割 | 技術 |
|---|---|
| Embedding | IBM watsonx.ai `intfloat/multilingual-e5-large`（dim=1024） |
| ベクトル DB | Milvus（COSINE / HNSW） |
| RAG 回答生成 | IBM watsonx.ai `openai/gpt-oss-120b` |
| LLM-as-a-Judge | OpenRouter Responses API |
| 評価 UI | Streamlit |
| Excel 出力 | openpyxl |

---

## ライセンス

本リポジトリのコードは MIT ライセンスです。  
`IT_RAG.json` および `eval_app/test_cases.json` は LLM 生成サンプルデータです。
