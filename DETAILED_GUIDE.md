# VEXIS-CLI-1 詳細ガイド

このドキュメントでは、VEXIS-CLI-1の詳細な使い方、高度な機能、トラブルシューティングについて説明します。

## 目次

- [詳細な使用例](#詳細な使用例)
- [高度な機能](#高度な機能)
- [アーキテクチャ詳細](#アーキテクチャ詳細)
- [トラブルシューティング](#トラブルシューティング)
- [パフォーマンス最適化](#パフォーマンス最適化)
- [開発者向け情報](#開発者向け情報)

## 詳細な使用例

### ファイル操作

```bash
# 自然言語でのファイル管理
vexis-cli "Find all Python files over 1MB and move them to the archive folder"

# バッチ処理
vexis-cli "Convert all PNG images in this directory to WebP format"

# 高度なファイル検索
vexis-cli "Find all files modified in the last 7 days containing 'TODO' in their content"
```

### コード開発

```bash
# コードレビュー
vexis-cli "Review this pull request and suggest improvements"

# ドキュメント生成
vexis-cli "Generate API documentation for the src/ai_agent module"

# リファクタリング
vexis-cli "Refactor this function to use modern Python patterns and add type hints"
```

### システム管理

```bash
# システム監視
vexis-cli "Check disk usage and alert if any partition is over 80% full"

# ログ分析
vexis-cli "Analyze the last 1000 lines of application.log for error patterns"

# パフォーマンス監視
vexis-cli "Monitor CPU usage for the next 5 minutes and alert if it exceeds 90%"
```

### ワークフロー自動化

```bash
# マルチステップタスク
vexis-cli "Set up a new Python project with virtual environment, install requirements, and initialize git"

# スケジュールタスク
vexis-cli "Create a cron job to backup the database every Sunday at 2 AM"

# デプロイ自動化
vexis-cli "Deploy the current application to staging environment and run health checks"
```

## 高度な機能

### Two-Phase Execution Engine

1. **Planning Phase**: リクエストを分析し実行計画を作成
2. **Execution Phase**: 計画を実行しリアルタイムで監視
3. **Verification Phase**: タスク完了を検証しエラーを処理

### Smart Context Management

- **Session Memory**: 複数のコマンド間でコンテキストを維持
- **File Awareness**: プロジェクト構造を理解
- **History Tracking**: 使用パターンから学習

### Error Recovery

- **Automatic Retries**: 指数バックオフ付きのインテリジェントリトライ
- **Fallback Strategies**: 失敗時に代替アプローチに切り替え
- **Detailed Logging**: デバッグ用の詳細なエラーレポート

## アーキテクチャ詳細

```
VEXIS-CLI-1 Architecture
├── 🧠 AI Agent Core
│   ├── Natural Language Processing
│   ├── Task Planning & Execution
│   └── Verification Engine
├── 🔌 External Integration
│   ├── Ollama Interface
│   ├── Cloud API Connectors
│   └── Platform Abstraction
├── 🎨 User Interface
│   ├── Rich Terminal Display
│   ├── Interactive Mode
│   └── Progress Indicators
└── 🛠️ Utilities
    ├── Configuration Management
    ├── Logging & Monitoring
    └── Error Handling
```

## トラブルシューティング

### 一般的な問題

#### Ollama接続失敗

**問題**: Ollamaサーバーに接続できない
**解決策**: 
```bash
# Ollamaが実行中か確認
ollama list

# Ollamaを再起動
ollama serve

# 設定を確認
cat ~/.ollama/config
```

#### モデルが見つからない

**問題**: Ollamaでモデルが利用できない
**解決策**:
```bash
# モデルをプル
ollama pull llama3.2:latest

# 利用可能なモデル一覧
ollama list
```

#### パーミッション拒否

**問題**: 特定のファイルやディレクトリにアクセスできない
**解決策**:
```bash
# パーミッションを確認
ls -la /path/to/file

# パーミッションを修正（安全な場合）
chmod 644 /path/to/file
```

### 高度なトラブルシューティング

#### メモリ不足エラー

**症状**: 大きなモデル実行時にメモリ不足
**解決策**:
- より小さなモデルを使用（例: `llama3.2:1b`）
- スワップメモリを増設
- 他のアプリケーションを終了

#### ネットワーク接続問題

**症状**: クラウドモデルに接続できない
**解決策**:
```bash
# 接続テスト
curl -I https://api.openai.com/v1/models

# プロキシ設定を確認
echo $HTTP_PROXY
echo $HTTPS_PROXY

# DNS解決を確認
nslookup api.openai.com
```

#### モデル応答が遅い

**症状**: モデルの応答が極端に遅い
**解決策**:
- GPUアクセラレーションを有効化
- より小さなモデルに切り替え
- コンテキスト長を短縮

### デバッグモード

デバッグログを有効化：

```bash
export AI_AGENT_DEBUG=true
python run.py --debug
```

## パフォーマンス最適化

### ベンチマーク

| タスク | ローカルモデル (Llama 3.2) | クラウドモデル (GPT-OSS) |
|------|-------------------------|---------------------|
| シンプルコマンド | ~2s | ~1s |
| コード生成 | ~5s | ~2s |
| 複雑な分析 | ~15s | ~5s |
| マルチステップタスク | ~30s | ~10s |

### リソース使用量

- **メモリ**: 2-8GB（モデルサイズに依存）
- **CPU**: 処理中に中程度の使用量
- **ネットワーク**: ローカルモデルでは最小限、クラウドモデルでは可変

### 最適化ヒント

1. **モデル選択**: タスクに適したサイズのモデルを選択
2. **コンテキスト管理**: 不要なコンテキストをクリア
3. **バッチ処理**: 類似タスクをまとめて実行
4. **キャッシュ活用**: 頻繁使用するモデルをローカルにキャッシュ

## 開発者向け情報

### 開発環境セットアップ

1. **クローンとインストール**
   ```bash
   git clone https://github.com/AInohogosya-team/VEXIS-CLI-1.git
   cd VEXIS-CLI-1
   pip install -e ".[dev]"
   ```

2. **テスト実行**
   ```bash
   python -m pytest tests/
   ```

3. **開発モード**
   ```bash
   python run.py --debug
   ```

### プロジェクト構造

```
src/ai_agent/
├── core_processing/          # AI処理ロジック
├── external_integration/     # 外部API統合
├── platform_abstraction/     # クロスプラットフォーム互換性
├── user_interface/          # CLIとTUIコンポーネント
└── utils/                    # ユーティリティ関数
```

### コントリビューション

1. リポジトリをフォーク
2. 機能ブランチを作成 (`git checkout -b feature/amazing-feature`)
3. 変更をコミット (`git commit -m 'Add amazing feature'`)
4. ブランチにプッシュ (`git push origin feature/amazing-feature`)
5. プルリクエストを開く

---

詳細な情報については、メインのREADME.mdや他のドキュメントを参照してください。
