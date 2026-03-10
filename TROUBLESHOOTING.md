# VEXIS-CLI-1 トラブルシューティング

このドキュメントでは、VEXIS-CLI-1で発生する可能性のある問題とその解決策を詳細に説明します。

## 目次

- [一般的な問題](#一般的な問題)
- [環境別の問題](#環境別の問題)
- [モデル関連の問題](#モデル関連の問題)
- [パフォーマンスの問題](#パフォーマンスの問題)
- [ネットワーク関連の問題](#ネットワーク関連の問題)
- [高度なデバッグ](#高度なデバッグ)

## 一般的な問題

### インストール問題

#### Pythonバージョン互換性

**エラーメッセージ**: `Python 3.8+ is required`
**解決策**:
```bash
# Pythonバージョンを確認
python --version

# pyenvで最新のPythonをインストール（推奨）
pyenv install 3.11.0
pyenv global 3.11.0

# またはHomebrewでインストール（macOS）
brew install python@3.11
```

#### 依存関係のインストール失敗

**エラーメッセージ**: `Failed building wheel for package-name`
**解決策**:
```bash
# pipをアップグレード
pip install --upgrade pip setuptools wheel

# システム依存関係をインストール（Ubuntu/Debian）
sudo apt-get install python3-dev build-essential

# macOSの場合
xcode-select --install

# 仮想環境でクリーンインストール
python -m venv venv
source venv/bin/activate  # Linux/macOS
# venv\Scripts\activate   # Windows
pip install -r requirements.txt
```

### 実行時エラー

#### モジュールインポートエラー

**エラーメッセージ**: `ModuleNotFoundError: No module named 'ai_agent'`
**解決策**:
```bash
# プロジェクトルートにいることを確認
pwd

# PYTHONPATHを設定
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"

# または開発モードでインストール
pip install -e .

# 実行方法を確認
python run.py  # プロジェクトルートから実行
```

#### 設定ファイルエラー

**エラーメッセージ**: `Configuration file not found or invalid`
**解決策**:
```bash
# デフォルト設定を確認
cat config.yaml

# 設定ファイルのバックアップと再作成
cp config.yaml config.yaml.backup
# 設定ファイルを編集して修正
```

## 環境別の問題

### macOS

#### アクセシビリティ権限

**症状**: GUI自動化が機能しない
**解決策**:
1. **システム設定** → **プライバシーとセキュリティ** → **アクセシビリティ**
2. ターミナルまたはPython実行ファイルに権限を付与
3. アプリケーションを再起動

#### Gatekeeperのブロック

**症状**: アプリケーションがブロックされる
**解決策**:
```bash
# アプリケーションのブロックを解除
sudo xattr -rd com.apple.quarantine /path/to/app

# または一時的に無効化
sudo spctl --master-disable
```

### Linux

#### ファイルパーミッション

**症状**: ファイル操作でパーミッション拒否
**解決策**:
```bash
# ユーザーを適切なグループに追加
sudo usermod -a -G docker $USER
sudo usermod -a -G input $USER

# 再ログインが必要
newgrp docker
newgrp input
```

#### ディスプレイサーバー

**症状**: GUI自動化がX11で機能しない
**解決策**:
```bash
# DISPLAY変数を設定
export DISPLAY=:0

# X11サーバーが実行中か確認
ps aux | grep X

# Waylandの場合はXwaylandを有効化
```

### Windows

#### PowerShell実行ポリシー

**症状**: スクリプト実行がブロックされる
**解決策**:
```powershell
# 管理者権限でPowerShellを実行
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# または一時的にバイパス
powershell -ExecutionPolicy Bypass -File script.ps1
```

#### Windows Defender

**症状**: ファイルが検疫される
**解決策**:
1. Windows Defenderで除外ルールを追加
2. プロジェクトフォルダを除外リストに追加
3. リアルタイム保護を一時的に無効化（非推奨）

## モデル関連の問題

### Ollama接続問題

#### サーバーが起動しない

**症状**: `ollama serve` が失敗する
**解決策**:
```bash
# ポートが使用中か確認
lsof -i :11434

# 別のポートでOllamaを起動
ollama serve --port 11435

# ログを確認
ollama logs

# Ollamaを再インストール
curl -fsSL https://ollama.com/install.sh | sh
```

#### モデルダウンロード失敗

**症状**: `ollama pull` がタイムアウト
**解決策**:
```bash
# タイムアウトを延長
export OLLAMA_REQUEST_TIMEOUT=300
ollama pull llama3.2:latest

# プロキシ経由でダウンロード
export HTTP_PROXY=http://proxy:port
export HTTPS_PROXY=http://proxy:port

# 手動ダウンロード（最終手段）
wget https://ollama.com/download/llama3.2:latest
```

### モデル実行エラー

#### メモリ不足

**症状**: `OutOfMemoryError` またはシステムがフリーズ
**解決策**:
```bash
# 小さなモデルを使用
ollama pull llama3.2:1b

# メモリ使用量を確認
free -h  # Linux
vm_stat  # macOS
tasklist # Windows

# スワップを増設（Linux）
sudo fallocate -l 8G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

#### GPU認識されない

**症状**: GPUが使用されずCPUのみで実行
**解決策**:
```bash
# NVIDIA GPUドライバを確認
nvidia-smi

# CUDAインストール（Linux）
sudo apt install nvidia-cuda-toolkit

# OllamaをGPU対応で再インストール
curl -fsSL https://ollama.com/install.sh | sh OLLAMA_GPU=nvidia
```

## パフォーマンスの問題

### 応答時間が遅い

#### ローカルモデルの最適化

**解決策**:
```bash
# モデルを量子化
ollama run llama3.2:latest --quantize q4_0

# より小さなモデルを使用
ollama pull qwen2.5:3b

# コンテキスト長を制限
export OLLAMA_MAX_LOADED_MODELS=1
```

#### クラウドモデルの最適化

**解決策**:
```yaml
# config.yamlでタイムアウトを調整
api:
  timeout: 60
  max_retries: 2
  
# バッチサイズを最適化
processing:
  batch_size: 32
  max_tokens: 2048
```

### メモリリーク

**症状**: 長時間使用するとメモリ使用量が増加
**解決策**:
```bash
# メモリ使用量を監視
watch -n 1 'free -h'

# Pythonガベージコレクションを手動実行
python -c "import gc; gc.collect()"

# 定期的にプロセスを再起動
killall python && python run.py
```

## ネットワーク関連の問題

### プロキシ環境

**症状**: クラウドAPIに接続できない
**解決策**:
```bash
# 環境変数を設定
export HTTP_PROXY=http://proxy.company.com:8080
export HTTPS_PROXY=http://proxy.company.com:8080
export NO_PROXY=localhost,127.0.0.1

# 設定ファイルでプロキシを指定
cat >> ~/.curlrc << EOF
proxy=http://proxy.company.com:8080
noproxy=localhost,127.0.0.1
EOF
```

### SSL/TLS証明書エラー

**症状**: `SSL: CERTIFICATE_VERIFY_FAILED`
**解決策**:
```bash
# 証明書ストアを更新（macOS）
brew update && brew upgrade ca-certificates

# 証明書を無視（非推奨、テストのみ）
export PYTHONHTTPSVERIFY=0
export SSL_VERIFY=false

# カスタム証明書を指定
export SSL_CERT_FILE=/path/to/cert.pem
export REQUESTS_CA_BUNDLE=/path/to/cert.pem
```

### DNS解決問題

**症状**: ホスト名が解決できない
**解決策**:
```bash
# DNSサーバーを変更
echo "nameserver 8.8.8.8" | sudo tee /etc/resolv.conf

# ホストファイルを確認
cat /etc/hosts

# DNSキャッシュをクリア
sudo dscacheutil -flushcache  # macOS
sudo systemctl restart dnsmasq  # Linux
```

## 高度なデバッグ

### 詳細ログの有効化

```bash
# 環境変数でデバッグモードを有効化
export AI_AGENT_DEBUG=true
export AI_AGENT_LOG_LEVEL=DEBUG
export OLLAMA_DEBUG=1

# ログファイルを指定
export AI_AGENT_LOG_FILE=/tmp/vexis-debug.log

# 実行
python run.py --debug --verbose
```

### プロファイリング

```bash
# Pythonプロファイラを使用
python -m cProfile -o profile.stats run.py

# 結果を解析
python -c "
import pstats
p = pstats.Stats('profile.stats')
p.sort_stats('cumulative').print_stats(20)
"

# メモリプロファイリング
pip install memory-profiler
python -m memory_profiler run.py
```

### デバッガーの使用

```bash
# pdbデバッガー
python -m pdb run.py

# ipdb（より高機能）
pip install ipdb
python -m ipdb run.py

# VS Codeリモートデバッグ
pip install debugpy
python -m debugpy --listen 5678 --wait-for-client run.py
```

### システムモニタリング

```bash
# リソース使用量をリアルタイム監視
htop          # CPUとメモリ
iotop         # ディスクI/O
nethogs       # ネットワーク使用量

# GPU使用量（NVIDIA）
watch -n 1 nvidia-smi

# システム情報の収集
python -c "
import psutil
print(f'CPU: {psutil.cpu_percent()}%')
print(f'Memory: {psutil.virtual_memory().percent}%')
print(f'Disk: {psutil.disk_usage(\"/\").percent}%')
"
```

---

問題が解決しない場合は、[GitHub Issues](https://github.com/AInohogosya-team/VEXIS-CLI-1/issues)で報告してください。
