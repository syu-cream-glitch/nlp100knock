# 5.大規模言語モデル

## 目次
1. gemini api（無料枠）について
2. .gitignoreについて

### gemini api（無料枠）について
1. apiキーをgoogle AI studioから取得．
2. 以下のコマンドで必要なライブラリをインストール
```bash
uv add google-generativeai python-dotenv
```
3. .envファイルにapiキーを設定
```bash
GEMINI_API_KEY=あなたのAPIキー
```

### .gitignoreについて
1. .gitignoreに以下を追記（機密情報を除外）
```
#GOOGLE_API_KEYを含む.envファイルを無視する
.env
```
