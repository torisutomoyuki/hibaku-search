"""
被爆証言検索 API サーバー
========================
起動方法:
    export OPENAI_API_KEY="sk-proj-..."
    export SUPABASE_URL="https://xxxx.supabase.co"
    export SUPABASE_KEY="eyJh..."
    python3 server.py
"""

import os
import json
import requests
from flask import Flask, request, jsonify, Response
from flask_cors import CORS
import openai

app = Flask(__name__)
CORS(app)

OPENAI_KEY   = os.environ.get("OPENAI_API_KEY")
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")
EMBEDDING_MODEL = "text-embedding-3-small"


def get_embedding(text):
    client = openai.OpenAI(api_key=OPENAI_KEY)
    response = client.embeddings.create(model=EMBEDDING_MODEL, input=text)
    return response.data[0].embedding


def search_supabase(embedding, match_count=10):
    url = f"{SUPABASE_URL}/rest/v1/rpc/search_testimonies"
    headers = {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "query_embedding": embedding,
        "match_count": match_count
    }
    resp = requests.post(url, headers=headers, json=payload)
    if resp.status_code != 200:
        raise Exception(f"Supabase error: {resp.status_code} {resp.text[:200]}")
    return resp.json()


@app.route("/search", methods=["POST"])
def search():
    data = request.get_json()
    query = data.get("query", "").strip()
    count = min(int(data.get("count", 10)), 200)

    if not query:
        return jsonify({"error": "クエリが空です"}), 400

    try:
        embedding = get_embedding(query)
        results = search_supabase(embedding, match_count=count)

        # 同じarticle_idの重複を除去（証言単位でまとめる）
        seen = set()
        unique = []
        for r in results:
            aid = r.get("chunk_id", "").rsplit("_", 2)[0]
            if aid not in seen:
                seen.add(aid)
                unique.append(r)

        return jsonify({"results": unique})

    except Exception as e:
        return jsonify({"error": str(e)}), 500
@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    question = data.get("question", "").strip()

    if not question:
        return jsonify({"error": "質問が空です"}), 400

    try:
        embedding = get_embedding(question)
        chunks = search_supabase(embedding, match_count=6)

        if not chunks:
            return jsonify({"error": "関連する証言が見つかりませんでした。"})

        context = ""
        sources = []
        seen = set()
        for c in chunks:
            aid = c.get("chunk_id", "").rsplit("_", 2)[0]
            if aid not in seen:
                seen.add(aid)
                sources.append({
                    "name": c.get("name") or "氏名不明",
                    "title": c.get("title", ""),
                    "url": c.get("source_url", ""),
                    "published_date": c.get("published_date", "")
                })
            context += f"【{c.get('name') or '氏名不明'}さんの証言】\n{c.get('text', '')}\n\n"

        import anthropic, json

        client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

        system_prompt = """You are a gentle, respectful guide to the Nagasaki atomic bomb testimony archive.

Respond in the same language as the user's question (Japanese or English).

Structure your response naturally, like a human guide speaking quietly:

For Japanese:
- Start with something like「いくつかの証言をご紹介します。」
- Introduce each person warmly by name
- End with a quiet synthesis starting with「これらの証言から見えてくるのは、」

For English:
- Start with "Let me share some testimonies with you."
- Introduce each person warmly
- End with a quiet synthesis starting with "Looking across these testimonies,"

Rules:
1. Only use information from the provided testimonies.
2. Always name each testimony giver.
3. Quote their words using 「」(Japanese) or "" (English).
4. Never use markdown symbols like #, ##, **, or *.
5. Write in flowing, natural paragraphs."""

        def generate():
            yield f"data: {json.dumps({'type': 'sources', 'sources': sources})}\n\n"
            with client.messages.stream(
                model="claude-haiku-4-5-20251001",
                max_tokens=1200,
                system=system_prompt,
                messages=[{"role": "user", "content": f"Question: {question}\n\nTestimonies:\n{context}"}]
            ) as stream:
                for text in stream.text_stream:
                    yield f"data: {json.dumps({'type': 'text', 'text': text})}\n\n"
            yield f"data: {json.dumps({'type': 'done'})}\n\n"

        return Response(generate(), mimetype="text/event-stream",
                       headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/stats", methods=["GET"])
def stats():
    try:
        url = f"{SUPABASE_URL}/rest/v1/testimonies"
        headers = {
            "apikey": SUPABASE_KEY,
            "Authorization": f"Bearer {SUPABASE_KEY}",
        }
        params = {"select": "themes,text", "limit": 9000}
        resp = requests.get(url, headers=headers, params=params)
        rows = resp.json()

        from collections import Counter
        import re

        # テーマ集計
        theme_counts = Counter()
        for row in rows:
            for theme in (row.get("themes") or []):
                theme_counts[theme] += 1

        # 頻出語集計
        word_counts = Counter()
        stop_words = {"する", "ある", "いる", "なる", "れる", "られ", "ない",
            "その", "この", "また", "そして", "しかし", "ため", "こと",
            "もの", "とき", "ところ", "まま", "よう", "から", "まで",
            "だっ", "だが", "でも", "けど", "けれど", "という", "てい",
            "てき", "てし", "てあ", "われ", "彼ら", "私た", "自分",
            # 広告・UI関連を除外
            "案内", "事業", "募集", "通販", "特集", "企画", "購読",
            "配達", "スタッフ", "カルチャ", "サービ", "ビジネ",
            "会社", "紙面", "投稿", "海外", "ニュース", "上部",
            "日朝", "翌日", "時ごろ", "公式", "祝い",}
        for row in rows:
            text = row.get("text", "") or ""
            words = re.findall(r'[一-龥ぁ-んァ-ン]{2,6}', text)
            for w in words:
                if w not in stop_words:
                    word_counts[w] += 1

        return jsonify({
            "themes": dict(theme_counts.most_common()),
            "words": dict(word_counts.most_common(120))
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    if not OPENAI_KEY:
        print("エラー: OPENAI_API_KEY が未設定")
        exit(1)
    if not SUPABASE_URL or not SUPABASE_KEY:
        print("エラー: SUPABASE_URL / SUPABASE_KEY が未設定")
        exit(1)
    print("サーバー起動中... http://localhost:5001")
    app.run(port=5001, debug=False)
