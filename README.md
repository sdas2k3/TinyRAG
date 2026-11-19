# TinyRAG

TinyRAG is a small Python project for learning retrieval-augmented generation (RAG). Ask a question about a fictional shop’s policies, products, or laptop troubleshooting. TinyRAG combines BM25 and local semantic search to find relevant Markdown passages and sends them to a Groq model for an answer with source citations.

## Setup

You’ll need Python 3.9 or newer. Run these commands from the `TinyRAG` directory.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
[ -f .env ] || (umask 077; cp .env.example .env)
```

Create a Groq API key and set `GROQ_API_KEY` in `.env`. The template sets `GROQ_MODEL` to `openai/gpt-oss-20b`. To switch models, edit `GROQ_MODEL` and restart Flask if it’s running. Values exported in your shell take precedence over `.env`. [Groq API keys](https://console.groq.com/keys)

The first run downloads the public BGE Small tokenizer. The first semantic or hybrid search also downloads the model weights, about 133 MB. These files are cached in `.cache/models/`, and subsequent retrieval runs locally on CPU. A generation request sends the question and retrieved passages to Groq. Keep the Groq API key in your local `.env` or shell environment.

## Run it

Start the Flask chat page, then open [http://127.0.0.1:8000/](http://127.0.0.1:8000/). The transcript scrolls inside the chat, with the message field below it. On narrow screens, select **Help topics** to expand the topic list. Press **Enter** to send and **Shift+Enter** for a new line. The assistant shows a typing indicator while it works. If a request fails, **Try again** resends that question.

```bash
python app.py
```

Use the CLI to ask a question or inspect the passages behind an answer.

```bash
python loader.py
python retriever.py --mode bm25 "My laptop won't charge"
python retriever.py --mode semantic "My laptop won't charge"
python retriever.py --mode hybrid "My laptop won't charge"
python ask.py "How much is standard shipping?"
python evaluate.py
```

Select **New chat** to clear the transcript and bring back the suggested questions. Each request contains only the current question, so include the relevant details in follow-ups.

## How it works

1. **Load and chunk.** `loader.py` reads ten Markdown files in `docs/` and splits at `##` headings. The sample corpus produces twenty chunks. Sections above 300 model tokens, including the search heading and special tokens, split at paragraph or sentence boundaries where possible. Each chunk keeps its full heading, filename, and text.
2. **Rank by words and meaning.** BM25 scores exact tokens in the heading and body using rarity, term frequency, and passage length. BGE Small embeds those same passages and the question. Passage vectors are reused in memory while their text and order stay the same. Semantic candidates need cosine similarity above `0.60`, a cutoff chosen using the sample queries.
3. **Fuse the rankings.** Each list contributes `1 / (60 + rank)` for each candidate it contains. Hybrid search adds the contributions, breaks ties by the better eligible semantic rank, and returns up to three passages. The retrieval CLI shows each passage's scores and rank in the selected mode. Hybrid is the default.
4. **Generate and display.** The CLI and Flask share `answer_question()`. It sends the fused passages to Groq and asks for source citations and missing information. The customer chat formats Markdown and removes citation markers before display. The CLI and the API's `answer` field retain the original cited reply. If neither search yields a candidate, it returns the local fallback without a Groq request.

`evaluate.py` runs each labeled question through all three modes. A question passes when every labeled passage appears in the top three, or when the result is empty for a question labeled with no candidates. These examples were used to tune retrieval, so the result measures behavior on this sample corpus.

## Try these questions

| Question | Expected result |
| --- | --- |
| How long do I have to request a refund? | 30 calendar days from delivery, citing `refunds.md`. |
| How much is standard shipping, and is the fee refundable if I return my order? | $5 per order, and the original shipping fee is non-refundable. The answer cites `shipping.md` and `refunds.md`. |
| Do you ship to Japan? | A shipping passage can match the question, but the answer should say the documents don’t provide enough information. |
| How much is standard shipping, and do you ship to Japan? | $5 per order with a citation to `shipping.md`, plus a note that the docs do not say whether the shop ships to Japan. |
| My laptop won't charge | Hybrid ranks Battery charging failure first. BM25 alone matches the laptop sleeve and warranty passages. |
| What are the opening hours? | No matching passage. TinyRAG returns its fallback locally. |

## Project layout

```text
TinyRAG/
├── app.py                 Flask routes and local server
├── ask.py                 Shared answer pipeline and CLI
├── loader.py              Markdown loading and section splitting
├── retriever.py           BM25, semantic ranking, fusion, and retrieval CLI
├── embeddings.py          Local BGE model, tokenizer, and cosine scoring
├── evaluate.py            Labeled questions and retrieval comparison
├── docs/                  Fictional shop and support documents
├── templates/index.html   Chat interface
├── static/                Chat styles and browser interactions
├── requirements.txt      Python dependencies
├── .env.example           Configuration template
└── AGENTS.md              Guidance for coding agents
```
