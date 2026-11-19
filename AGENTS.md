# TinyRAG contributor guidance

TinyRAG is a small teaching project. Keep the Markdown → chunks → retrieval → answer flow easy to trace. See [README.md](README.md) for setup and usage.

## Implementation

- Keep document loading in `loader.py`, keyword scoring in `retriever.py`, and Groq calls in `ask.py`. Both the CLI and Flask should use `ask.answer_question()`.
- Keep the filename and heading with each chunk so retrieval results can be inspected and answers can cite their sources.
- Preserve the local fallback when retrieval returns no matches. With partial evidence, the generation prompt should ask for supported facts and identify what’s missing.
- Keep Groq credentials on the server. Use `.env.example` for setting names and safe example values. Treat `.env` as local and keep it out of commits, logs, and example output.
- Keep dependencies in `requirements.txt`. Update the README when setup, commands, configuration, or user-visible behavior changes.

## Checks

Run relevant checks from the project root using the project’s virtual environment.

```bash
source .venv/bin/activate
python -m pip check
python loader.py
python retriever.py "How much is standard shipping, and is the fee refundable if I return my order?"
python evaluate.py
python ask.py "What are the opening hours?"
```

- With the bundled docs, the loader should print twenty chunks and the shipping question should retrieve both the shipping price and the refund policy. The opening-hours question should reach the fallback without a Groq request.
- For Flask changes, exercise `GET /` and `POST /ask` with Flask’s test client. Check a valid question, invalid input, and a service error when those paths are affected.
- When changing generation, verify the request payload and test the supported and missing-evidence examples in the README. Use a mock for local checks and fictional sample questions for any live checks.
