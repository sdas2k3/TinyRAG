import re

from flask import Flask, jsonify, render_template, request
from markdown_it import MarkdownIt

from ask import AnswerError, answer_question


app = Flask(__name__)
MAX_QUESTION_LENGTH = 1000
MARKDOWN = MarkdownIt("commonmark", {"html": False}).enable("table").disable("image")

FILE_REFERENCE = r"`?[\w./-]+\.(?:md|markdown)`?"
FILE_LIST = rf"{FILE_REFERENCE}(?:[ \t]*[,;、，；][ \t]*{FILE_REFERENCE})*"
BRACKETED_CITATION = "(?:" + "|".join(
    rf"{re.escape(left)}[ \t]*{FILE_LIST}[ \t]*{re.escape(right)}"
    for left, right in (("[", "]"), ("[[", "]]"), ("【", "】"), ("［", "］"), ("〔", "〕"))
) + ")"
PARENTHETICAL_CITATION = (
    rf"[ \t]*(?:sources?[:：][ \t]*)?(?:{FILE_LIST}|{BRACKETED_CITATION})[ \t]*"
)
CITATION = re.compile(
    rf"[ \t]*(?:{BRACKETED_CITATION}(?:\([^\n)]*\))?"
    rf"|\({PARENTHETICAL_CITATION}\)|（{PARENTHETICAL_CITATION}）)",
    re.IGNORECASE,
)
SOURCE_LINE = re.compile(
    rf"^[ \t]*(?:\*\*)?sources?[:：][ \t]*(?:\*\*)?[ \t]*(?:{FILE_LIST})?[ \t]*(?:\r?\n|$)",
    re.IGNORECASE | re.MULTILINE,
)


@app.get("/")
def index():
    return render_template("index.html", max_question_length=MAX_QUESTION_LENGTH)


@app.post("/ask")
def ask():
    data = request.get_json(silent=True)
    question = data.get("question") if isinstance(data, dict) else None
    if not isinstance(question, str) or not question.strip():
        return jsonify(error="Enter a question to continue."), 400
    question = question.strip()
    if len(question) > MAX_QUESTION_LENGTH:
        return jsonify(error=f"Use {MAX_QUESTION_LENGTH} characters or fewer."), 400

    try:
        answer = answer_question(question)
    except AnswerError as error:
        return jsonify(error=str(error)), 503

    # Keep the original answer for API callers and strip citations from the chat's copy.
    customer_answer = SOURCE_LINE.sub("", CITATION.sub("", answer)).strip()
    return jsonify(answer=answer, answer_html=MARKDOWN.render(customer_answer))


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8000)
