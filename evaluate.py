"""Compare each retrieval mode against labeled questions from the sample corpus."""

from pathlib import Path

from loader import load_chunks
from retriever import SEARCH_MODES, retrieve


# Each answer is labeled by its source filename and section heading.
# An empty tuple requires a search with zero candidates.
QUESTIONS = (
    ("What's the deadline to get my money back?", (("refunds.md", "Refund window"),)),
    ("How long do I have to request a refund?", (("refunds.md", "Refund window"),)),
    (
        "How much is standard shipping, and is the fee refundable if I return my order?",
        (("shipping.md", "Standard shipping fee"), ("refunds.md", "Shipping fees on returns")),
    ),
    ("My laptop won't charge", (("laptop-power.md", "Battery charging failure"),)),
    ("Where can I check the specifications for FB-65C?", (("accessories.md", "Compatible replacement adapters"),)),
    ("65 W", (("accessories.md", "Compatible replacement adapters"),)),
    ("Will my laptop fit in this sleeve?", (("accessories.md", "Choosing a laptop sleeve"),)),
    ("My computer turns itself off while I work", (("laptop-power.md", "Unexpected shutdowns"),)),
    ("How long are manufacturing defects covered?", (("warranty.md", "Defect coverage"),)),
    ("What proof do I need to start a warranty claim?", (("warranty.md", "Starting a warranty claim"),)),
    ("The courier says it arrived but I can't find my package", (("delivery-issues.md", "Parcel marked delivered but missing"),)),
    ("Can I swap an unused accessory for another color?", (("exchanges.md", "Size and color exchanges"),)),
    ("I received a different item from the one I bought", (("exchanges.md", "Wrong items received"),)),
    ("My card was billed twice", (("payments.md", "Duplicate card authorizations"),)),
    ("Where do I download my invoice?", (("payments.md", "Downloading an invoice"),)),
    ("Can I cancel before dispatch?", (("cancellations.md", "Cancelling before dispatch"),)),
    ("I entered the wrong delivery address", (("cancellations.md", "Correcting a delivery address"),)),
    ("What are the opening hours?", ()),
    ("Who won the World Cup?", ()),
    ("Can I pay in cash?", ()),
)


def main() -> None:
    chunks = load_chunks(Path(__file__).resolve().parent / "docs")
    passes = dict.fromkeys(SEARCH_MODES, 0)
    for question, expected in QUESTIONS:
        print(question)
        for mode in SEARCH_MODES:
            hits = retrieve(question, chunks, mode)
            actual = [(hit.chunk.filename, hit.chunk.heading) for hit in hits]
            passed = set(expected).issubset(actual) if expected else not actual
            passes[mode] += passed
            marker = "PASS" if passed else "MISS"
            headings = " | ".join(heading for _, heading in actual) or "no candidates"
            print(f"  {mode:8} {marker}  {headings}")

    print("\nQuestions with all expected passages in the top three, or zero candidates when expected")
    for mode, count in passes.items():
        print(f"  {mode:8} {count}/{len(QUESTIONS)}")


if __name__ == "__main__":
    main()
