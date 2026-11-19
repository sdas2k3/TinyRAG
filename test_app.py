import unittest
from unittest.mock import patch

from app import app


class CustomerCitationTests(unittest.TestCase):
    def render_reply(self, answer):
        with patch("app.answer_question", return_value=answer):
            response = app.test_client().post("/ask", json={"question": "Can I get a refund?"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json["answer"], answer)
        return response.json["answer_html"]

    def test_removes_lenticular_bracket_citations_from_refund_reply(self):
        answer = (
            "The policy states that a refund is only available for **unused** items that are in "
            "their original packaging and can be requested within 30 calendar days of "
            "delivery【refunds.md】.\\\n"
            "Your laptop is damaged, so it would not be considered unused, and the warranty "
            "coverage specifically excludes accidental damage【warranty.md】.\n\n"
            "Therefore, you cannot claim a refund (nor a warranty repair) for an accidentally damaged laptop."
        )

        html = self.render_reply(answer)

        self.assertIn("<strong>unused</strong>", html)
        self.assertIn("30 calendar days of delivery.<br", html)
        self.assertIn("excludes accidental damage.", html)
        self.assertIn("(nor a warranty repair)", html)
        self.assertNotIn(".md", html)
        self.assertNotIn("【", html)

    def test_removes_citation_formats_and_preserves_ordinary_content(self):
        answer = (
            "See the [help page](https://example.test/help) for details [first.md].\n\n"
            "Keep 【your receipt】, [this note], and `manual.md`. [[docs/second.markdown]]\n\n"
            "Contact support (Source: 【third.md】) ［fourth.md］〔fifth.md〕.\n\n"
            "Sources: sixth.md, seventh.md"
        )

        html = self.render_reply(answer)

        self.assertEqual(
            html,
            '<p>See the <a href="https://example.test/help">help page</a> for details.</p>\n'
            "<p>Keep 【your receipt】, [this note], and <code>manual.md</code>.</p>\n"
            "<p>Contact support.</p>\n",
        )
