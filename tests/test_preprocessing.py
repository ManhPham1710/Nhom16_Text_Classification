import unittest
from src.preprocessing import clean_text


class TestPreprocessing(unittest.TestCase):
    def test_vietnamese_text(self):
        self.assertEqual(clean_text("XIN CHÀO!!!"), "xin chào")

    def test_url_and_email_removed(self):
        result = clean_text("Liên hệ a@b.com tại https://example.com")
        self.assertNotIn("@", result)
        self.assertNotIn("https", result)

    def test_empty(self):
        self.assertEqual(clean_text("   "), "")

    def test_html_unescape(self):
        self.assertEqual(clean_text("A &amp; B"), "a b")


if __name__ == "__main__":
    unittest.main()
