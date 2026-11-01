import unittest
from unittest import mock

import psycopg

from packages.shared.backfill_safety import connect_redacted, redact_db_url

URL = "postgresql+psycopg://some_user:s3cret-pw@db.example.com/curve?sslmode=require"


class RedactTests(unittest.TestCase):
    def test_url_credentials_removed(self):
        out = redact_db_url(f'invalid connection option "{URL}"')
        self.assertNotIn("s3cret-pw", out)
        self.assertNotIn("some_user", out)
        self.assertIn("db.example.com", out)

    def test_keyword_password_removed(self):
        self.assertNotIn("s3cret", redact_db_url("host=x password=s3cret user=y"))
        self.assertNotIn("s3 cret", redact_db_url("host=x password='s3 cret' user=y"))

    def test_plain_text_unchanged(self):
        self.assertEqual(redact_db_url("connect failed: timeout"), "connect failed: timeout")

    def test_connect_failure_carries_no_url(self):
        err = psycopg.OperationalError(f"bad dsn {URL}")
        with mock.patch("packages.shared.backfill_safety.psycopg.connect", side_effect=err):
            with self.assertRaises(RuntimeError) as cm:
                connect_redacted(URL)
        self.assertNotIn("s3cret-pw", str(cm.exception))
        self.assertIsNone(cm.exception.__cause__)
        self.assertTrue(cm.exception.__suppress_context__)


if __name__ == "__main__":
    unittest.main()
