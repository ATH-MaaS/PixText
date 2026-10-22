"""Credential checks run without installing the inference runtime."""

import importlib.util
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "check_secrets.py"
SPEC = importlib.util.spec_from_file_location("check_secrets", SCRIPT)
GUARD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(GUARD)


class SecretGuardTests(unittest.TestCase):
    def test_detects_cloud_id_and_each_signed_url(self):
        samples = [
            b"LTAI" + b"x" * 20,
            b"AKIA" + b"A" * 16,
            *[
                b"https://example.invalid/object?" + field + b"=fixture"
                for field in (
                    b"OSSAccessKeyId",
                    b"Signature",
                    b"SecurityToken",
                    b"x-oss-signature",
                    b"X-Amz-Signature",
                    b"X-Goog-Signature",
                    b"access_token",
                    b"api_key",
                )
            ],
        ]
        for sample in samples:
            with self.subTest(sample=sample):
                self.assertTrue(GUARD.findings(sample))

    def test_detects_private_keys_service_tokens_and_passwords(self):
        samples = [
            b"-----BEGIN " + b"OPENSSH PRIVATE KEY-----",
            b"ghp_" + b"x" * 30,
            b"password='" + b"x" * 20 + b"'",
            b"access_key_secret='" + b"x" * 30 + b"'",
            b"https://" + b"user:fixture" + b"@example.invalid/object",
        ]
        for sample in samples:
            self.assertTrue(GUARD.findings(sample))

    def test_preserves_location_without_values(self):
        result = GUARD.findings(b"safe\nhttps://example.invalid/?" + b"Signature" + b"=fixture\n")
        self.assertEqual(result, [(2, "signed URL or URL token")])

    def test_allows_documentation_and_checksums(self):
        self.assertFalse(
            GUARD.findings(
                b"Never commit OSSAccessKeyId, Signature or SecurityToken.\n"
                b"Set OSS_ACCESS_KEY_SECRET through the approved credential provider.\n" + b"a" * 64
            )
        )

    def test_binary_assets_are_explicitly_out_of_scope(self):
        self.assertFalse(
            GUARD.findings(b"\x00https://example.invalid/?" + b"Signature" + b"=fixture")
        )

    def test_cli_scans_index_and_history_without_leaking_values(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            subprocess.run(["git", "init", "-q"], cwd=root, check=True)
            path = root / "README.md"
            secret = "https://example.invalid/?" + "Signature" + "=fixture-sensitive-value"
            path.write_text(secret)
            subprocess.run(["git", "add", "README.md"], cwd=root, check=True)
            # A clean working copy must not hide a secret already staged in Git.
            path.write_text("safe\n")
            staged = subprocess.run(
                [sys.executable, str(SCRIPT), "--staged"],
                cwd=root,
                text=True,
                capture_output=True,
            )
            self.assertEqual(staged.returncode, 1)
            self.assertIn("README.md", staged.stdout)
            self.assertNotIn("fixture-sensitive-value", staged.stdout + staged.stderr)
            tracked = subprocess.run(
                [sys.executable, str(SCRIPT), "--tracked"],
                cwd=root,
                text=True,
                capture_output=True,
            )
            self.assertEqual(tracked.returncode, 0)
            # Create reachable history without running unrelated system Git hooks.
            tree = subprocess.check_output(["git", "write-tree"], cwd=root).decode().strip()
            commit = (
                subprocess.check_output(
                    [
                        "git",
                        "-c",
                        "user.name=Fixture",
                        "-c",
                        "user.email=fixture@example.invalid",
                        "commit-tree",
                        tree,
                        "-m",
                        "Fixture",
                    ],
                    cwd=root,
                )
                .decode()
                .strip()
            )
            subprocess.run(
                ["git", "update-ref", "refs/heads/audit-fixture", commit], cwd=root, check=True
            )
            historical = subprocess.run(
                [sys.executable, str(SCRIPT), "--history"],
                cwd=root,
                text=True,
                capture_output=True,
            )
            self.assertEqual(historical.returncode, 1)
            self.assertNotIn("fixture-sensitive-value", historical.stdout + historical.stderr)


if __name__ == "__main__":
    unittest.main()
