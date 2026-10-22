"""Detect credential material in Git content without printing matching values.

Uses only the Python standard library. Historical findings remain failures until
the affected Git history is separately remediated; deleting HEAD is insufficient.
"""

from __future__ import annotations

import argparse
import re
import subprocess
from pathlib import Path

PATTERNS = {
    "cloud access key ID": re.compile(rb"\b(?:LTAI[A-Za-z0-9]{12,32}|AKIA[A-Z0-9]{16})\b"),
    "signed URL or URL token": re.compile(
        rb"(?i)[?&](?:OSSAccessKeyId|Signature|SecurityToken|security-token|"
        rb"x-oss-signature|x-amz-signature|x-goog-signature|access_token|api_key)=[^\s&<>]+"
    ),
    "private key": re.compile(
        rb"-----BEGIN (?:RSA |EC |DSA |OPENSSH |ENCRYPTED )?PRIVATE KEY-----"
    ),
    "service token": re.compile(
        rb"\b(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|"
        rb"sk-[A-Za-z0-9_-]{20,}|xox[baprs]-[A-Za-z0-9-]{20,})\b"
    ),
    "credentials in URL": re.compile(rb"https?://[^\s/:]+:[^\s/@]+@"),
    "credential assignment": re.compile(
        rb"(?i)(?:access[_.-]?key[_.-]?secret|secret[_.-]?key|api[_.-]?key|password|passwd|"
        rb"token)\s*[\"']?\s*[:=]\s*[\"']([A-Za-z0-9_+/=.-]{16,})[\"']"
    ),
}


def findings(data: bytes) -> list[tuple[int, str]]:
    # Binary asset contents need separate artifact/metadata inspection.
    if b"\x00" in data[:8192]:
        return []
    result = []
    for number, line in enumerate(data.splitlines(), 1):
        for label, pattern in PATTERNS.items():
            if pattern.search(line):
                result.append((number, label))
    return result


def git(*args: str) -> bytes:
    return subprocess.check_output(["git", *args])


def scan(mode: str) -> int:
    root = Path(git("rev-parse", "--show-toplevel").decode().strip())
    count = 0
    if mode == "history":
        # Scan every reachable commit, branch and tag, deduplicating blob content.
        objects = git("rev-list", "--objects", "--all").splitlines()
        candidates = []
        for row in objects:
            oid, _, name = row.partition(b" ")
            if git("cat-file", "-t", oid.decode()).strip() == b"blob":
                candidates.append((oid.decode(), name.decode(errors="replace")))
    else:
        paths = (
            git("ls-files", "-z")
            if mode == "tracked"
            else git("diff", "--cached", "--name-only", "--diff-filter=ACMR", "-z")
        )
        candidates = [(None, p.decode()) for p in paths.split(b"\x00") if p]
    for oid, name in candidates:
        if oid is not None:
            data = git("cat-file", "blob", oid)
        elif mode == "staged":
            data = git("show", f":{name}")
        else:
            path = root / name
            # Do not follow tracked links to files outside the checkout.
            if not path.is_file() or path.is_symlink():
                continue
            data = path.read_bytes()
        for number, label in findings(data):
            location = f"{oid[:12]}:{name}" if oid else name
            print(f"{location!r}:{number}: {label} [value redacted]")
            count += 1
    print(f"Secret check ({mode}): {count} finding(s)")
    return int(count > 0)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--tracked", action="store_true", help="Scan tracked working-tree files")
    modes.add_argument("--history", action="store_true", help="Scan all locally reachable history")
    modes.add_argument("--staged", action="store_true", help="Scan staged additions and changes")
    args = parser.parse_args()
    mode = "history" if args.history else "tracked" if args.tracked else "staged"
    return scan(mode)


if __name__ == "__main__":
    raise SystemExit(main())
