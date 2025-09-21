#!/usr/bin/env python3
"""
Decode a fixed list of token IDs using several OpenAI tiktoken encodings.
Usage:
    python scripts/decode_tokens.py

This script prints a line per encoding, e.g.:
    [cl100k_base] -> 'some decoded text'
If an encoding fails, it prints:
    [cl100k_base] failed: <error>

To extend:
- Add encoding names to the ENCODINGS list.
- Change TOKEN_IDS to a different list of integers.
"""
from __future__ import annotations

TOKEN_IDS = [2485, 1129, 268, 34466, 2726, 26583, 24240, 88512, 8526, 31255, 8]
ENCODINGS = [
    "cl100k_base",
    "o200k_base",
    "p50k_base",
    "r50k_base",
]

def main() -> None:
    try:
        import tiktoken  # type: ignore
    except Exception as e:
        # If tiktoken isn't installed, make it clear and exit non-zero
        print(f"Failed to import tiktoken: {e}")
        raise

    for name in ENCODINGS:
        try:
            enc = tiktoken.get_encoding(name)
            text = enc.decode(TOKEN_IDS)
            # repr() ensures visibility of non-printable chars/newlines
            print(f"[{name}] -> {text!r}")
        except Exception as e:
            print(f"[{name}] failed: {e}")

if __name__ == "__main__":
    main()