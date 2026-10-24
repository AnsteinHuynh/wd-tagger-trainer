"""One fact: did the compaction lose a number or a run id, and does it obey the style.

    python scratch/sophie/audit_compact.py OLD NEW
"""

import re
import sys

from tokenizers import Tokenizer

TOK = Tokenizer.from_file(r"I:\MyApps\LLAMA-Models\DeepSeek-V4.1-Flash\tokenizer.json")


def main() -> None:
    old = open(sys.argv[1], encoding="utf-8").read()
    new = open(sys.argv[2], encoding="utf-8").read()
    print(f"tokens {len(TOK.encode(old).ids)} -> {len(TOK.encode(new).ids)}"
          f" ({100 - 100 * len(TOK.encode(new).ids) / len(TOK.encode(old).ids):.1f}% cut)")

    def nums(s):
        return set(x.replace(",", "") for x in re.findall(r"\d[\d,]*(?:\.\d+)?", s))

    miss = sorted(nums(old) - nums(new), key=lambda x: (len(x), x))
    print(f"numeric strings missing: {len(miss)}")
    print(miss)

    def sha(s):
        return set(re.findall(r"\b[0-9a-f]{8}\b", s))

    print(f"sha8 in old {len(sha(old))}, missing from new: {sorted(sha(old) - sha(new))}")

    lines = new.splitlines()
    print("period-ending non-table lines:",
          sum(1 for L in lines if L.rstrip().endswith(".") and not L.strip().startswith("|")))
    print("** occurrences:", new.count("**"))
    print("list-marker lines:", sum(1 for L in lines if re.match(r"^\s*[-*+] ", L)))
    print("lines:", len(lines), "chars:", len(new))


if __name__ == "__main__":
    main()
