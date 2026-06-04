#!/usr/bin/env python3
"""Compare predicted_answer (reconstructed file) vs prompt_response (original file).

Aligns rows by `index`, reports exact-match stats, and lists mismatches.
"""

import json
import sys
from pathlib import Path

ORIG = Path("/Users/amyxu/Downloads/gsm_responses.json")
RECON = Path("/Users/amyxu/Downloads/gsm_responses_reconstructed_input_answer.json")


def normalize(value):
    if value is None:
        return None
    return str(value).strip()


def load(path: Path):
    with open(path) as f:
        return json.load(f)


def main(orig_path: Path = ORIG, recon_path: Path = RECON) -> int:
    orig = load(orig_path)
    recon = load(recon_path)

    orig_by_idx = {r["index"]: r for r in orig}
    recon_by_idx = {r["index"]: r for r in recon}

    common = sorted(set(orig_by_idx) & set(recon_by_idx))
    only_orig = sorted(set(orig_by_idx) - set(recon_by_idx))
    only_recon = sorted(set(recon_by_idx) - set(orig_by_idx))

    matches = 0
    mismatches = []
    for i in common:
        pr = orig_by_idx[i].get("prompt_response")
        pa = recon_by_idx[i].get("predicted_answer")
        if normalize(pr) == normalize(pa):
            matches += 1
        else:
            mismatches.append((i, pr, pa, recon_by_idx[i].get("ground_truth")))

    total = len(common)
    print(f"Original:      {orig_path}")
    print(f"Reconstructed: {recon_path}")
    print(f"Compared {total} indices present in both files.")
    if only_orig:
        head = only_orig[:10]
        print(f"  Only in original: {len(only_orig)} -> {head}{'...' if len(only_orig) > 10 else ''}")
    if only_recon:
        head = only_recon[:10]
        print(f"  Only in reconstructed: {len(only_recon)} -> {head}{'...' if len(only_recon) > 10 else ''}")

    if total:
        print(f"Exact matches: {matches}/{total} ({matches / total:.2%})")
        print(f"Mismatches:    {len(mismatches)}/{total} ({len(mismatches) / total:.2%})")

    if mismatches:
        print("\nMismatches (index | prompt_response | predicted_answer | ground_truth):")
        for i, pr, pa, gt in mismatches:
            print(f"  {i:4d} | {pr!r} | {pa!r} | {gt!r}")

    return 0 if not mismatches else 1


if __name__ == "__main__":
    orig = Path(sys.argv[1]) if len(sys.argv) > 1 else ORIG
    recon = Path(sys.argv[2]) if len(sys.argv) > 2 else RECON
    sys.exit(main(orig, recon))
