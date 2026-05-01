"""Preprocess Stack Overflow into per-user shards.

We pull from the HuggingFace `pacovaldez/stackoverflow-questions` mirror (or a
locally-cached parquet file) and group rows by user, writing the top-N users
by example count to `data_root/{train,test}/{user_id}.txt` with one post per
line. A `users.txt` file lists the chosen user ids.

This script is deterministic given (--seed, --max-users, --min-posts).

Usage:
    python scripts/prepare_stackoverflow.py \\
        --output data/stackoverflow_subset --max-users 10000 --min-posts 50
"""
from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", required=True)
    p.add_argument("--max-users", type=int, default=10000)
    p.add_argument("--min-posts", type=int, default=50)
    p.add_argument("--test-frac", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--source", default="huggingface",
                   choices=["huggingface", "parquet"],
                   help="Where to read raw Stack Overflow data from.")
    p.add_argument("--parquet-path", default=None,
                   help="If --source=parquet, path to a local parquet file "
                        "with columns including user_id (or owner_user_id) and body/text.")
    args = p.parse_args()

    out = Path(args.output).expanduser()
    (out / "train").mkdir(parents=True, exist_ok=True)
    (out / "test").mkdir(parents=True, exist_ok=True)

    if args.source == "huggingface":
        from datasets import load_dataset
        # Streaming mode keeps memory bounded
        ds = load_dataset("pacovaldez/stackoverflow-questions", split="train", streaming=True)
        rows_iter = ds
        user_field_candidates = ("owner_user_id", "user_id", "owner")
        text_field_candidates = ("body", "text", "question_body")
    else:
        import pandas as pd
        if args.parquet_path is None:
            sys.exit("--parquet-path required when --source=parquet")
        df = pd.read_parquet(args.parquet_path)
        rows_iter = (row.to_dict() for _, row in df.iterrows())
        user_field_candidates = ("owner_user_id", "user_id", "owner", "OwnerUserId")
        text_field_candidates = ("body", "text", "Body")

    # First pass: count posts per user
    counts: dict[str, int] = defaultdict(int)
    for row in rows_iter:
        u = next((row[k] for k in user_field_candidates if k in row and row[k] is not None), None)
        if u is None:
            continue
        counts[str(u)] += 1

    # Choose top-N users with at least min-posts
    eligible = [(u, c) for u, c in counts.items() if c >= args.min_posts]
    eligible.sort(key=lambda kv: (-kv[1], kv[0]))
    chosen = [u for u, _ in eligible[: args.max_users]]
    chosen_set = set(chosen)
    print(f"Eligible users: {len(eligible)}; chosen: {len(chosen)}")

    # Second pass: collect posts for chosen users
    if args.source == "huggingface":
        from datasets import load_dataset
        ds = load_dataset("pacovaldez/stackoverflow-questions", split="train", streaming=True)
        rows_iter = ds
    else:
        import pandas as pd
        df = pd.read_parquet(args.parquet_path)
        rows_iter = (row.to_dict() for _, row in df.iterrows())

    user_posts: dict[str, list[str]] = defaultdict(list)
    html_re = re.compile(r"<[^>]+>")
    for row in rows_iter:
        u = next((row[k] for k in user_field_candidates if k in row and row[k] is not None), None)
        if u is None or str(u) not in chosen_set:
            continue
        text = next((row[k] for k in text_field_candidates if k in row and row[k]), None)
        if not text:
            continue
        # Strip HTML and collapse whitespace; one post per line
        text = html_re.sub(" ", str(text))
        text = re.sub(r"\s+", " ", text).strip()
        if text:
            user_posts[str(u)].append(text)

    import random
    rng = random.Random(args.seed)
    written = 0
    for u, posts in user_posts.items():
        rng.shuffle(posts)
        n_test = max(1, int(len(posts) * args.test_frac))
        test_posts = posts[:n_test]
        train_posts = posts[n_test:]
        if not train_posts:
            continue
        (out / "train" / f"{u}.txt").write_text("\n".join(train_posts), encoding="utf-8")
        (out / "test" / f"{u}.txt").write_text("\n".join(test_posts), encoding="utf-8")
        written += 1

    (out / "users.txt").write_text("\n".join(sorted(user_posts.keys())), encoding="utf-8")
    print(f"Wrote {written} user shards to {out}")


if __name__ == "__main__":
    main()
