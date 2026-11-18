"""Predict half-life scores from a JSON list of annotated transcript records."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from bioseq_models import load_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("transcripts", type=Path)
    parser.add_argument("--species", choices=("human", "mouse"), default="human")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--output", type=Path, default=Path("saluki_scores.csv"))
    args = parser.parse_args()
    records = json.loads(args.transcripts.read_text())
    if not isinstance(records, list):
        parser.error("The input must be a JSON list of annotated transcript objects")
    model = load_model(
        "saluki" if args.species == "human" else "saluki_mouse",
        device=args.device,
        batch_size=args.batch_size,
    )
    scores = model.score_transcripts(records)
    with args.output.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["transcript_id", "half_life_score"])
        for i, (record, score) in enumerate(zip(records, scores[:, 0])):
            writer.writerow(
                [record.get("transcript_id", f"transcript_{i + 1}"), float(score)]
            )
    print(args.output)


if __name__ == "__main__":
    main()
