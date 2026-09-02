import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.attack import attack


def parse_args():
    parser = argparse.ArgumentParser(description="Run a single prompt generation example.")
    parser.add_argument(
        "query",
        nargs="?",
        default="请将这句话改写为安全评测提示",
        help="Input query to transform.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(attack(args.query))
