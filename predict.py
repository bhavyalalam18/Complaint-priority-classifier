"""
predict.py — Command-line tool to classify complaints from the terminal.

Usage:
    # Single complaint
    python predict.py "System is completely down"

    # Multiple complaints from a file (one per line)
    python predict.py --file complaints_test.txt

    # Interactive mode
    python predict.py
"""

import sys
import argparse
from classifier import predict


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------
COLORS = {
    "high"  : "\033[91m",   # red
    "medium": "\033[93m",   # yellow
    "low"   : "\033[92m",   # green
    "reset" : "\033[0m",
    "bold"  : "\033[1m",
    "dim"   : "\033[2m",
}

ICONS = {"high": "🔴", "medium": "🟡", "low": "🟢"}

def color(text, *keys):
    prefix = "".join(COLORS[k] for k in keys)
    return f"{prefix}{text}{COLORS['reset']}"

def print_result(result: dict):
    p        = result["priority"]
    icon     = ICONS.get(p, "⚪")
    conf_pct = result["confidence"] * 100

    # Check if translation was used
    translated = result.get("translated_text")
    source     = "translated" if translated else "direct"

    print(f"\n  {icon}  Priority  : {color(p.upper(), 'bold', p)}")
    print(f"  📊  Confidence: {conf_pct:.1f}%  ({color(source, 'dim')})")
    print(f"  📝  Text      : {result['text'][:80]}{'...' if len(result['text']) > 80 else ''}")

    # Show translation if available
    if translated:
        print(f"  🌐  Translated: {translated[:80]}")

    # Score bar
    scores = result["scores"]
    print(f"\n  Scores:")
    for cls in ["high", "medium", "low"]:
        bar_len = int(scores[cls] * 30)
        bar     = "█" * bar_len + "░" * (30 - bar_len)
        print(f"    {cls:<8} {color(bar, cls)}  {scores[cls]*100:.1f}%")
    print()


# ---------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------
def classify_single(text: str):
    result = predict(text)
    print_result(result)
    return result


def classify_file(filepath: str):
    try:
        with open(filepath, encoding="utf-8") as f:
            lines = [l.strip() for l in f if l.strip()]
    except FileNotFoundError:
        print(f"❌ File not found: {filepath}")
        sys.exit(1)

    print(f"\n🔍 Classifying {len(lines)} complaints from '{filepath}'...\n")
    print("-" * 60)

    summary = {"high": 0, "medium": 0, "low": 0}
    for i, line in enumerate(lines, 1):
        print(f"[{i}/{len(lines)}] {line[:60]}...")
        result = predict(line)
        summary[result["priority"]] += 1
        print_result(result)

    print("=" * 60)
    print(color("  Summary:", "bold"))
    for p in ["high", "medium", "low"]:
        print(f"    {ICONS[p]}  {p.upper():<8}: {summary[p]}")
    print()


def interactive_mode():
    print(color("\n🎯 Complaint Priority Classifier — Interactive Mode", "bold"))
    print(color("   Type a complaint and press Enter. Type 'quit' to exit.\n", "dim"))

    while True:
        try:
            text = input("  Complaint: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\n👋 Goodbye!")
            break

        if not text:
            continue
        if text.lower() in ("quit", "exit", "q"):
            print("\n👋 Goodbye!")
            break

        try:
            result = predict(text)
            print_result(result)
        except FileNotFoundError:
            print("❌ Model not found. Run 'python train.py' first.")
            break
        except Exception as e:
            print(f"❌ Error: {e}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Classify complaint priority from the command line."
    )
    parser.add_argument(
        "text", nargs="?", default=None,
        help="Complaint text to classify"
    )
    parser.add_argument(
        "--file", "-f", default=None,
        help="Path to a text file with one complaint per line"
    )
    args = parser.parse_args()

    if args.file:
        classify_file(args.file)
    elif args.text:
        classify_single(args.text)
    else:
        interactive_mode()


if __name__ == "__main__":
    main()