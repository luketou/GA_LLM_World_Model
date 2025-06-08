"""Weakly-supervised finetuning stub."""
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=1)
    args = parser.parse_args()
    # TODO: implement finetuning
    print(f"Finetuning for {args.epochs} epochs...")


if __name__ == "__main__":
    main()
