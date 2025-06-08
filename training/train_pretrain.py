"""Self-supervised pretraining stub."""
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=1)
    args = parser.parse_args()
    # TODO: implement pretraining
    print(f"Pretraining for {args.epochs} epochs...")


if __name__ == "__main__":
    main()
