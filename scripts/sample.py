"""Generate predictions using the frozen formal R9K1M2 protocol."""

import argparse
from edit_flows.inference import DATA, CHECKPOINT, predict


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--products", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--checkpoint", default=str(CHECKPOINT))
    p.add_argument("--vocab", default=str(DATA / "example.vocab.src"))
    p.add_argument("--protocol", choices=["r9"], default="r9")
    p.add_argument("--device", default="cuda")
    p.add_argument("--max-products", type=int)
    return p


def run(args):
    return predict(
        args.products,
        args.output,
        args.checkpoint,
        args.vocab,
        args.protocol,
        device=args.device,
        max_products=args.max_products,
    )


if __name__ == "__main__":
    run(parser().parse_args())
