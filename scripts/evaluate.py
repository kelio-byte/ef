"""用途：按正式采样协议推理并评测 dev1000 或完整 test。
输入：数据分割、checkpoint 和预测输出目录。
输出：候选预测、采样元数据及 Top-k 评测指标。
"""

import argparse
from pathlib import Path
from edit_flows.inference import DATA, CHECKPOINT, predict
from score import score


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--split", choices=["dev1000", "test"], default="dev1000")
    p.add_argument("--protocol", choices=["r9"], default="r9")
    p.add_argument("--checkpoint", default=str(CHECKPOINT))
    p.add_argument("--output", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--max-products", type=int)
    p.add_argument("--score-only", action="store_true")
    args = p.parse_args()
    if args.max_products is not None and (
        args.max_products <= 0 or args.max_products % 20
    ):
        p.error("--max-products must be positive and divisible by 20")
    src = DATA / args.split / ("src.txt" if args.split == "dev1000" else "src-test.txt")
    tgt = DATA / args.split / ("tgt.txt" if args.split == "dev1000" else "tgt-test.txt")
    if not args.score_only:
        predict(
            src,
            args.output,
            args.checkpoint,
            protocol=args.protocol,
            device=args.device,
            max_products=args.max_products,
        )
    score(Path(args.output) / "predictions.txt", tgt, args.workers)


if __name__ == "__main__":
    main()
