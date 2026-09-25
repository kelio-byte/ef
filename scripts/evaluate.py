"""用途：按 K=1、M 可配置分支策略推理并评测 dev1000 或完整 test。
输入：数据分割、checkpoint 和预测输出目录。
输出：候选预测、采样元数据及 Top-k 评测指标。
"""

import argparse
from pathlib import Path
from edit_flows.inference import DATA, CHECKPOINT, predict
from score import score


def main():
    """作用：解析评测选项并运行推理和评分。输入：命令行参数。输出：写入预测、元数据和指标文件。"""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--split", choices=["dev1000", "test"], default="dev1000")
    p.add_argument("--n-runs", type=int, default=9,
                   help="每个产品的独立采样次数（正式设置为 9）")
    p.add_argument("--n-children", type=int, default=2,
                   help="每步采样的子候选数 M（默认 2）")
    p.add_argument("--checkpoint", default=str(CHECKPOINT))
    p.add_argument("--output", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--max-products", type=int)
    p.add_argument("--score-only", action="store_true")
    args = p.parse_args()
    if args.n_runs < 1:
        p.error("--n-runs must be a positive integer")
    if args.n_children < 1:
        p.error("--n-children must be a positive integer")
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
            n_runs=args.n_runs,
            n_children=args.n_children,
            device=args.device,
            max_products=args.max_products,
        )
    score(Path(args.output) / "predictions.txt", tgt, args.workers)


if __name__ == "__main__":
    main()
