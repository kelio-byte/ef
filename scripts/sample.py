"""用途：使用 K=1、M 可配置分支策略为产品生成逆合成候选。
输入：产品 token 文件、checkpoint、词表和采样参数。
输出：候选预测文件及采样元数据。
"""

import argparse
from edit_flows.inference import DATA, CHECKPOINT, predict


def parser():
    """作用：定义独立采样命令行选项。输入：无。输出：配置好的参数解析器。"""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--products", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--checkpoint", default=str(CHECKPOINT))
    p.add_argument("--vocab", default=str(DATA / "example.vocab.src"))
    p.add_argument("--n-runs", type=int, default=9,
                   help="每个产品的独立采样次数（默认 9）")
    p.add_argument("--n-children", type=int, default=2,
                   help="每步采样的子候选数 M（默认 2）")
    p.add_argument("--device", default="cuda")
    p.add_argument("--max-products", type=int)
    return p


def run(args):
    """作用：调用正式推理入口生成候选。输入：采样参数对象。输出：预测文件路径。"""
    return predict(
        args.products,
        args.output,
        checkpoint=args.checkpoint,
        vocab=args.vocab,
        n_runs=args.n_runs,
        n_children=args.n_children,
        device=args.device,
        max_products=args.max_products,
    )


if __name__ == "__main__":
    run(parser().parse_args())
