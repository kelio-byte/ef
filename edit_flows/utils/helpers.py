import torch
from torch import Tensor
from einops import rearrange


def x2prob(x: Tensor, vocab_size: int) -> Tensor:
    return torch.nn.functional.one_hot(x, num_classes=vocab_size).float()


def sample_p(pt: Tensor, temperature: float = 1.0) -> Tensor:
    b, l, _ = pt.shape
    pt = rearrange(pt, "b l c -> (b l) c")
    xt = torch.multinomial(pt / temperature, 1)
    return xt.reshape(b, l)


def safe_chr(c: int, compact: bool = False) -> str:
    from edit_flows.utils.tokens import PAD_TOKEN, BOS_TOKEN, GAP_TOKEN

    if c == GAP_TOKEN:
        return "_" if compact else "<GAP>"
    elif c == PAD_TOKEN:
        return "#" if compact else "<PAD>"
    elif c == BOS_TOKEN:
        return "^" if compact else "<BOS>"
    try:
        ch = chr(c)
        if ch.isprintable() and (ch == " " or not ch.isspace()):
            return ch
        return "."
    except Exception:
        return "."


def pretty_parse(x: Tensor, compact: bool = False) -> str:
    return "".join(safe_chr(int(c), compact=compact) for c in x.cpu().numpy().flatten())


def pretty_print(x: Tensor, compact: bool = False) -> None:
    print(pretty_parse(x, compact=compact))
