"""用途：集中定义模型使用的特殊 token 编号与词表大小。
输入：普通词表大小（需要计算扩展词表时）。
输出：BOS、PAD、GAP 等编号及词表维度。
"""

def bos_token_id(real_vocab_size: int) -> int:
    return BOS_TOKEN


def pad_token_id(real_vocab_size: int) -> int:
    return PAD_TOKEN


def gap_token_id(real_vocab_size: int) -> int:
    return GAP_TOKEN


def model_vocab_size(real_vocab_size: int) -> int:
    return real_vocab_size + 4


def z_vocab_size(real_vocab_size: int) -> int:
    return real_vocab_size + 4


PAD_TOKEN = 0
BOS_TOKEN = 1
GAP_TOKEN = 2
UNK_TOKEN = 3
