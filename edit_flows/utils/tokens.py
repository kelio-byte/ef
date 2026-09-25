"""用途：集中定义模型使用的特殊 token 编号与词表大小。
输入：普通词表大小（需要计算扩展词表时）。
输出：BOS、PAD、GAP 等编号及词表维度。
"""

def bos_token_id(real_vocab_size: int) -> int:
    """作用：返回 BOS 编号。输入：普通词表大小（接口兼容参数）。输出：BOS token 整数编号。"""
    return BOS_TOKEN


def pad_token_id(real_vocab_size: int) -> int:
    """作用：返回 PAD 编号。输入：普通词表大小（接口兼容参数）。输出：PAD token 整数编号。"""
    return PAD_TOKEN


def gap_token_id(real_vocab_size: int) -> int:
    """作用：返回 GAP 编号。输入：普通词表大小（接口兼容参数）。输出：GAP token 整数编号。"""
    return GAP_TOKEN


def model_vocab_size(real_vocab_size: int) -> int:
    """作用：计算含四种特殊 token 的模型词表大小。输入：普通词表大小。输出：扩展词表大小。"""
    return real_vocab_size + 4


def z_vocab_size(real_vocab_size: int) -> int:
    """作用：计算状态空间词表大小。输入：普通词表大小。输出：扩展词表大小。"""
    return real_vocab_size + 4


PAD_TOKEN = 0
BOS_TOKEN = 1
GAP_TOKEN = 2
UNK_TOKEN = 3
