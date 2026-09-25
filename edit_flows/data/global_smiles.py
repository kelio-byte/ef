"""用途：将 global 编码的 SMILES 顺序还原为普通 SMILES。
输入：global SMILES 字符串。
输出：还原后的 SMILES 字符串。
"""

import re

dot_bracket_regex = re.compile("(\\(|\\)|\\.|[^\\(\\)\\.]+)")


def inverse_global_align(smiles: str):
    """作用：还原 global 表示中的分支与组分顺序。输入：global SMILES 字符串。输出：普通 SMILES 字符串。"""
    smiles = "." + smiles + ")"
    stack = []
    level = []
    res = []
    for c in dot_bracket_regex.findall(smiles):
        if c == ".":
            stack.append("")
            level.append(0)
            continue
        if c == "(":
            level[-1] += 1
        if c == ")":
            while len(level) != 0 and level[-1] == 0:
                res.append(stack.pop())
                level.pop()
            if len(level) == 0:
                break
            level[-1] -= 1
        if c == ")":
            if stack[-1][-1] == "(":
                stack[-1] = stack[-1][:-1]
            else:
                stack[-1] += c
        else:
            stack[-1] += c
    res.reverse()
    return ".".join(res)
