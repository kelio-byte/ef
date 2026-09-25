import re

dot_bracket_regex = re.compile("(\\(|\\)|\\.|[^\\(\\)\\.]+)")


def inverse_global_align(smiles: str):
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
