"""Transforms of RNA data"""
import torch as tr

# All possible matching brackets for base pairing
MATCHING_BRACKETS = [
    ["(", ")"],
    ["[", "]"],
    ["{", "}"],
    ["<", ">"],
    ["A", "a"],
    ["B", "a"],
]


def bp2matrix(L, base_pairs):
    matrix = tr.zeros((L, L))

    for bp in base_pairs:
        # base pairs are 1-based
        matrix[bp[0] - 1, bp[1] - 1] = 1
        matrix[bp[1] - 1, bp[0] - 1] = 1
    return matrix


def fold2bp(struc, xop="(", xcl=")"):
    """Get base pairs from one page folding (using only one type of brackets).
    BP are 1-indexed"""
    openxs = []
    bps = []
    if struc.count(xop) != struc.count(xcl):
        return False
    for i, x in enumerate(struc):
        if x == xop:
            openxs.append(i)
        elif x == xcl:
            if len(openxs) > 0:
                bps.append([openxs.pop() + 1, i + 1])
            else:
                return False
    return bps

def dot2bp(struc):
    bp = []
    if not set(struc).issubset(
        set(["."] + [c for par in MATCHING_BRACKETS for c in par])
    ):
        return False

    for brackets in MATCHING_BRACKETS:
        if brackets[0] in struc:
            bpk = fold2bp(struc, brackets[0], brackets[1])
            if bpk:
                bp = bp + bpk
            else:
                return False
    return list(sorted(bp))