#!/usr/bin/env python3
"""复核笔记里的手算例子。只使用标准库。

    python3 tools/check_worked_examples.py
"""

from __future__ import annotations

import math
from collections import Counter
from fractions import Fraction


def softmax(logits: list[float]) -> list[float]:
    peak = max(logits)
    weights = [math.exp(x - peak) for x in logits]
    total = sum(weights)
    return [w / total for w in weights]


def log_softmax(logits: list[float]) -> list[float]:
    peak = max(logits)
    log_sum_exp = peak + math.log(sum(math.exp(x - peak) for x in logits))
    return [x - log_sum_exp for x in logits]


def assert_close(actual: float, expected: float, tol: float = 1e-9) -> None:
    if abs(actual - expected) > tol:
        raise AssertionError(f"{actual} != {expected} (tol={tol})")


def check_softmax() -> None:
    logits = [1.0, 2.0, 0.0]
    probs = softmax(logits)
    rounded = tuple(round(p, 4) for p in probs)
    if rounded != (0.2447, 0.6652, 0.0900):
        raise AssertionError(rounded)
    assert_close(sum(probs), 1.0)
    nll = -math.log(probs[1])
    assert_close(nll, -log_softmax(logits)[1])
    if round(nll, 4) != 0.4076:
        raise AssertionError(round(nll, 4))
    if round(math.exp(nll), 4) != 1.5032:
        raise AssertionError(round(math.exp(nll), 4))

    sharp = tuple(round(p, 4) for p in softmax([x / 0.5 for x in logits]))
    flat = tuple(round(p, 4) for p in softmax([x / 2.0 for x in logits]))
    if sharp != (0.1173, 0.8668, 0.0159):
        raise AssertionError(sharp)
    if flat != (0.3072, 0.5065, 0.1863):
        raise AssertionError(flat)

    shifted = [1000.0, 1001.0, 999.0]
    for left, right in zip(softmax(shifted), probs):
        assert_close(left, right, tol=1e-12)
    try:
        math.exp(1000.0)
    except OverflowError:
        pass
    else:
        raise AssertionError("exp(1000) should overflow float range")

    # 每个 logit 加同一个常数，分布不变。
    for left, right in zip(softmax([x + 5.0 for x in logits]), probs):
        assert_close(left, right, tol=1e-12)


def check_perplexity() -> None:
    mean_nll = (math.log(2.0) + math.log(4.0)) / 2.0
    perplexity = math.exp(mean_nll)
    assert_close(perplexity, math.sqrt(8.0))
    assert_close(math.exp(-math.log(1.0 / 3.0)), 3.0)
    assert_close(math.exp(0.0), 1.0)


def check_least_squares() -> None:
    # 拟合 y = a + b x，点为 (0, 1), (1, 2), (2, 2)。
    a = Fraction(7, 6)
    b = Fraction(1, 2)
    design = ((1, 0), (1, 1), (1, 2))
    observed = (1, 2, 2)
    preds = [a + b * x for _, x in design]
    residuals = [Fraction(y) - pred for y, pred in zip(observed, preds)]
    if preds != [Fraction(7, 6), Fraction(5, 3), Fraction(13, 6)]:
        raise AssertionError(preds)
    if residuals != [Fraction(-1, 6), Fraction(1, 3), Fraction(-1, 6)]:
        raise AssertionError(residuals)
    col_const = (1, 1, 1)
    col_x = (0, 1, 2)
    if sum(r * c for r, c in zip(residuals, col_const)) != 0:
        raise AssertionError("residual not orthogonal to the constant column")
    if sum(r * c for r, c in zip(residuals, col_x)) != 0:
        raise AssertionError("residual not orthogonal to the x column")


def check_eigen_and_condition() -> None:
    def matvec(matrix: tuple[tuple[int, int], tuple[int, int]], vector: tuple[int, int]) -> tuple[int, int]:
        return (
            matrix[0][0] * vector[0] + matrix[0][1] * vector[1],
            matrix[1][0] * vector[0] + matrix[1][1] * vector[1],
        )

    matrix = ((2, 1), (1, 2))
    if matvec(matrix, (1, 1)) != (3, 3):
        raise AssertionError("eigenvalue 3")
    if matvec(matrix, (1, -1)) != (1, -1):
        raise AssertionError("eigenvalue 1")
    gram = ((5, 4), (4, 5))  # A^T A
    if matvec(gram, (1, 1)) != (9, 9):
        raise AssertionError("eigenvalue 9")
    if matvec(gram, (1, -1)) != (1, -1):
        raise AssertionError("eigenvalue 1 of gram matrix")


def check_rank_one_and_lora() -> None:
    # A = u v^T, u = (1, 2), v = (3, 4)
    matrix = ((3, 4), (6, 8))
    null = (4, -3)
    image = (
        matrix[0][0] * null[0] + matrix[0][1] * null[1],
        matrix[1][0] * null[0] + matrix[1][1] * null[1],
    )
    if image != (0, 0):
        raise AssertionError(image)
    full = 4096 * 4096
    low_rank = 8 * (4096 + 4096)
    if (full, low_rank, Fraction(low_rank, full)) != (16_777_216, 65_536, Fraction(1, 256)):
        raise AssertionError((full, low_rank))


def check_logistic() -> None:
    def sigmoid(value: float) -> float:
        return 1.0 / (1.0 + math.exp(-value))

    probability = sigmoid(0.0)
    assert_close(probability, 0.5)
    y_true = 1.0
    grad_logit = probability - y_true
    assert_close(grad_logit, -0.5)
    learning_rate = 0.1
    weight = 0.0 - learning_rate * grad_logit * 1.0
    bias = 0.0 - learning_rate * grad_logit
    assert_close(weight, 0.05)
    assert_close(bias, 0.05)
    updated = sigmoid(weight * 1.0 + bias)
    if round(updated, 4) != 0.5250:
        raise AssertionError(round(updated, 4))

    def loss_at(logit: float) -> float:
        prob = sigmoid(logit)
        return -math.log(prob)  # y = 1

    eps = 1e-6
    finite = (loss_at(eps) - loss_at(0.0)) / eps
    assert_close(finite, -0.5, tol=1e-4)


def check_backprop() -> None:
    x = 2.0
    target = 1.0

    def forward(w1: float, b1: float, w2: float, b2: float) -> tuple[float, float, float, float]:
        z1 = w1 * x + b1
        hidden = z1 if z1 > 0.0 else 0.0
        y_hat = w2 * hidden + b2
        loss = 0.5 * (y_hat - target) ** 2
        return z1, hidden, y_hat, loss

    w1, b1, w2, b2 = 0.5, 0.0, -1.5, 0.0
    z1, hidden, y_hat, loss = forward(w1, b1, w2, b2)
    assert_close(z1, 1.0)
    assert_close(hidden, 1.0)
    assert_close(y_hat, -1.5)
    assert_close(loss, 3.125)
    dloss_dy = y_hat - target
    dloss_dw2 = dloss_dy * hidden
    dloss_db2 = dloss_dy
    dloss_dhidden = dloss_dy * w2
    dloss_dz1 = dloss_dhidden * (1.0 if z1 > 0.0 else 0.0)
    dloss_dw1 = dloss_dz1 * x
    dloss_db1 = dloss_dz1
    assert_close(dloss_dy, -2.5)
    assert_close(dloss_dw2, -2.5)
    assert_close(dloss_db2, -2.5)
    assert_close(dloss_dhidden, 3.75)
    assert_close(dloss_dz1, 3.75)
    assert_close(dloss_dw1, 7.5)
    assert_close(dloss_db1, 3.75)

    eps = 1e-6
    base = forward(w1, b1, w2, b2)[3]
    assert_close((forward(w1 + eps, b1, w2, b2)[3] - base) / eps, dloss_dw1, tol=1e-4)
    assert_close((forward(w1, b1 + eps, w2, b2)[3] - base) / eps, dloss_db1, tol=1e-4)
    assert_close((forward(w1, b1, w2 + eps, b2)[3] - base) / eps, dloss_dw2, tol=1e-4)
    assert_close((forward(w1, b1, w2, b2 + eps)[3] - base) / eps, dloss_db2, tol=1e-4)

    dead_w1 = -0.5
    dead_z, dead_h, _, _ = forward(dead_w1, b1, w2, b2)
    if not (dead_z < 0.0 and dead_h == 0.0):
        raise AssertionError("expected a dead ReLU")
    dead_base = forward(dead_w1, b1, w2, b2)[3]
    dead_fd = (forward(dead_w1 + eps, b1, w2, b2)[3] - dead_base) / eps
    assert_close(dead_fd, 0.0, tol=1e-8)


def bpe_pair_counts(corpus: list[tuple[list[str], int]]) -> Counter[tuple[str, str]]:
    counts: Counter[tuple[str, str]] = Counter()
    for symbols, freq in corpus:
        for left, right in zip(symbols, symbols[1:]):
            counts[(left, right)] += freq
    return counts


def bpe_merge(
    corpus: list[tuple[list[str], int]], pair: tuple[str, str]
) -> list[tuple[list[str], int]]:
    left, right = pair
    merged = left + right
    updated: list[tuple[list[str], int]] = []
    for symbols, freq in corpus:
        pieces: list[str] = []
        index = 0
        while index < len(symbols):
            if (
                index + 1 < len(symbols)
                and symbols[index] == left
                and symbols[index + 1] == right
            ):
                pieces.append(merged)
                index += 2
            else:
                pieces.append(symbols[index])
                index += 1
        updated.append((pieces, freq))
    return updated


def check_bpe() -> None:
    corpus = [
        (list("cat"), 5),
        (list("car"), 4),
        (list("cap"), 2),
        (list("map"), 3),
    ]
    first = bpe_pair_counts(corpus)
    if first[("c", "a")] != 11 or first[("a", "t")] != 5 or first[("a", "p")] != 5:
        raise AssertionError(first)
    if max(first.values()) != 11:
        raise AssertionError("first merge is not unique in this note")
    corpus = bpe_merge(corpus, ("c", "a"))
    second = bpe_pair_counts(corpus)
    if second[("ca", "t")] != 5 or max(second.values()) != 5:
        raise AssertionError(second)
    corpus = bpe_merge(corpus, ("ca", "t"))
    third = bpe_pair_counts(corpus)
    if third[("ca", "r")] != 4 or max(third.values()) != 4:
        raise AssertionError(third)
    corpus = bpe_merge(corpus, ("ca", "r"))
    # 用符号列表而不是把符号拼回字符串。拼回去会把 ["ca", "p"] 和 ["cap"] 看成同一个词。
    pieces = {(tuple(symbols), freq) for symbols, freq in corpus}
    expected = {
        (("cat",), 5),
        (("car",), 4),
        (("ca", "p"), 2),
        (("m", "a", "p"), 3),
    }
    if pieces != expected:
        raise AssertionError(pieces)


def two_sum_sorted(nums: list[int], target: int) -> tuple[int, int] | None:
    """有序数组上返回任意一对下标 i < j，使 nums[i] + nums[j] == target。"""
    left, right = 0, len(nums) - 1
    while left < right:
        total = nums[left] + nums[right]
        if total == target:
            return left, right
        if total < target:
            left += 1
        else:
            right -= 1
    return None


def lower_bound(nums: list[int], target: int) -> int:
    """升序数组中第一个 >= target 的下标；不存在则返回 len(nums)。"""
    lo, hi = 0, len(nums)
    while lo < hi:
        mid = (lo + hi) // 2
        if nums[mid] < target:
            lo = mid + 1
        else:
            hi = mid
    return lo


def find_buggy(nums: list[int], target: int) -> int:
    lo, hi = 0, len(nums) - 1
    while lo < hi:
        mid = (lo + hi) // 2
        if nums[mid] == target:
            return mid
        if nums[mid] < target:
            lo = mid + 1
        else:
            hi = mid - 1
    return -1


def find_fixed(nums: list[int], target: int) -> int:
    lo, hi = 0, len(nums) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        if nums[mid] == target:
            return mid
        if nums[mid] < target:
            lo = mid + 1
        else:
            hi = mid - 1
    return -1


def check_algorithms() -> None:
    if two_sum_sorted([1, 2, 3, 4], 7) != (2, 3):
        raise AssertionError("two sum 3+4")
    # -3 + 4 = 1，两端先相遇；-1 + 2 也是合法对，函数只保证任意一对。
    if two_sum_sorted([-3, -1, 0, 2, 4], 1) != (0, 4):
        raise AssertionError("two sum negatives")
    if two_sum_sorted([1, 1], 2) != (0, 1):
        raise AssertionError("two sum duplicates")
    if two_sum_sorted([1, 2], 4) is not None:
        raise AssertionError("two sum missing")
    if two_sum_sorted([1], 1) is not None:
        raise AssertionError("two sum one element")

    nums = [1, 3, 3, 5]
    expected = {0: 0, 1: 0, 3: 1, 4: 3, 5: 3, 6: 4}
    for target, index in expected.items():
        if lower_bound(nums, target) != index:
            raise AssertionError((target, lower_bound(nums, target)))

    if find_buggy([1], 1) != -1:
        raise AssertionError("buggy search should miss the only element")
    if find_buggy([1, 3], 3) != -1:
        raise AssertionError("buggy search should miss the right end")
    if find_fixed([1], 1) != 0 or find_fixed([1, 3], 3) != 1:
        raise AssertionError("fixed search")
    if find_fixed([1, 3, 3, 5], 4) != -1:
        raise AssertionError("fixed search missing")


def mean_nll_copy_current(
    probs: list[list[float]], tokens: list[int], pad_id: int
) -> float:
    """错误实现：用位置 t 的分布去拟合 tokens[t] 自己。"""
    total = 0.0
    count = 0
    for index, token in enumerate(tokens):
        if token == pad_id:
            continue
        total += -math.log(probs[index][token])
        count += 1
    return total / count


def mean_nll_next_token(
    probs: list[list[float]], tokens: list[int], pad_id: int
) -> float:
    """约定 A：位置 t 的分布预测 tokens[t + 1]，填充目标不计入。"""
    total = 0.0
    count = 0
    for index in range(len(tokens) - 1):
        target = tokens[index + 1]
        if target == pad_id:
            continue
        total += -math.log(probs[index][target])
        count += 1
    if count == 0:
        raise ValueError("no supervised tokens")
    return total / count


def sft_labels(tokens: list[int], prompt_len: int, pad_id: int) -> list[int | None]:
    """约定 A 下的监督微调标签。提示词对应的预测目标和填充都不计损失。"""
    labels: list[int | None] = []
    for offset, token in enumerate(tokens[1:]):
        target_index = offset + 1
        if token == pad_id or target_index < prompt_len:
            labels.append(None)
        else:
            labels.append(token)
    return labels


def check_training_step() -> None:
    # 词表大小 8。每个位置把 0.7 放在当前 token 上，把 0.1 放在下一个 token 上。
    tokens = [1, 5, 7, 0]
    pad_id = 0
    vocab = 8
    probs: list[list[float]] = []
    for index, token in enumerate(tokens):
        row = [0.0] * vocab
        nxt = tokens[index + 1] if index + 1 < len(tokens) else None
        row[token] = 0.7
        if nxt is not None and nxt != token:
            row[nxt] = 0.1
            share = 0.2 / (vocab - 2)
        else:
            share = 0.3 / (vocab - 1)
        for vocab_id in range(vocab):
            if row[vocab_id] == 0.0:
                row[vocab_id] = share
        probs.append(row)
        assert_close(sum(row), 1.0, tol=1e-12)

    buggy = mean_nll_copy_current(probs, tokens, pad_id)
    correct = mean_nll_next_token(probs, tokens, pad_id)
    assert_close(buggy, -math.log(0.7))
    assert_close(correct, -math.log(0.1))
    if round(buggy, 4) != 0.3567 or round(correct, 4) != 2.3026:
        raise AssertionError((round(buggy, 4), round(correct, 4)))

    labels = sft_labels([1, 8, 9, 2, 0, 0], prompt_len=2, pad_id=0)
    if labels != [None, 9, 2, None, None]:
        raise AssertionError(labels)


def check_classification_counts() -> None:
    true_positive, false_positive, false_negative, true_negative = 40, 10, 20, 30
    precision = Fraction(true_positive, true_positive + false_positive)
    recall = Fraction(true_positive, true_positive + false_negative)
    accuracy = Fraction(
        true_positive + true_negative,
        true_positive + false_positive + false_negative + true_negative,
    )
    if (precision, recall, accuracy) != (Fraction(4, 5), Fraction(2, 3), Fraction(7, 10)):
        raise AssertionError((precision, recall, accuracy))


def main() -> None:
    check_softmax()
    check_perplexity()
    check_least_squares()
    check_eigen_and_condition()
    check_rank_one_and_lora()
    check_logistic()
    check_backprop()
    check_bpe()
    check_algorithms()
    check_training_step()
    check_classification_counts()
    print("ok")


if __name__ == "__main__":
    main()
