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


def check_probability() -> None:
    joints = (
        Fraction(1, 2) * Fraction(3, 4),
        Fraction(1, 2) * Fraction(1, 4),
        Fraction(1, 2) * Fraction(1, 4),
        Fraction(1, 2) * Fraction(3, 4),
    )
    if joints != (Fraction(3, 8), Fraction(1, 8), Fraction(1, 8), Fraction(3, 8)):
        raise AssertionError(joints)
    if sum(joints) != 1:
        raise AssertionError(sum(joints))

    p_x2_a = Fraction(3, 4) * Fraction(1, 2) + Fraction(1, 4) * Fraction(1, 2)
    if p_x2_a != Fraction(1, 2):
        raise AssertionError(p_x2_a)
    if joints[0] / p_x2_a != Fraction(3, 4):
        raise AssertionError("bayes")
    if joints[3] == Fraction(1, 2) * Fraction(1, 2):
        raise AssertionError("joint should not factor")

    # X = 1 for token B. Var(X1+X2) uses the joint, not the sum of variances.
    probs = {
        (0, 0): Fraction(3, 8),
        (0, 1): Fraction(1, 8),
        (1, 0): Fraction(1, 8),
        (1, 1): Fraction(3, 8),
    }
    mean_prod = sum(x * y * p for (x, y), p in probs.items())
    if mean_prod != Fraction(3, 8):
        raise AssertionError(mean_prod)
    cov = mean_prod - Fraction(1, 2) * Fraction(1, 2)
    if cov != Fraction(1, 8):
        raise AssertionError(cov)
    var_sum = Fraction(1, 4) + Fraction(1, 4) + 2 * cov
    if var_sum != Fraction(3, 4):
        raise AssertionError(var_sum)
    second = sum((x + y) ** 2 * p for (x, y), p in probs.items())
    if second - 1 != var_sum:
        raise AssertionError(second)

    p_hat = Fraction(3, 4)
    if p_hat != Fraction(3, 4):
        raise AssertionError(p_hat)
    nll_sum = -(3 * math.log(3 / 4) + math.log(1 / 4))
    assert_close(nll_sum, 4 * math.log(4) - 3 * math.log(3))
    if round(nll_sum, 4) != 2.2493:
        raise AssertionError(round(nll_sum, 4))

    counts = (2, 1, 1)
    total = sum(counts)
    mle = tuple(Fraction(c, total) for c in counts)
    if mle != (Fraction(1, 2), Fraction(1, 4), Fraction(1, 4)):
        raise AssertionError(mle)
    mean_nll = -sum(
        (c / total) * math.log(c / total) for c in counts
    )
    assert_close(mean_nll, 1.5 * math.log(2))
    if round(mean_nll, 4) != 1.0397:
        raise AssertionError(round(mean_nll, 4))
    uniform_ce = math.log(3)
    if round(uniform_ce, 4) != 1.0986:
        raise AssertionError(round(uniform_ce, 4))
    assert_close(uniform_ce - mean_nll, math.log(3) - 1.5 * math.log(2))
    if uniform_ce <= mean_nll:
        raise AssertionError("uniform cross entropy should be larger")

    data = (1, 2, 3)
    mean = Fraction(sum(data), len(data))
    sq = sum((Fraction(x) - mean) ** 2 for x in data)
    if (mean, sq, sq / 3, sq / 2) != (2, 2, Fraction(2, 3), 1):
        raise AssertionError((mean, sq))


def check_optimizer() -> None:
    # L = w^2 / 2, gradient is w, w <- (1-lr) w.
    def trajectory(lr: Fraction, steps: int = 4) -> tuple[Fraction, ...]:
        w = Fraction(1)
        values = [w]
        for _ in range(steps):
            w = (1 - lr) * w
            values.append(w)
        return tuple(values)

    if trajectory(Fraction(1, 2)) != (1, Fraction(1, 2), Fraction(1, 4), Fraction(1, 8), Fraction(1, 16)):
        raise AssertionError(trajectory(Fraction(1, 2)))
    if trajectory(2) != (1, -1, 1, -1, 1):
        raise AssertionError(trajectory(2))
    if trajectory(3) != (1, -2, 4, -8, 16):
        raise AssertionError(trajectory(3))

    # Mean reduction vs sum reduction.
    mean_step = 0.0 - 0.1 * ((1 + 3) / 2)
    sum_step = 0.0 - 0.1 * (1 + 3)
    assert_close(mean_step, -0.2)
    assert_close(sum_step, -0.4)

    w = Fraction(1)
    v = Fraction(0)
    lr = Fraction(1, 10)
    mu = Fraction(9, 10)
    expected = (
        (Fraction(1), Fraction(9, 10)),
        (Fraction(9, 5), Fraction(18, 25)),
        (Fraction(117, 50), Fraction(243, 500)),
    )
    for step, (v_hat, w_hat) in enumerate(expected, start=1):
        g = w
        v = mu * v + g
        w = w - lr * v
        if (v, w) != (v_hat, w_hat):
            raise AssertionError((step, v, w))
    # Drop the velocity after step 2 and take one more step from w = 18/25.
    w_reset = Fraction(18, 25) - lr * Fraction(18, 25)
    if w_reset != Fraction(81, 125):
        raise AssertionError(w_reset)

    plain = Fraction(1)
    for _ in range(3):
        plain = plain - lr * plain
    if plain != Fraction(729, 1000):
        raise AssertionError(plain)

    beta1, beta2 = 0.9, 0.999
    g = 0.5
    m = (1 - beta1) * g
    v_mom = (1 - beta2) * g * g
    m_hat = m / (1 - beta1)
    v_hat = v_mom / (1 - beta2)
    assert_close(m, 0.05)
    assert_close(v_mom, 0.00025)
    assert_close(m_hat, 0.5)
    assert_close(v_hat, 0.25)
    assert_close(m_hat / math.sqrt(v_hat), 1.0)
    uncorrected = m / math.sqrt(v_mom)
    assert_close(uncorrected, math.sqrt(10))
    assert_close(0.001 * uncorrected, 0.001 * math.sqrt(10))

    m2 = beta1 * m + (1 - beta1) * g
    v2 = beta2 * v_mom + (1 - beta2) * g * g
    assert_close(m2 / (1 - beta1**2), 0.5)
    assert_close(v2 / (1 - beta2**2), 0.25)

    def adamw(theta, grad, lam, eps):
        updated = []
        for th, g_i in zip(theta, grad):
            decayed = th * (1 - 0.001 * lam)
            if g_i == 0.0:
                adapt = 0.0
            else:
                m_i = (1 - beta1) * g_i
                v_i = (1 - beta2) * g_i * g_i
                m_h = m_i / (1 - beta1)
                v_h = v_i / (1 - beta2)
                adapt = m_h / (math.sqrt(v_h) + eps)
            updated.append(decayed - 0.001 * adapt)
        return updated

    def coupled(theta, grad, lam, eps):
        updated = []
        for th, g_i in zip(theta, grad):
            g_p = g_i + lam * th
            m_i = (1 - beta1) * g_p
            v_i = (1 - beta2) * g_p * g_p
            m_h = m_i / (1 - beta1)
            v_h = v_i / (1 - beta2)
            adapt = m_h / (math.sqrt(v_h) + eps)
            updated.append(th - 0.001 * adapt)
        return updated

    exact = adamw((1.0, 1.0), (0.5, 0.0), 0.01, 0.0)
    assert_close(exact[0], 0.99899)
    assert_close(exact[1], 0.99999)
    coupled_exact = coupled((1.0, 1.0), (0.5, 0.0), 0.01, 0.0)
    assert_close(coupled_exact[0], 0.999)
    assert_close(coupled_exact[1], 0.999)
    tiny = adamw((1.0, 1.0), (0.5, 0.0), 0.01, 1e-8)
    assert_close(tiny[1], 0.99999)
    assert_close(tiny[0], 0.99899, tol=1e-8)

    def cosine_multiplier(step: int, warmup: int = 4, total: int = 8) -> float:
        if step < warmup:
            return step / warmup
        progress = (step - warmup) / (total - warmup)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    assert_close(cosine_multiplier(0), 0.0)
    assert_close(cosine_multiplier(1), 0.25)
    assert_close(cosine_multiplier(4), 1.0)
    assert_close(cosine_multiplier(5), 0.5 + math.sqrt(2) / 4)
    assert_close(cosine_multiplier(6), 0.5)
    assert_close(cosine_multiplier(8), 0.0)
    assert_close(cosine_multiplier(4) * 0.001, 0.001)
    assert_close(cosine_multiplier(6) * 0.001, 0.0005)

    def nll_double_well(value: float) -> float:
        return value**4 / 4 - value**2 / 2

    def grad_double_well(value: float) -> float:
        return value**3 - value

    start = 0.5
    nxt = start - 0.1 * grad_double_well(start)
    assert_close(grad_double_well(start), -0.375)
    assert_close(nxt, 0.5375)
    assert_close(nll_double_well(start), -0.109375)
    assert_close(nll_double_well(1.0), -0.25)
    assert_close(nll_double_well(-1.0), -0.25)
    if not (nll_double_well(nxt) < nll_double_well(start) < 0.0):
        raise AssertionError("loss should fall while staying above the minima")
    other = -0.5 - 0.1 * grad_double_well(-0.5)
    assert_close(other, -0.5375)


def softmax_with_mask(row: list[float]) -> list[float]:
    finite = [x for x in row if x > float("-inf")]
    if not finite:
        raise AssertionError("softmax of an all-masked row")
    peak = max(finite)
    weights = [0.0 if x == float("-inf") else math.exp(x - peak) for x in row]
    total = sum(weights)
    return [w / total for w in weights]


def causal_attention(
    q: list[list[float]], k: list[list[float]], v: list[list[float]]
) -> tuple[list[list[float]], list[list[float]]]:
    scale = math.sqrt(len(q[0]))
    weights: list[list[float]] = []
    outputs: list[list[float]] = []
    for i, qi in enumerate(q):
        scores = []
        for j, kj in enumerate(k):
            dot = sum(a * b for a, b in zip(qi, kj))
            scores.append(float("-inf") if j > i else dot / scale)
        row = softmax_with_mask(scores)
        mixed = [0.0] * len(v[0])
        for weight, vj in zip(row, v):
            for d, value in enumerate(vj):
                mixed[d] += weight * value
        weights.append(row)
        outputs.append(mixed)
    return weights, outputs


def check_attention() -> None:
    q = [
        [0.0, 2.0, 0.0, 0.0],
        [1.0, 1.0, 0.0, 0.0],
        [1.0, 2.0, 0.0, 0.0],
    ]
    k = [
        [2.0, 0.0, 0.0, 0.0],
        [0.0, 2.0, 0.0, 0.0],
        [0.0, 0.0, 2.0, 0.0],
    ]
    v = [
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0],
    ]
    weights, outputs = causal_attention(q, k, v)
    if weights[0] != [1.0, 0.0, 0.0]:
        raise AssertionError(weights[0])
    if weights[1] != [0.5, 0.5, 0.0]:
        raise AssertionError(weights[1])
    third = softmax([1.0, 2.0, 0.0])
    for left, right in zip(weights[2], third):
        assert_close(left, right)
    if tuple(round(p, 4) for p in third) != (0.2447, 0.6652, 0.0900):
        raise AssertionError(tuple(round(p, 4) for p in third))
    assert_close(outputs[0][0], 1.0)
    assert_close(outputs[0][1], 0.0)
    assert_close(outputs[1][0], 0.5)
    assert_close(outputs[1][1], 0.5)
    rounded_out = tuple(round(x, 4) for x in outputs[2])
    if rounded_out != (0.3348, 0.7553):
        raise AssertionError(rounded_out)
    # Rounded weights times V differ in the fourth decimal.
    rough0 = 0.2447 * 1 + 0.6652 * 0 + 0.0900 * 1
    rough1 = 0.2447 * 0 + 0.6652 * 1 + 0.0900 * 1
    if (round(rough0, 4), round(rough1, 4)) != (0.3347, 0.7552):
        raise AssertionError((rough0, rough1))

    unmasked = softmax([0.0, 2.0, 0.0])
    if tuple(round(p, 4) for p in unmasked) != (0.1065, 0.7870, 0.1065):
        raise AssertionError(tuple(round(p, 4) for p in unmasked))
    unmasked_out = (
        unmasked[0] * 1 + unmasked[2] * 1,
        unmasked[1] * 1 + unmasked[2] * 1,
    )
    if tuple(round(x, 4) for x in unmasked_out) != (0.2130, 0.8935):
        raise AssertionError(tuple(round(x, 4) for x in unmasked_out))

    zeroed = softmax([0.0, 0.0, 0.0])
    for weight in zeroed:
        assert_close(weight, 1 / 3)
    if (1 + 0 + 1) / 3 != 2 / 3 or (0 + 1 + 1) / 3 != 2 / 3:
        raise AssertionError("masked-as-zero output")

    # Two heads, all scores 0, W_O = I keeps the concatenation.
    head1 = ((1.0, 0.0), (0.5, 0.5))
    head2 = ((0.0, 1.0), (0.5, 0.5))
    concat = (
        head1[0] + head2[0],
        head1[1] + head2[1],
    )
    if concat != ((1.0, 0.0, 0.0, 1.0), (0.5, 0.5, 0.5, 0.5)):
        raise AssertionError(concat)
    w_o = (
        (1.0, 0.0, 1.0, 0.0),
        (0.0, 1.0, 0.0, 1.0),
        (0.0, 0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0, 0.0),
    )
    projected = tuple(
        tuple(sum(c * w for c, w in zip(row, basis)) for basis in w_o) for row in concat
    )
    if projected != ((1.0, 1.0, 0.0, 0.0), (1.0, 1.0, 0.0, 0.0)):
        raise AssertionError(projected)

    # Bidirectional, no positions: swapping inputs swaps outputs.
    s = 1 / math.sqrt(2)
    left = softmax([s, 0.0])
    if tuple(round(p, 4) for p in left) != (0.6698, 0.3302):
        raise AssertionError(tuple(round(p, 4) for p in left))
    ab0 = (left[0], left[1])
    ab1 = (left[1], left[0])
    if tuple(round(x, 4) for x in ab0) != (0.6698, 0.3302):
        raise AssertionError(ab0)
    if tuple(round(x, 4) for x in ab1) != (0.3302, 0.6698):
        raise AssertionError(ab1)

    if 10000 ** (2 / 4) != 100:
        raise AssertionError("frequency base")
    pe0 = (math.sin(0), math.cos(0), math.sin(0), math.cos(0.0 / 100))
    if pe0 != (0.0, 1.0, 0.0, 1.0):
        raise AssertionError(pe0)
    added = tuple(e + p for e, p in zip((1, 2, 3, 4), pe0))
    if added != (1, 3, 3, 5):
        raise AssertionError(added)
    pe1 = (math.sin(1), math.cos(1), math.sin(0.01), math.cos(0.01))
    if tuple(round(x, 4) for x in pe1[:2]) != (0.8415, 0.5403):
        raise AssertionError(pe1)
    if round(pe1[2], 4) != 0.0100:
        raise AssertionError(round(pe1[2], 4))
    if round(pe1[3], 6) != 0.999950:
        raise AssertionError(round(pe1[3], 6))
    pe2 = (math.sin(2), math.cos(2))
    rotated = (
        math.cos(1) * pe1[0] + math.sin(1) * pe1[1],
        -math.sin(1) * pe1[0] + math.cos(1) * pe1[1],
    )
    assert_close(rotated[0], pe2[0])
    assert_close(rotated[1], pe2[1])
    if tuple(round(x, 4) for x in pe2) != (0.9093, -0.4161):
        raise AssertionError(tuple(round(x, 4) for x in pe2))

    full = 4 * 512 * 512
    with_bias = full + 4 * 512
    if (full, with_bias) != (1_048_576, 1_050_624):
        raise AssertionError((full, with_bias))
    if 4 * 4 * 4 != 64 or 64 + 4 * 4 != 80:
        raise AssertionError("d_model 4 attention params")
    if 8 * 512 * 512 != 2_097_152:
        raise AssertionError("ffn params")
    if 1024**2 != 1_048_576 or 2048**2 != 4 * 1024**2:
        raise AssertionError("score matrix growth")

    # Pre-norm vs post-norm with a constant sublayer (1/2, -1/2).
    pre = (1 + 0.5, 3 - 0.5)
    if pre != (1.5, 2.5):
        raise AssertionError(pre)
    mean = sum(pre) / 2
    var = sum((x - mean) ** 2 for x in pre) / 2
    post = tuple((x - mean) / math.sqrt(var) for x in pre)
    assert_close(post[0], -1.0)
    assert_close(post[1], 1.0)

    # FFN: y = x @ W.T + b, one ReLU unit is off.
    x_ffn = (1.0, -2.0)
    w1 = ((1.0, 1.0), (1.0, -1.0))
    pre = tuple(sum(x_ffn[k] * row[k] for k in range(2)) for row in w1)
    if pre != (-1.0, 3.0):
        raise AssertionError(pre)
    relu = tuple(h if h > 0.0 else 0.0 for h in pre)
    if relu != (0.0, 3.0):
        raise AssertionError(relu)
    w2 = ((2.0, 1.0), (0.0, 1.0))
    ffn = tuple(sum(relu[k] * row[k] for k in range(2)) for row in w2)
    if ffn != (3.0, 3.0):
        raise AssertionError(ffn)
    residual = (x_ffn[0] + ffn[0], x_ffn[1] + ffn[1])
    if residual != (4.0, 1.0):
        raise AssertionError(residual)


def heap_push(heap: list[int], value: int) -> None:
    heap.append(value)
    i = len(heap) - 1
    while i > 0:
        parent = (i - 1) // 2
        if heap[parent] <= heap[i]:
            break
        heap[parent], heap[i] = heap[i], heap[parent]
        i = parent


def heap_pop(heap: list[int]) -> int:
    if len(heap) == 1:
        return heap.pop()
    top = heap[0]
    heap[0] = heap.pop()
    i = 0
    n = len(heap)
    while True:
        left, right = 2 * i + 1, 2 * i + 2
        smallest = i
        if left < n and heap[left] < heap[smallest]:
            smallest = left
        if right < n and heap[right] < heap[smallest]:
            smallest = right
        if smallest == i:
            break
        heap[i], heap[smallest] = heap[smallest], heap[i]
        i = smallest
    return top


def two_sum(nums: list[int], target: int) -> tuple[int, int] | None:
    seen: dict[int, int] = {}
    for j, value in enumerate(nums):
        i = seen.get(target - value)
        if i is not None:
            return i, j
        seen[value] = j
    return None


def sorted_squares(nums: list[int]) -> list[int]:
    left, right = 0, len(nums) - 1
    out = [0] * len(nums)
    for pos in range(len(nums) - 1, -1, -1):
        if abs(nums[left]) >= abs(nums[right]):
            out[pos] = nums[left] * nums[left]
            left += 1
        else:
            out[pos] = nums[right] * nums[right]
            right -= 1
    return out


def min_length_at_least(nums: list[int], target: int) -> int:
    best = 0
    left = 0
    total = 0
    for right, value in enumerate(nums):
        total += value
        while total >= target and left <= right:
            length = right - left + 1
            if best == 0 or length < best:
                best = length
            total -= nums[left]
            left += 1
    return best


def count_subarrays_sum(nums: list[int], k: int) -> int:
    seen = {0: 1}
    prefix = 0
    count = 0
    for value in nums:
        prefix += value
        count += seen.get(prefix - k, 0)
        seen[prefix] = seen.get(prefix, 0) + 1
    return count


def count_subarrays_sum_without_zero(nums: list[int], k: int) -> int:
    seen: dict[int, int] = {}
    prefix = 0
    count = 0
    for value in nums:
        prefix += value
        count += seen.get(prefix - k, 0)
        seen[prefix] = seen.get(prefix, 0) + 1
    return count


def feasible_parts(nums: list[int], parts_limit: int, cap: int) -> bool:
    parts = 1
    total = 0
    for value in nums:
        if value > cap:
            return False
        if total + value > cap:
            parts += 1
            total = value
        else:
            total += value
    return parts <= parts_limit


def min_max_sum(nums: list[int], parts_limit: int) -> int:
    lo, hi = max(nums), sum(nums)
    while lo < hi:
        mid = (lo + hi) // 2
        if feasible_parts(nums, parts_limit, mid):
            hi = mid
        else:
            lo = mid + 1
    return lo


def min_max_sum_buggy(nums: list[int], parts_limit: int) -> int:
    lo, hi = max(nums), sum(nums)
    while lo < hi:
        mid = (lo + hi) // 2
        if feasible_parts(nums, parts_limit, mid):
            hi = mid - 1
        else:
            lo = mid + 1
    return lo


def next_greater(nums: list[int]) -> list[int | None]:
    answer: list[int | None] = [None] * len(nums)
    stack: list[int] = []
    for i, value in enumerate(nums):
        while stack and nums[stack[-1]] < value:
            answer[stack.pop()] = value
        stack.append(i)
    return answer


def check_structures() -> None:
    if two_sum([2, 7, 11, 15], 9) != (0, 1):
        raise AssertionError("two sum hash")
    if two_sum([3, 3], 6) != (0, 1):
        raise AssertionError("two sum duplicate")
    if two_sum([3], 6) is not None:
        raise AssertionError("two sum self")
    buckets = [[] for _ in range(3)]
    for key in (3, 6, 9):
        buckets[key % 3].append(key)
    if buckets != [[3, 6, 9], [], []]:
        raise AssertionError(buckets)

    heap: list[int] = []
    snapshots = []
    for value in (4, 1, 3, 2):
        heap_push(heap, value)
        snapshots.append(list(heap))
    if snapshots != [[4], [1, 4], [1, 4, 3], [1, 2, 3, 4]]:
        raise AssertionError(snapshots)
    popped = (heap_pop(heap), heap_pop(heap))
    if popped != (1, 2) or heap != [3, 4]:
        raise AssertionError((popped, heap))

    top: list[int] = []
    for value in (5, 1, 4, 2, 3):
        if len(top) < 2:
            heap_push(top, value)
        elif value > top[0]:
            heap_pop(top)
            heap_push(top, value)
    if sorted(top) != [4, 5]:
        raise AssertionError(top)

    if sorted_squares([-4, -1, 0, 3, 10]) != [0, 1, 9, 16, 100]:
        raise AssertionError(sorted_squares([-4, -1, 0, 3, 10]))
    if min_length_at_least([2, 3, 1, 2, 4, 3], 7) != 2:
        raise AssertionError("window")
    if count_subarrays_sum([1, 1, 1], 2) != 2:
        raise AssertionError("prefix")
    if count_subarrays_sum_without_zero([1, 1, 1], 2) != 1:
        raise AssertionError("missing empty prefix")
    prefix = [0]
    for value in (1, 2, 3, 4):
        prefix.append(prefix[-1] + value)
    if prefix != [0, 1, 3, 6, 10] or prefix[4] - prefix[1] != 9:
        raise AssertionError(prefix)

    nums = [7, 2, 5, 10, 8]
    if not feasible_parts(nums, 2, 18) or feasible_parts(nums, 2, 17):
        raise AssertionError("cap 18/17")
    if min_max_sum(nums, 2) != 18 or min_max_sum_buggy(nums, 2) != 17:
        raise AssertionError((min_max_sum(nums, 2), min_max_sum_buggy(nums, 2)))
    if min_max_sum(nums, 1) != 32 or min_max_sum(nums, 5) != 10:
        raise AssertionError("edge splits")
    if next_greater([2, 1, 2, 4, 3]) != [4, 2, 4, None, None]:
        raise AssertionError(next_greater([2, 1, 2, 4, 3]))

    data: list[int | None] = [None] * 4
    head = tail = size = 0
    cap = 4

    def enqueue(value: int) -> None:
        nonlocal tail, size
        if size == cap:
            raise AssertionError("full")
        data[tail] = value
        tail = (tail + 1) % cap
        size += 1

    def dequeue() -> int | None:
        nonlocal head, size
        value = data[head]
        data[head] = None
        head = (head + 1) % cap
        size -= 1
        return value

    for value in (1, 2, 3):
        enqueue(value)
    if dequeue() != 1:
        raise AssertionError("dequeue")
    enqueue(4)
    enqueue(5)
    if (list(data), head, tail, size) != ([5, 2, 3, 4], 1, 1, 4):
        raise AssertionError((data, head, tail, size))


def check_activation_norm() -> None:
    def sigmoid(value: float) -> float:
        return 1.0 / (1.0 + math.exp(-value))

    assert_close(sigmoid(0.0) * (1.0 - sigmoid(0.0)), 0.25)
    assert_close(0.25**4, 1 / 256)
    sigma5 = sigmoid(5.0)
    deriv5 = sigma5 * (1.0 - sigma5)
    assert_close(sigma5, 0.9933071490757153, tol=1e-12)
    assert_close(deriv5, 0.006648056670790033, tol=1e-12)
    assert_close(deriv5**4, 1.9533445339986894e-09, tol=1e-18)
    assert_close(1.0 - math.tanh(2.0) ** 2, 0.07065082485316443, tol=1e-12)

    silu2 = 2.0 * sigmoid(2.0)
    assert_close(silu2, 1.7615941559557646, tol=1e-12)
    gelu = 0.5 * 2.0 * (1.0 + math.erf(2.0 / math.sqrt(2.0)))
    gelu_tanh = 0.5 * 2.0 * (
        1.0 + math.tanh(math.sqrt(2.0 / math.pi) * (2.0 + 0.044715 * 8.0))
    )
    assert_close(gelu, 1.9544997361036416, tol=1e-12)
    assert_close(gelu_tanh, 1.954597694087775, tol=1e-12)
    if (round(gelu, 4), round(gelu_tanh, 4)) != (1.9545, 1.9546):
        raise AssertionError((round(gelu, 4), round(gelu_tanh, 4)))

    fan_in, fan_out = 4, 2
    xavier = math.sqrt(6 / (fan_in + fan_out))
    he = math.sqrt(6 / fan_in)
    default = 1 / math.sqrt(fan_in)
    assert_close(xavier, 1.0)
    assert_close(he, math.sqrt(1.5))
    assert_close(default, 0.5)
    assert_close(xavier**2 / 3, 1 / 3)
    assert_close(he**2 / 3, 0.5)
    assert_close(default**2 / 3, 1 / 12)
    # a = sqrt(5) leaky-ReLU gain collapses to the Linear default bound.
    gain = math.sqrt(2 / (1 + 5))
    assert_close(gain * math.sqrt(3 / fan_in), default)
    second = 0.5 * fan_in * (1 / (3 * fan_in))
    assert_close(second, 1 / 6)
    assert_close(second**3, 1 / 216)

    def layer_norm(values: tuple[float, ...]) -> tuple[float, ...]:
        n = len(values)
        mu = sum(values) / n
        var = sum((v - mu) ** 2 for v in values) / n
        scale = math.sqrt(var)
        return tuple((v - mu) / scale for v in values)

    def rms_norm(values: tuple[float, ...]) -> tuple[float, ...]:
        rms = math.sqrt(sum(v * v for v in values) / len(values))
        return tuple(v / rms for v in values)

    ln = layer_norm((1.0, 3.0, 5.0))
    root = math.sqrt(1.5)
    assert_close(ln[0], -root)
    assert_close(ln[1], 0.0)
    assert_close(ln[2], root)
    shifted = layer_norm((0.0, 2.0, 4.0))
    for left, right in zip(ln, shifted):
        assert_close(left, right)
    assert_close(1.0 + 2.0 * root, 3.449489742783178, tol=1e-12)
    rms = rms_norm((1.0, 3.0, 5.0))
    scale = math.sqrt(35 / 3)
    assert_close(scale, 3.415650255319866, tol=1e-12)
    for value, coord in zip(rms, (1.0, 3.0, 5.0)):
        assert_close(value, coord / scale)
    if not (ln[0] < 0.0 < rms[0]):
        raise AssertionError("LayerNorm and RMSNorm signs")

    # BatchNorm over two samples, two features.
    feat0 = (1.0, 3.0)
    feat1 = (2.0, 6.0)
    mean0 = sum(feat0) / 2
    mean1 = sum(feat1) / 2
    std0 = math.sqrt(sum((v - mean0) ** 2 for v in feat0) / 2)
    std1 = math.sqrt(sum((v - mean1) ** 2 for v in feat1) / 2)
    bn = (
        ((feat0[0] - mean0) / std0, (feat1[0] - mean1) / std1),
        ((feat0[1] - mean0) / std0, (feat1[1] - mean1) / std1),
    )
    if bn != ((-1.0, -1.0), (1.0, 1.0)):
        raise AssertionError(bn)
    for sample in ((1.0, 2.0), (3.0, 6.0)):
        normalized = layer_norm(sample)
        assert_close(normalized[0], -1.0)
        assert_close(normalized[1], 1.0)

    running = 0.0
    running = 0.9 * running + 0.1 * 2.0
    assert_close(running, 0.2)
    running = 0.9 * running + 0.1 * 6.0
    assert_close(running, 0.78)

    pre = (1.0, 3.0, 5.0)
    post = layer_norm(pre)
    if sum(v * v for v in pre) != 35:
        raise AssertionError("pre length")
    assert_close(sum(v * v for v in post), 3.0)


def check_pretraining() -> None:
    token_nll = (math.log(8.0) + math.log(4.0)) / 4.0
    seq_nll = (math.log(8.0) + math.log(4.0)) / 2.0
    assert_close(token_nll, 1.25 * math.log(2.0))
    assert_close(seq_nll, 2.5 * math.log(2.0))
    assert_close(math.exp(token_nll), 2.0**1.25)
    assert_close(math.exp(seq_nll), 2.0**2.5)
    if round(token_nll, 4) != 0.8664 or round(math.exp(token_nll), 4) != 2.3784:
        raise AssertionError((round(token_nll, 4), round(math.exp(token_nll), 4)))
    if round(seq_nll, 4) != 1.7329 or round(math.exp(seq_nll), 4) != 5.6569:
        raise AssertionError((round(seq_nll, 4), round(math.exp(seq_nll), 4)))

    if Fraction(400, 1000) != Fraction(2, 5) or Fraction(100, 700) != Fraction(1, 7):
        raise AssertionError("dedup weights")

    def h1(value: int) -> int:
        return value % 7

    def h2(value: int) -> int:
        return (2 * value) % 7

    set_a = (1, 2, 3)
    set_b = (1, 2, 4)
    if min(h1(v) for v in set_a) != 1 or min(h1(v) for v in set_b) != 1:
        raise AssertionError("h1")
    if min(h2(v) for v in set_a) != 2 or min(h2(v) for v in set_b) != 1:
        raise AssertionError("h2")
    if Fraction(2, 4) != Fraction(1, 2):
        raise AssertionError("jaccard")

    mix = 0.5 * 0.1 + 0.5 * 0.02
    assert_close(mix, 0.06)
    assert_close(0.5 * 0.1, 0.05)

    n_params = 10**8
    n_tokens = 2 * 10**9
    compute = 6 * n_params * n_tokens
    if compute != 1_200_000_000_000_000_000:
        raise AssertionError(compute)
    if 2 * n_params * n_tokens * 3 != compute:
        raise AssertionError("2ND is one third")
    if compute // 10**14 != 12_000:
        raise AssertionError("seconds")
    if 2 * 10**5 * 10**4 != n_tokens:
        raise AssertionError("tokens from steps")
    if 6 * n_params * (2 * 10**5) != 120_000_000_000_000:
        raise AssertionError("flops per step")

    plan_a = compute // (6 * n_params)
    plan_b_n = 4 * 10**8
    plan_b = compute // (6 * plan_b_n)
    if (plan_a, plan_b) != (2 * 10**9, 5 * 10**8):
        raise AssertionError((plan_a, plan_b))
    if Fraction(plan_b, plan_b_n) != Fraction(5, 4):
        raise AssertionError("tokens per param")
    if (14 * 10**11) // (70 * 10**9) != 20:
        raise AssertionError("chinchilla ratio")

    d_model, n_layer, seq = 512, 8, 1024
    small_n = 12 * n_layer * d_model**2
    if small_n != 25_165_824:
        raise AssertionError(small_n)
    if Fraction(seq, 6 * d_model) != Fraction(1, 3):
        raise AssertionError("attention ratio")
    if Fraction(seq, 12 * d_model) != Fraction(1, 6):
        raise AssertionError("kaplan mask ratio")
    forward_body = 2 * small_n
    forward_out = 2 * 50_257 * d_model
    if (forward_body, forward_out) != (50_331_648, 51_463_168):
        raise AssertionError((forward_body, forward_out))
    adjusted = compute // 3 * 4
    if adjusted != 1_600_000_000_000_000_000:
        raise AssertionError(adjusted)

    wide_n = 12 * 80 * 8192**2
    wide_body = 2 * wide_n
    wide_out = 2 * 50_257 * 8192
    if (wide_n, wide_body, wide_out) != (64_424_509_440, 128_849_018_880, 823_410_688):
        raise AssertionError((wide_n, wide_body, wide_out))
    ratio = wide_out / wide_body
    if round(ratio, 6) != 0.006391:
        raise AssertionError(round(ratio, 6))

    val = (3.55, 3.10, 3.05, 3.20)
    if val.index(min(val)) != 2:
        raise AssertionError("best checkpoint is step 400")
    assert_close(math.exp(3.05), 21.11534442254061, tol=1e-12)
    assert_close(math.exp(3.20), 24.532530197109352, tol=1e-12)
    if (round(math.exp(3.05), 4), round(math.exp(3.20), 4)) != (21.1153, 24.5325):
        raise AssertionError("perplexity table")
    if 400 * 2 * 10**5 != 8 * 10**7:
        raise AssertionError("tokens at step 400")
    assert_close(math.exp(0.05), 1.051271096376024, tol=1e-12)


def check_alignment() -> None:
    def sigmoid(value: float) -> float:
        return 1.0 / (1.0 + math.exp(-value))

    answer = (-math.log(0.4) - math.log(0.8)) / 2.0
    mixed = (-math.log(0.9) + answer * 2.0) / 3.0
    if (round(answer, 4), round(mixed, 4), round(-math.log(0.9), 4)) != (0.5697, 0.4149, 0.1054):
        raise AssertionError((round(answer, 4), round(mixed, 4), round(-math.log(0.9), 4)))

    for reward_w, reward_l in ((1.0, -1.0), (6.0, 4.0)):
        gap = reward_w - reward_l
        if gap != 2.0:
            raise AssertionError(gap)
        assert_close(sigmoid(gap), 1.0 / (1.0 + math.exp(-2.0)))
    if (round(sigmoid(2.0), 4), round(-math.log(sigmoid(2.0)), 4)) != (0.8808, 0.1269):
        raise AssertionError("sigma(2)")
    if (round(sigmoid(-2.0), 4), round(-math.log(sigmoid(-2.0)), 4)) != (0.1192, 2.1269):
        raise AssertionError("sigma(-2)")
    assert_close(-math.log(sigmoid(-2.0)), 2.0 - math.log(sigmoid(2.0)))

    delta = math.log(3.0)
    assert_close(sigmoid(delta), 0.75, tol=1e-12)
    if round(delta, 4) != 1.0986:
        raise AssertionError(round(delta, 4))
    pair_sum = -(3 * math.log(0.75) + math.log(0.25))
    if (round(pair_sum, 4), round(pair_sum / 4.0, 4)) != (2.2493, 0.5623):
        raise AssertionError((round(pair_sum, 4), round(pair_sum / 4.0, 4)))
    for value in (-2.0, 0.0, 2.0, 10.0):
        if 4.0 * sigmoid(-value) <= 0.0:
            raise AssertionError("unanimous preference likelihood still increases")
    # Uniform reference, reward gap ln 3: the shared 1/2 cancels, refusal gets 3/4.
    weights = (Fraction(1, 2) * 3, Fraction(1, 2))
    if weights[0] / sum(weights) != Fraction(3, 4):
        raise AssertionError(weights)

    pi_hack, pi_good = sigmoid(2.0), sigmoid(-2.0)
    assert_close(pi_hack + pi_good, 1.0)
    proxy = 1.0 + 2.0 * pi_hack
    kl = math.log(2.0) + pi_hack * math.log(pi_hack) + pi_good * math.log(pi_good)
    objective = proxy - kl
    point = 3.0 - math.log(2.0)
    if (round(pi_hack, 4), round(proxy, 4), round(pi_good, 4)) != (0.8808, 2.7616, 0.1192):
        raise AssertionError((round(pi_hack, 4), round(proxy, 4), round(pi_good, 4)))
    if (round(kl, 4), round(objective, 4), round(point, 4)) != (0.3278, 2.4338, 2.3069):
        raise AssertionError((round(kl, 4), round(objective, 4), round(point, 4)))
    if not point < objective:
        raise AssertionError("point mass should score below the optimum")

    pi_hack_4 = sigmoid(0.5)
    if (round(pi_hack_4, 4), round(1.0 - pi_hack_4, 4), round(1.0 + 2.0 * pi_hack_4, 4)) != (
        0.6225,
        0.3775,
        2.2449,
    ):
        raise AssertionError("beta 4")

    def clipped(rho: float, advantage: float, eps: float = 0.2) -> float:
        ratio = min(max(rho, 1.0 - eps), 1.0 + eps)
        return min(rho * advantage, ratio * advantage)

    if (clipped(1.5, 1.0), clipped(0.5, 1.0), clipped(1.5, -1.0), clipped(0.5, -1.0)) != (
        1.2,
        0.5,
        -1.5,
        -0.8,
    ):
        raise AssertionError("ppo clip")
    if (round(math.exp(0.1), 4), round(math.exp(1.5), 4)) != (1.1052, 4.4817):
        raise AssertionError("policy ratios")
    if not 0.8 < math.exp(0.1) < 1.2:
        raise AssertionError("exp(0.1) should sit inside the clip interval")
    assert_close(2.0 - 0.1 * ((-1.0) - (-8.0)), 1.3)
    assert_close(2.0 - 0.2 * 7.0, 0.6)

    logp_w = -0.5 + -0.5
    logp_l = -1.5 + -2.5
    logp_ref_w = -1.0 + -1.0
    logp_ref_l = -1.0 + -2.0
    if (logp_w, logp_l, logp_ref_w, logp_ref_l) != (-1.0, -4.0, -2.0, -3.0):
        raise AssertionError((logp_w, logp_l, logp_ref_w, logp_ref_l))
    margin = (logp_w - logp_ref_w) - (logp_l - logp_ref_l)
    if margin != 2.0:
        raise AssertionError(margin)
    if (
        round(sigmoid(margin), 4),
        round(-math.log(sigmoid(margin)), 4),
        round(sigmoid(-margin), 4),
    ) != (0.8808, 0.1269, 0.1192):
        raise AssertionError("dpo pair")
    if 0.5 * margin != 1.0:
        raise AssertionError("beta 0.5")
    mean_margin = ((-0.5) - (-1.0)) - ((-2.0) - (-1.5))
    if mean_margin != 1.0:
        raise AssertionError(mean_margin)
    if (round(sigmoid(1.0), 4), round(-math.log(sigmoid(1.0)), 4)) != (0.7311, 0.3133):
        raise AssertionError("sigma(1)")
    if logp_w - logp_l != 3.0:
        raise AssertionError("dropped reference")
    if (round(sigmoid(3.0), 4), round(-math.log(sigmoid(3.0)), 4)) != (0.9526, 0.0486):
        raise AssertionError("sigma(3)")
    reversed_margin = ((-3.0) - logp_ref_w) - ((-2.0) - logp_ref_l)
    if reversed_margin != -2.0:
        raise AssertionError(reversed_margin)
    # Renormalizing the two policy sequences adds one constant and cancels.
    total = math.exp(logp_w) + math.exp(logp_l)
    renorm_margin = (math.log(math.exp(logp_w) / total) - logp_ref_w) - (
        math.log(math.exp(logp_l) / total) - logp_ref_l
    )
    assert_close(renorm_margin, margin)


def longest_shared_run(left: tuple[str, ...], right: tuple[str, ...]) -> int:
    best = 0
    for i in range(len(left)):
        for j in range(len(right)):
            length = 0
            while (
                i + length < len(left)
                and j + length < len(right)
                and left[i + length] == right[j + length]
            ):
                length += 1
            if length > best:
                best = length
    return best


def pass_at_k(n: int, c: int, k: int) -> float:
    """Chen 等的无偏估计：1 - C(n-c, k) / C(n, k)。"""
    if n - c < k:
        return 1.0
    fail = 1.0
    for i in range(n - c + 1, n + 1):
        fail *= 1.0 - k / i
    return 1.0 - fail


def check_evaluation() -> None:
    # 参考续写的困惑度，和「另一种写法也算对」的任务成功。
    model_a = (0.5, 0.1, 0.4)
    model_b = (0.25, 0.6, 0.15)
    if sum(model_a) != 1 or sum(model_b) != 1:
        raise AssertionError("probabilities")
    assert_close(math.exp(-math.log(model_a[0])), 2.0)
    assert_close(math.exp(-math.log(model_b[0])), 4.0)
    if (round(-math.log(model_a[0]), 4), round(-math.log(model_b[0]), 4)) != (0.6931, 1.3863):
        raise AssertionError("reference nll")
    if (model_a[0] + model_a[1], model_b[0] + model_b[1]) != (0.6, 0.85):
        raise AssertionError("task success")

    sum_correct = 2 * math.log(0.5)
    sum_wrong = math.log(0.4)
    mean_correct = sum_correct / 2
    if (round(sum_correct, 4), round(sum_wrong, 4)) != (-1.3863, -0.9163):
        raise AssertionError("logprob sums")
    if (round(mean_correct, 4), round(sum_wrong, 4)) != (-0.6931, -0.9163):
        raise AssertionError("logprob means")
    if not (sum_wrong > sum_correct and mean_correct > sum_wrong):
        raise AssertionError("length changes the winner")

    assert_close(pass_at_k(5, 2, 2), 0.7)
    if Fraction(math.comb(3, 2), math.comb(5, 2)) != Fraction(3, 10):
        raise AssertionError("binomial ratio")
    product = (1 - Fraction(2, 4)) * (1 - Fraction(2, 5))
    if product != Fraction(3, 10):
        raise AssertionError(product)
    plugin = 1 - (1 - Fraction(2, 5)) ** 2
    if plugin != Fraction(16, 25):
        raise AssertionError(plugin)
    if pass_at_k(5, 4, 2) != 1.0:
        raise AssertionError("all but one correct")
    if 1 - (Fraction(1, 5)) ** 2 != Fraction(24, 25):
        raise AssertionError("plugin when c=4")
    assert_close(pass_at_k(5, 2, 1), 0.4)
    # 连乘和组合数在另一组上也相同。
    n, c, k = 10, 3, 4
    assert_close(pass_at_k(n, c, k), 1 - math.comb(n - c, k) / math.comb(n, k))

    reference = {"猫": 1, "在": 1, "垫子": 1}
    candidate = ("猫", "猫", "猫", "在")
    raw = sum(1 for token in candidate if token in reference)
    clipped = 0
    used: dict[str, int] = {}
    for token in candidate:
        if used.get(token, 0) < reference.get(token, 0):
            clipped += 1
            used[token] = used.get(token, 0) + 1
    if (raw, clipped) != (4, 2):
        raise AssertionError((raw, clipped))
    if Fraction(clipped, len(candidate)) != Fraction(1, 2):
        raise AssertionError("clipped precision")

    if (65 + 45) / 2 / 100 != 0.55 or 65 / 100 != 0.65:
        raise AssertionError("position protocol")

    k_elo = 32
    expected = 1 / 2
    rating_a = 1000 + k_elo * (1 - expected)
    rating_b = 1000 + k_elo * (0 - expected)
    if (rating_a, rating_b) != (1016, 984):
        raise AssertionError((rating_a, rating_b))
    gap = 1 / (1 + 10 ** ((1000 - 1400) / 400))
    assert_close(gap, 10 / 11)
    if round(gap, 4) != 0.9091:
        raise AssertionError(round(gap, 4))
    assert_close(1 / (1 + math.exp(-math.log(10))), 10 / 11)
    if round(math.log(10), 4) != 2.3026:
        raise AssertionError(round(math.log(10), 4))

    train = ("报道", "泰坦尼克", "的", "船长", "是", "谁", "的", "新闻")
    exact = ("泰坦尼克", "的", "船长", "是", "谁")
    paraphrase = ("沉船", "的", "船长", "叫", "什么")
    if longest_shared_run(exact, train) != 5:
        raise AssertionError("five-gram")
    if longest_shared_run(paraphrase, train) != 2:
        raise AssertionError("paraphrase overlap")

    def exposure(num_canaries: int, rank: int) -> float:
        return math.log2(num_canaries) - math.log2(rank)

    if (exposure(1024, 1), exposure(1024, 512)) != (10, 1):
        raise AssertionError("canary exposure")

    p_hat = 80 / 100
    se = math.sqrt(p_hat * (1 - p_hat) / 100)
    assert_close(se, 0.04)
    half = 1.96 * se
    assert_close(half, 0.0784)
    if (round(p_hat - half, 4), round(p_hat + half, 4)) != (0.7216, 0.8784):
        raise AssertionError("wald interval")

    only_a, only_b, both, neither = 10, 5, 70, 15
    if only_a + only_b + both + neither != 100:
        raise AssertionError("paired counts")
    if (both + only_a, both + only_b) != (80, 75):
        raise AssertionError("accuracies")
    diff = (only_a - only_b) / 100
    var_d = (only_a + only_b) / 100 - diff**2
    assert_close(var_d, 0.1475)
    se_paired = math.sqrt(var_d / 100)
    if round(se_paired, 4) != 0.0384 or round(1.96 * se_paired, 4) != 0.0753:
        raise AssertionError((round(se_paired, 4), round(1.96 * se_paired, 4)))
    if (round(diff - 1.96 * se_paired, 4), round(diff + 1.96 * se_paired, 4)) != (-0.0253, 0.1253):
        raise AssertionError("paired interval")
    cov = both / 100 - 0.8 * 0.75
    assert_close(cov, 0.10)
    var_check = 0.8 * 0.2 + 0.75 * 0.25 - 2 * cov
    assert_close(var_check, var_d)
    se_independent = math.sqrt(0.04**2 + 0.75 * 0.25 / 100)
    assert_close(se_independent, math.sqrt(0.003475))
    if round(se_independent, 4) != 0.0589:
        raise AssertionError(round(se_independent, 4))

    for n_items, below in ((384, False), (385, True)):
        width = 0.98 / math.sqrt(n_items)
        if (width <= 0.05) != below:
            raise AssertionError((n_items, width))
    if (round(0.98 / math.sqrt(384), 5), round(0.98 / math.sqrt(385), 5)) != (0.05001, 0.04995):
        raise AssertionError("sample size widths")

    se_phone = math.sqrt(0.6 * 0.4 / 40)
    half_phone = 1.96 * se_phone
    if (round(se_phone, 5), round(half_phone, 5)) != (0.07746, 0.15182):
        raise AssertionError((round(se_phone, 5), round(half_phone, 5)))
    if (round(0.6 - half_phone, 4), round(0.6 + half_phone, 4)) != (0.4482, 0.7518):
        raise AssertionError("phone interval")
    if round(1.96 * round(se_phone, 4), 4) != 0.1519:
        raise AssertionError("rounding the standard error first")
    assert_close(math.sqrt(1.0 * 0.0 / 40), 0.0)


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
    check_probability()
    check_optimizer()
    check_attention()
    check_structures()
    check_activation_norm()
    check_pretraining()
    check_alignment()
    check_evaluation()
    print("ok")


if __name__ == "__main__":
    main()
