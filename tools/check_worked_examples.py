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
    print("ok")


if __name__ == "__main__":
    main()
