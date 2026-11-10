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


def round_half_away(value: Fraction) -> int:
    """最近整数；恰好在两个整数正中时远离 0。"""
    if value < 0:
        return -round_half_away(-value)
    return (value.numerator + value.denominator // 2) // value.denominator


def symmetric_quantize(weights: tuple[Fraction, ...], qmax: int) -> tuple[tuple[int, ...], Fraction, tuple[Fraction, ...]]:
    peak = max(abs(w) for w in weights)
    if peak == 0:
        return tuple(0 for _ in weights), Fraction(0), tuple(Fraction(0) for _ in weights)
    scale = peak / qmax
    codes = tuple(round_half_away(w / scale) for w in weights)
    if any(code < -qmax or code > qmax for code in codes):
        raise AssertionError(codes)
    restored = tuple(code * scale for code in codes)
    errors = tuple(w - hat for w, hat in zip(weights, restored))
    return codes, scale, errors


def check_inference() -> None:
    # Mistral 7B 表 1 的 GQA 形状。留下 2048，批量 4，每元素 2 字节。
    layers, kv_heads, head_dim, seq, batch, width = 32, 8, 128, 2048, 4, 2
    kv_bytes = layers * 2 * kv_heads * head_dim * seq * batch * width
    if kv_bytes != 2**30:
        raise AssertionError(kv_bytes)
    query_heads = 32
    mha_bytes = layers * 2 * query_heads * head_dim * seq * batch * width
    mqa_bytes = layers * 2 * 1 * head_dim * seq * batch * width
    if (mha_bytes, mqa_bytes) != (2**32, 2**27):
        raise AssertionError((mha_bytes, mqa_bytes))
    if mha_bytes != 4 * kv_bytes or kv_bytes != 8 * mqa_bytes:
        raise AssertionError("MHA/GQA/MQA ratios")
    # 滚动缓冲按窗口分配。前缀 2048 只写满一半，前缀 8192 覆写旧槽，分配长度都是 4096。
    window = 4096
    occupied = {2048: min(2048, window), 8192: min(8192, window)}
    if occupied != {2048: 2048, 8192: 4096}:
        raise AssertionError(occupied)
    rolled = layers * 2 * kv_heads * head_dim * window * batch * width
    if rolled != 2**31:
        raise AssertionError(rolled)
    weight_bytes = 2 * 4096**2
    kv_one_layer = 2 * kv_heads * head_dim * 2048 * 1 * width
    if (weight_bytes, kv_one_layer, weight_bytes // kv_one_layer) != (33_554_432, 8_388_608, 4):
        raise AssertionError((weight_bytes, kv_one_layer))

    # 方阵 d=4096，fp16。强度 = (S d) / (d + 2 S)。
    d = 4096
    decode_intensity = Fraction(1 * d, d + 2 * 1)
    prefill_intensity = Fraction(2048 * d, d + 2 * 2048)
    if decode_intensity != Fraction(2048, 2049) or prefill_intensity != 1024:
        raise AssertionError((decode_intensity, prefill_intensity))
    peak, bandwidth = 10**14, 2 * 10**12
    ridge = peak // bandwidth
    if ridge != 50:
        raise AssertionError(ridge)
    if not (decode_intensity < ridge < prefill_intensity):
        raise AssertionError("roofline sides")
    if bandwidth * 1 != 2 * 10**12 or Fraction(bandwidth, peak) != Fraction(1, 50):
        raise AssertionError("bandwidth fraction")
    exact_fraction = decode_intensity / ridge
    if exact_fraction != Fraction(2048, 50 * 2049):
        raise AssertionError(exact_fraction)
    batch8 = Fraction(8 * d, d + 2 * 8)
    if batch8 != Fraction(2048, 257) or not (batch8 < ridge):
        raise AssertionError(batch8)
    # 忽略激活时 fp16 强度等于批量，脊点 50 要求批量 50。
    if Fraction(50) != ridge:
        raise AssertionError("batch at the ridge")

    # 注意力只计 KV 读。fp16 下强度约为 query 头 / KV 头。
    if query_heads // kv_heads != 4:
        raise AssertionError("attention intensity")
    score_over_kv = Fraction(query_heads, 2 * kv_heads * head_dim)
    if score_over_kv != Fraction(1, 64):
        raise AssertionError(score_over_kv)

    # 静态批的槽步，以及块大小 4 的分页。
    useful, slots = 2 + 2 + 2 + 8, 4 * 8
    if Fraction(useful, slots) != Fraction(7, 16) or slots - useful != 18:
        raise AssertionError((useful, slots))
    lengths = (5, 5, 6)
    block = 4
    blocks = sum((n + block - 1) // block for n in lengths)
    paged_slots = blocks * block
    # 5%4=1 浪费 3；6%4=2 浪费 2。整除的块没有内部碎片。
    internal = sum(0 if n % block == 0 else block - n % block for n in lengths)
    if (blocks, paged_slots, internal) != (6, 24, 8):
        raise AssertionError((blocks, paged_slots, internal))
    if 3 * 16 != 48 or 48 - sum(lengths) != 32:
        raise AssertionError("contiguous reservation")

    # 投机解码。α=4/5，γ=3，c=1/10。正中舍入不参与这条。
    alpha = Fraction(4, 5)
    gamma = 3
    probs = []
    for k in range(gamma):
        probs.append(alpha**k * (1 - alpha))
    probs.append(alpha**gamma)
    if probs != [Fraction(1, 5), Fraction(4, 25), Fraction(16, 125), Fraction(64, 125)]:
        raise AssertionError(probs)
    if sum(probs) != 1:
        raise AssertionError(sum(probs))
    accepted = sum(k * p for k, p in enumerate(probs))
    generated = accepted + 1
    if accepted != Fraction(244, 125) or generated != Fraction(369, 125):
        raise AssertionError((accepted, generated))
    closed = (1 - alpha ** (gamma + 1)) / (1 - alpha)
    if closed != generated:
        raise AssertionError(closed)
    cost = Fraction(1, 10)
    speedup = generated / (gamma * cost + 1)
    if speedup != Fraction(738, 325):
        raise AssertionError(speedup)
    if round(float(accepted), 4) != 1.9520:
        raise AssertionError(round(float(accepted), 4))
    if round(float(generated), 4) != 2.9520:
        raise AssertionError(round(float(generated), 4))
    if round(float(speedup), 4) != 2.2708:
        raise AssertionError(round(float(speedup), 4))
    if gamma + 1 != 4:
        raise AssertionError("alpha = 1")
    if (1 - Fraction(0)) / 1 != 1:
        raise AssertionError("alpha = 0")

    # 对称 int8 / int4，以及组大小 2 的后一组。
    weights = (Fraction(2, 5), Fraction(-6, 5), Fraction(1, 5), Fraction(4, 5))
    q8, scale8, err8 = symmetric_quantize(weights, 127)
    if scale8 != Fraction(6, 635) or q8 != (42, -127, 21, 85):
        raise AssertionError((q8, scale8))
    if err8 != (Fraction(2, 635), 0, Fraction(1, 635), Fraction(-2, 635)):
        raise AssertionError(err8)
    if max(abs(e) for e in err8) != Fraction(2, 635):
        raise AssertionError(max(err8))
    if scale8 / 2 != Fraction(3, 635):
        raise AssertionError("half step int8")
    q4, scale4, err4 = symmetric_quantize(weights, 7)
    if scale4 != Fraction(6, 35) or q4 != (2, -7, 1, 5):
        raise AssertionError((q4, scale4))
    if err4 != (Fraction(2, 35), 0, Fraction(1, 35), Fraction(-2, 35)):
        raise AssertionError(err4)
    if max(abs(e) for e in err4) != Fraction(2, 35):
        raise AssertionError("int4")
    if scale4 / scale8 != Fraction(127, 7):
        raise AssertionError("step ratio")
    group = weights[2:]
    qg, scaleg, errg = symmetric_quantize(group, 127)
    if scaleg != Fraction(4, 635) or qg != (32, 127):
        raise AssertionError((qg, scaleg))
    if errg != (Fraction(-1, 635), 0):
        raise AssertionError(errg)

    # 非对称 int8，铺满 [-127, 127]，整数跨度 254。权重 (0, 1, 5)。
    asym = (Fraction(0), Fraction(1), Fraction(5))
    span = 254
    scale_a = (asym[-1] - asym[0]) / span
    if scale_a != Fraction(5, 254):
        raise AssertionError(scale_a)
    q_a = tuple(round_half_away((w - asym[0]) / scale_a) - 127 for w in asym)
    if q_a != (-127, -76, 127):
        raise AssertionError(q_a)
    restored = tuple(asym[0] + (q + 127) * scale_a for q in q_a)
    if restored != (Fraction(0), Fraction(255, 254), Fraction(5)):
        raise AssertionError(restored)
    if Fraction(1) - restored[1] != Fraction(-1, 254):
        raise AssertionError("asymmetric error")
    q_s, scale_s, err_s = symmetric_quantize(asym, 127)
    if scale_s != Fraction(5, 127) or q_s != (0, 25, 127):
        raise AssertionError((q_s, scale_s))
    if err_s[1] != Fraction(2, 127) or err_s[1] != Fraction(4, 254):
        raise AssertionError(err_s)

    if round_half_away(Fraction(1, 2)) != 1 or round_half_away(Fraction(-1, 2)) != -1:
        raise AssertionError("ties")
    if round_half_away(Fraction(127, 4)) != 32 or round_half_away(Fraction(127, 5)) != 25:
        raise AssertionError("31.75 and 25.4")

    # TTFT 200 ms，TPOT 25 ms，N = 9。排队时间为 0。
    ttft, tpot, n_out = 200, 25, 9
    e2e = ttft + (n_out - 1) * tpot
    if e2e != 400 or ttft + n_out * tpot != 425:
        raise AssertionError((e2e, ttft + n_out * tpot))
    throughput = Fraction(n_out) / Fraction(e2e, 1000)
    if throughput != Fraction(45, 2):
        raise AssertionError(throughput)


def rag_chunks(tokens: list[str], size: int, stride: int) -> list[list[str]]:
    """起点 0, stride, 2 stride, ... 只要起点还在文档内就切一块。"""
    chunks: list[list[str]] = []
    start = 0
    while start < len(tokens):
        chunks.append(tokens[start : start + size])
        start += stride
    return chunks


def rag_search_layer(
    graph: dict[str, list[str]],
    dist: dict[str, int],
    entry: str,
    ef: int,
) -> set[str]:
    """HNSW 一层搜索：ef 是结果列表的上限。距离更小者优先。"""
    visited = {entry}
    candidates = {entry}
    found = {entry}
    while candidates:
        current = min(candidates, key=lambda name: (dist[name], name))
        candidates.remove(current)
        furthest = max(found, key=lambda name: (dist[name], name))
        if dist[current] > dist[furthest]:
            break
        for neighbor in graph[current]:
            if neighbor in visited:
                continue
            visited.add(neighbor)
            furthest = max(found, key=lambda name: (dist[name], name))
            if dist[neighbor] < dist[furthest] or len(found) < ef:
                candidates.add(neighbor)
                found.add(neighbor)
                if len(found) > ef:
                    found.remove(max(found, key=lambda name: (dist[name], name)))
    return found


def check_rag() -> None:
    query = (3, 1)
    vectors = {"甲": (3, 1), "乙": (9, -7), "丙": (1, 2), "丁": (0, 3)}

    def dot(left: tuple[int, int], right: tuple[int, int]) -> int:
        return left[0] * right[0] + left[1] * right[1]

    def dist2(left: tuple[int, int], right: tuple[int, int]) -> int:
        return (left[0] - right[0]) ** 2 + (left[1] - right[1]) ** 2

    dots = {name: dot(query, vector) for name, vector in vectors.items()}
    if dots != {"甲": 10, "乙": 20, "丙": 5, "丁": 3}:
        raise AssertionError(dots)
    distances = {name: dist2(query, vector) for name, vector in vectors.items()}
    if distances != {"甲": 0, "乙": 100, "丙": 5, "丁": 13}:
        raise AssertionError(distances)
    q_norm = math.sqrt(dot(query, query))
    cosines = {
        name: dots[name] / (q_norm * math.sqrt(dot(vector, vector)))
        for name, vector in vectors.items()
    }
    assert_close(cosines["甲"], 1.0)
    assert_close(cosines["乙"], 2 / math.sqrt(13))
    assert_close(cosines["丙"], 1 / math.sqrt(2))
    assert_close(cosines["丁"], 1 / math.sqrt(10))
    rounded_cos = tuple(round(cosines[name], 4) for name in ("乙", "丙", "丁"))
    if rounded_cos != (0.5547, 0.7071, 0.3162):
        raise AssertionError(rounded_cos)
    by_dot = tuple(sorted(dots, key=lambda name: -dots[name]))
    by_cos = tuple(sorted(cosines, key=lambda name: -cosines[name]))
    by_dist = tuple(sorted(distances, key=lambda name: distances[name]))
    if (by_dot, by_cos, by_dist) != (
        ("乙", "甲", "丙", "丁"),
        ("甲", "丙", "乙", "丁"),
        ("甲", "丙", "丁", "乙"),
    ):
        raise AssertionError((by_dot, by_cos, by_dist))

    # 未归一化的点积做 softmax 时，乙几乎拿走全部质量。
    weights = [math.exp(value - dots["乙"]) for value in dots.values()]
    if weights[list(dots).index("乙")] / sum(weights) <= 0.9999:
        raise AssertionError("inner-product softmax should concentrate on 乙")

    def recall(ranked: tuple[str, ...] | list[str], relevant: set[str], k: int) -> Fraction:
        hit = len(set(ranked[:k]) & relevant)
        return Fraction(hit, len(relevant))

    def hit(ranked: tuple[str, ...] | list[str], relevant: set[str], k: int) -> int:
        return int(bool(set(ranked[:k]) & relevant))

    topical = {"甲", "丙"}
    if (recall(by_cos, topical, 1), recall(by_cos, topical, 2)) != (Fraction(1, 2), 1):
        raise AssertionError("cosine recall")
    if (hit(by_cos, topical, 1), hit(by_cos, topical, 2)) != (1, 1):
        raise AssertionError("cosine hit")
    if (recall(by_dot, topical, 1), recall(by_dot, topical, 2), recall(by_dot, topical, 3)) != (
        0,
        Fraction(1, 2),
        1,
    ):
        raise AssertionError("inner-product recall")
    if (hit(by_dot, topical, 1), hit(by_dot, topical, 2)) != (0, 1):
        raise AssertionError("inner-product hit")
    if (recall(by_dist, topical, 1), recall(by_dist, topical, 2)) != (Fraction(1, 2), 1):
        raise AssertionError("euclidean recall")

    # Lucene IDF 与 Robertson–Zaragoza 式 (3.3)。N = 4，两个查询词都出现在 2 篇里。
    n_docs, n_containing = 4, 2
    robertson_idf = math.log((n_docs - n_containing + 0.5) / (n_containing + 0.5))
    lucene_idf = math.log(1 + (n_docs - n_containing + 0.5) / (n_containing + 0.5))
    assert_close(robertson_idf, 0.0)
    assert_close(lucene_idf, math.log(2))
    if round(lucene_idf, 4) != 0.6931:
        raise AssertionError(round(lucene_idf, 4))

    k1, b = Fraction(6, 5), Fraction(3, 4)
    avgdl = Fraction(4)
    lengths = {"甲": 9, "乙": 3, "丙": 2, "丁": 2}
    term_freqs = {
        "甲": (1, 1),
        "乙": (3, 0),
        "丙": (0, 1),
        "丁": (0, 0),
    }

    def bm25_coefficient(dl: int, freqs: tuple[int, int]) -> Fraction:
        norm = (1 - b) + b * Fraction(dl, avgdl)
        total = Fraction(0)
        for freq in freqs:
            if freq == 0:
                continue
            total += Fraction(freq) / (freq + k1 * norm)
        return total

    coefficients = {
        name: bm25_coefficient(lengths[name], term_freqs[name]) for name in lengths
    }
    if coefficients != {
        "甲": Fraction(80, 133),
        "乙": Fraction(40, 53),
        "丙": Fraction(4, 7),
        "丁": Fraction(0),
    }:
        raise AssertionError(coefficients)
    bm25_scores = {name: float(coefficients[name]) * lucene_idf for name in coefficients}
    rounded_bm25 = tuple(round(bm25_scores[name], 4) for name in ("甲", "乙", "丙", "丁"))
    if rounded_bm25 != (0.4169, 0.5231, 0.3961, 0.0):
        raise AssertionError(rounded_bm25)
    bm25_order = tuple(sorted(bm25_scores, key=lambda name: -bm25_scores[name]))
    if bm25_order != ("乙", "甲", "丙", "丁"):
        raise AssertionError(bm25_order)
    answer = {"甲"}
    if (recall(bm25_order, answer, 1), recall(bm25_order, answer, 2)) != (0, 1):
        raise AssertionError("bm25 recall of 甲")
    # 把「含有查询词」当成相关集时，命中变成 1，召回率只有 1/3。天数仍不在第一名。
    lexical = {"乙", "甲", "丙"}
    if (recall(bm25_order, lexical, 1), hit(bm25_order, lexical, 1)) != (Fraction(1, 3), 1):
        raise AssertionError("relabeling the relevant set")
    # 词频饱和的上限是 IDF。乙走到了 40/53。
    if coefficients["乙"] != Fraction(40, 53):
        raise AssertionError("saturation progress")
    if round(float(Fraction(40, 53)), 4) != 0.7547:
        raise AssertionError("40/53")
    variant = tuple(round(2.2 * bm25_scores[name], 4) for name in ("甲", "乙", "丙"))
    if variant != (0.9172, 1.1509, 0.8714):
        raise AssertionError(variant)
    # 先收成四位再乘，乙的第四位会从 1.1509 写成 1.1508。
    if round(round(bm25_scores["乙"], 4) * 2.2, 4) != 1.1508:
        raise AssertionError("rounding before scaling")
    if round((11 / 5) * lucene_idf, 4) != 1.5249:
        raise AssertionError("variant ceiling")

    # Robertson 与 Zaragoza 提到的另一组常见参数。顺序在这四篇上不变。
    k1_alt, b_alt = Fraction(2), Fraction(1, 2)

    def bm25_alt(dl: int, freqs: tuple[int, int]) -> Fraction:
        norm = (1 - b_alt) + b_alt * Fraction(dl, avgdl)
        total = Fraction(0)
        for freq in freqs:
            if freq == 0:
                continue
            total += Fraction(freq) / (freq + k1_alt * norm)
        return total

    alt = {name: bm25_alt(lengths[name], term_freqs[name]) for name in lengths}
    if alt != {"甲": Fraction(8, 17), "乙": Fraction(12, 19), "丙": Fraction(2, 5), "丁": Fraction(0)}:
        raise AssertionError(alt)
    alt_rounded = tuple(round(float(alt[name]) * lucene_idf, 4) for name in ("甲", "乙", "丙"))
    if alt_rounded != (0.3262, 0.4378, 0.2773):
        raise AssertionError(alt_rounded)
    if not alt["乙"] > alt["甲"] > alt["丙"]:
        raise AssertionError("alternate parameters should keep 乙 first")

    # 直接相加时，余弦的差距盖过 BM25。
    mixed = {name: bm25_scores[name] + cosines[name] for name in vectors}
    mixed_rounded = tuple(round(mixed[name], 4) for name in ("甲", "乙", "丙", "丁"))
    if mixed_rounded != (1.4169, 1.0778, 1.1032, 0.3162):
        raise AssertionError(mixed_rounded)
    if round(bm25_scores["乙"] - bm25_scores["甲"], 4) != 0.1062:
        raise AssertionError("bm25 lead")
    if round(cosines["甲"] - cosines["乙"], 4) != 0.4453:
        raise AssertionError("cosine lead")
    mixed_order = tuple(sorted(mixed, key=lambda name: -mixed[name]))
    if mixed_order != ("甲", "丙", "乙", "丁"):
        raise AssertionError(mixed_order)

    bm25_rank = {name: index + 1 for index, name in enumerate(bm25_order)}
    cosine_rank = {name: index + 1 for index, name in enumerate(by_cos)}
    k_rrf = 60
    rrf_scores = {
        name: Fraction(1, k_rrf + bm25_rank[name]) + Fraction(1, k_rrf + cosine_rank[name])
        for name in vectors
    }
    if rrf_scores != {
        "甲": Fraction(123, 3782),
        "乙": Fraction(124, 3843),
        "丙": Fraction(125, 3906),
        "丁": Fraction(1, 32),
    }:
        raise AssertionError(rrf_scores)
    rrf_rounded = tuple(round(float(rrf_scores[name]), 4) for name in ("甲", "乙", "丙"))
    if rrf_rounded != (0.0325, 0.0323, 0.0320):
        raise AssertionError(rrf_rounded)
    assert_close(float(rrf_scores["丁"]), 0.03125)
    rrf_order = tuple(sorted(rrf_scores, key=lambda name: -rrf_scores[name]))
    if rrf_order != ("甲", "乙", "丙", "丁"):
        raise AssertionError(rrf_order)
    if (recall(bm25_order, answer, 1), recall(rrf_order, answer, 1)) != (0, 1):
        raise AssertionError("rrf recall")

    # 交叉编码器的 logit 经 sigmoid 后顺序不变。
    rerank_logits = {"甲": 2.0, "乙": 0.0, "丙": -1.0}
    rerank_probs = {name: 1 / (1 + math.exp(-logit)) for name, logit in rerank_logits.items()}
    if (
        round(rerank_probs["甲"], 4),
        rerank_probs["乙"],
        round(rerank_probs["丙"], 4),
    ) != (0.8808, 0.5, 0.2689):
        raise AssertionError(rerank_probs)
    rerank_order = tuple(sorted(rerank_logits, key=lambda name: -rerank_logits[name]))
    if rerank_order != ("甲", "乙", "丙"):
        raise AssertionError(rerank_order)

    # 一个 token 上，RAG-Sequence 与 RAG-Token 都是这个和式。
    p_seven = Fraction(3, 5) * Fraction(3, 4) + Fraction(2, 5) * Fraction(1, 10)
    p_three = Fraction(3, 5) * Fraction(1, 4) + Fraction(2, 5) * Fraction(9, 10)
    if (p_seven, p_three) != (Fraction(49, 100), Fraction(51, 100)):
        raise AssertionError((p_seven, p_three))
    if not (Fraction(3, 4) > Fraction(1, 4) and p_three > p_seven):
        raise AssertionError("marginalization should flip the argmax")

    document = "本公司 规定 退货 期限 是 七 天 请 保留 发票".split()
    if len(document) != 10:
        raise AssertionError(len(document))
    disjoint = rag_chunks(document, 4, 4)
    if disjoint != [
        ["本公司", "规定", "退货", "期限"],
        ["是", "七", "天", "请"],
        ["保留", "发票"],
    ]:
        raise AssertionError(disjoint)
    if any("退货" in chunk and "七" in chunk for chunk in disjoint):
        raise AssertionError("disjoint chunks should split the evidence")
    overlapped = rag_chunks(document, 4, 2)
    if overlapped[1] != ["退货", "期限", "是", "七"]:
        raise AssertionError(overlapped)
    if sum(len(chunk) for chunk in overlapped) != 18:
        raise AssertionError(sum(len(chunk) for chunk in overlapped))
    if sum("退货" in chunk and "七" in chunk for chunk in overlapped) != 1:
        raise AssertionError("one overlapped chunk keeps both terms")

    jia = "退货 的 期限 是 七 天 并且 需要 发票".split()
    yi = ["退货", "退货", "退货"]
    if (len(jia), len(yi)) != (9, 3):
        raise AssertionError("chunk lengths")
    budget = 7
    jia_first = (jia + yi)[:budget]
    yi_first = (yi + jia)[:budget]
    if jia_first != "退货 的 期限 是 七 天 并且".split() or "七" not in jia_first:
        raise AssertionError(jia_first)
    if yi_first != "退货 退货 退货 退货 的 期限 是".split() or "七" in yi_first:
        raise AssertionError(yi_first)

    def pack(items: list[tuple[str, int]], limit: int) -> list[str]:
        chosen: list[str] = []
        remaining = limit
        for name, length in items:
            if length <= remaining:
                chosen.append(name)
                remaining -= length
        return chosen

    short_first = pack([("丙", 2), ("丁", 2), ("乙", 3), ("甲", 9)], 11)
    rerank_pack = pack([("甲", 9), ("乙", 3), ("丙", 2)], 11)
    if short_first != ["丙", "丁", "乙"] or rerank_pack != ["甲", "丙"]:
        raise AssertionError((short_first, rerank_pack))

    # 三条主张：上下文忠实 2/3，引用正确 1/3。
    supported_by_prompt = (True, True, False)
    citation_ok = (True, False, False)
    if (
        Fraction(sum(supported_by_prompt), 3),
        Fraction(sum(citation_ok), 3),
    ) != (Fraction(2, 3), Fraction(1, 3)):
        raise AssertionError("faithfulness")
    # 丢掉没有引用的第三条之后，前两条都由提示支持。
    if Fraction(sum(supported_by_prompt[:2]), 2) != 1:
        raise AssertionError("dropped claim")

    # IVF：nprobe = 1 打开错误的桶。
    centroids = (0, 10)
    points = {"a": 4, "b": 9}
    lists: dict[int, list[str]] = {centroid: [] for centroid in centroids}
    for name, point in points.items():
        owner = min(centroids, key=lambda centroid: abs(point - centroid))
        lists[owner].append(name)
    if lists != {0: ["a"], 10: ["b"]}:
        raise AssertionError(lists)
    query_1d = 6
    probed = min(centroids, key=lambda centroid: abs(query_1d - centroid))
    if probed != 10 or abs(query_1d - points["b"]) != 3 or abs(query_1d - points["a"]) != 2:
        raise AssertionError("ivf distances")

    # HNSW 底层。ef = 1 停在 B；ef = 2 能走到 C。
    hnsw_dist = {"A": 5, "B": 2, "C": 1, "M": 3}
    base_graph = {"A": ["B"], "B": ["A", "M"], "M": ["B", "C"], "C": ["M"]}
    if rag_search_layer(base_graph, hnsw_dist, "A", 1) != {"B"}:
        raise AssertionError("ef 1")
    if rag_search_layer(base_graph, hnsw_dist, "A", 2) != {"C", "B"}:
        raise AssertionError("ef 2")
    upper_graph = {"A": ["M"], "M": ["A"], "B": [], "C": []}
    if rag_search_layer(upper_graph, hnsw_dist, "A", 1) != {"M"}:
        raise AssertionError("upper layer")
    if rag_search_layer(base_graph, hnsw_dist, "M", 1) != {"C"}:
        raise AssertionError("descend at M")


def solve2(a11, a12, a21, a22, b1, b2):
    det = a11 * a22 - a12 * a21
    if det == 0:
        raise AssertionError("singular")
    return (b1 * a22 - a12 * b2) / det, (a11 * b2 - b1 * a21) / det


def mean_squared(points, weights) -> Fraction:
    total = Fraction(0)
    for row in points:
        pred = sum(weight * x for weight, x in zip(weights, row[:-1]))
        total += (row[-1] - pred) ** 2
    return total / len(points)


def soft_threshold(z: Fraction, lam: Fraction) -> Fraction:
    if z > lam:
        return z - lam
    if z < -lam:
        return z + lam
    return Fraction(0)


def check_generalization() -> None:
    # 三行、两个特征，没有截距。(X^T X + λ I) w = X^T y。
    train = (
        (Fraction(1), Fraction(0), Fraction(1)),
        (Fraction(0), Fraction(1), Fraction(1)),
        (Fraction(1), Fraction(1), Fraction(0)),
    )
    val = (
        (Fraction(1), Fraction(0), Fraction(1, 2)),
        (Fraction(0), Fraction(1), Fraction(0)),
        (Fraction(1), Fraction(1), Fraction(1, 2)),
    )
    test = (
        (Fraction(1), Fraction(0), Fraction(1)),
        (Fraction(0), Fraction(1), Fraction(0)),
        (Fraction(1), Fraction(1), Fraction(1)),
    )
    solutions = {}
    for lam in (Fraction(0), Fraction(1), Fraction(3)):
        w1, w2 = solve2(2 + lam, 1, 1, 2 + lam, 1, 1)
        if w1 != w2 or w1 != 1 / (3 + lam):
            raise AssertionError((lam, w1, w2))
        solutions[lam] = (w1, w2)

    if solutions[0] != (Fraction(1, 3), Fraction(1, 3)):
        raise AssertionError(solutions[0])
    if solutions[1] != (Fraction(1, 4), Fraction(1, 4)):
        raise AssertionError(solutions[1])
    if solutions[3] != (Fraction(1, 6), Fraction(1, 6)):
        raise AssertionError(solutions[3])

    train_mse = {lam: mean_squared(train, w) for lam, w in solutions.items()}
    val_mse = {lam: mean_squared(val, w) for lam, w in solutions.items()}
    test_mse = {lam: mean_squared(test, w) for lam, w in solutions.items()}
    if train_mse != {0: Fraction(4, 9), 1: Fraction(11, 24), 3: Fraction(1, 2)}:
        raise AssertionError(train_mse)
    if val_mse != {0: Fraction(1, 18), 1: Fraction(1, 24), 3: Fraction(1, 18)}:
        raise AssertionError(val_mse)
    if test_mse != {0: Fraction(2, 9), 1: Fraction(7, 24), 3: Fraction(7, 18)}:
        raise AssertionError(test_mse)
    if train_mse[1] - train_mse[0] != Fraction(1, 72):
        raise AssertionError("train gap")
    if val_mse[0] - val_mse[1] != Fraction(1, 72):
        raise AssertionError("val gap")
    if min(val_mse, key=val_mse.get) != 1:
        raise AssertionError("validation should select λ = 1")
    if not (test_mse[0] < test_mse[1]):
        raise AssertionError("test prefers the unselected λ")

    # X^T (y - Xw) = λ w。λ = 0 时残差垂直于列。
    for lam, (w1, w2) in solutions.items():
        residuals = []
        for x1, x2, y in train:
            residuals.append(y - (w1 * x1 + w2 * x2))
        dot1 = residuals[0] * 1 + residuals[1] * 0 + residuals[2] * 1
        dot2 = residuals[0] * 0 + residuals[1] * 1 + residuals[2] * 1
        if dot1 != lam * w1 or dot2 != lam * w2:
            raise AssertionError((lam, dot1, dot2))

    def matvec(matrix, vector):
        return (
            matrix[0][0] * vector[0] + matrix[0][1] * vector[1],
            matrix[1][0] * vector[0] + matrix[1][1] * vector[1],
        )

    gram = ((2, 1), (1, 2))
    if matvec(gram, (1, 1)) != (3, 3) or matvec(gram, (1, -1)) != (1, -1):
        raise AssertionError("eigenvalues 3 and 1")
    shifted = ((3, 1), (1, 3))
    if matvec(shifted, (1, 1)) != (4, 4) or matvec(shifted, (1, -1)) != (2, -2):
        raise AssertionError("eigenvalues 4 and 2")
    if Fraction(3, 1) != 3 or Fraction(4, 2) != 2:
        raise AssertionError("condition numbers")

    # 正交设计上的一个坐标。L1 目标 (1/2)(w-z)^2 + λ|w|，L2 是 z/(1+λ)。
    lam = Fraction(1)
    cases = {
        Fraction(3): (Fraction(2), Fraction(3, 2)),
        Fraction(1, 2): (Fraction(0), Fraction(1, 4)),
        Fraction(-1, 2): (Fraction(0), Fraction(-1, 4)),
        Fraction(-3): (Fraction(-2), Fraction(-3, 2)),
    }
    for z, (w_l1, w_l2) in cases.items():
        if soft_threshold(z, lam) != w_l1 or z / (1 + lam) != w_l2:
            raise AssertionError((z, soft_threshold(z, lam)))
    # |z| ≤ λ 时，次梯度 s = z/λ 使 -z + λ s = 0。
    z = Fraction(1, 2)
    s = z / lam
    if not (-1 <= s <= 1) or -z + lam * s != 0:
        raise AssertionError("subgradient")

    # 两个点的留一。全数据训练误差选 λ = 0，留一平均选 λ = 1。
    def fit_through_origin(xy, lam):
        return sum(x * y for x, y in xy) / (sum(x * x for x, _ in xy) + lam)

    points = ((1, 1), (1, 3))
    for lam, w_hat, train_error, loo in (
        (Fraction(0), Fraction(2), Fraction(1), Fraction(4)),
        (Fraction(1), Fraction(4, 3), Fraction(13, 9), Fraction(13, 4)),
    ):
        if fit_through_origin(points, lam) != w_hat:
            raise AssertionError((lam, fit_through_origin(points, lam)))
        train_sse = sum((y - w_hat * x) ** 2 for x, y in points)
        if train_sse / 2 != train_error:
            raise AssertionError((lam, train_sse))
        held = []
        for index in range(2):
            rest = [points[j] for j in range(2) if j != index]
            w_fold = fit_through_origin(rest, lam)
            x, y = points[index]
            held.append((y - w_fold * x) ** 2)
        if held[0] != held[1] and lam == 0:
            raise AssertionError(held)
        if sum(held) / 2 != loo:
            raise AssertionError((lam, held))
    if (Fraction(25, 4), Fraction(1, 4)) != (
        (3 - Fraction(1, 2)) ** 2,
        (1 - Fraction(3, 2)) ** 2,
    ):
        raise AssertionError("loo squares")

    # 偏差–方差。f = 1，σ² = 1，两个训练集等可能。
    sigma2 = Fraction(1)

    def decompose(predictions):
        mean = sum(predictions) / len(predictions)
        bias2 = (Fraction(1) - mean) ** 2
        var = sum((p - mean) ** 2 for p in predictions) / len(predictions)
        return bias2, var, sigma2 + bias2 + var

    if decompose((Fraction(3), Fraction(-1))) != (0, 4, 5):
        raise AssertionError("model A")
    if decompose((Fraction(2), Fraction(1))) != (Fraction(1, 4), Fraction(1, 4), Fraction(3, 2)):
        raise AssertionError("model B")
    if decompose((Fraction(0), Fraction(0))) != (1, 0, 2):
        raise AssertionError("constant 0")

    def direct_average(predictions):
        total = Fraction(0)
        count = 0
        for yhat in predictions:
            for y in (0, 2):
                total += (Fraction(y) - yhat) ** 2
                count += 1
        return total / count

    if direct_average((Fraction(3), Fraction(-1))) != 5:
        raise AssertionError("direct A")
    if direct_average((Fraction(2), Fraction(1))) != Fraction(3, 2):
        raise AssertionError("direct B")
    if direct_average((Fraction(0),)) != 2:
        raise AssertionError("direct zero")

    # y ∈ {0, 2}。永远预测 0 的误差期望是 2。样本均值用穷举核对。
    def expected_mean_errors(n: int):
        train_sum = Fraction(0)
        test_sum = Fraction(0)
        count = 0
        for mask in range(1 << n):
            sample = tuple(2 if mask & (1 << i) else 0 for i in range(n))
            mean = Fraction(sum(sample), n)
            train_sum += sum((Fraction(y) - mean) ** 2 for y in sample) / n
            test_sum += sum((Fraction(y) - mean) ** 2 for y in (0, 2)) / 2
            count += 1
        return train_sum / count, test_sum / count

    if expected_mean_errors(2) != (Fraction(1, 2), Fraction(3, 2)):
        raise AssertionError(expected_mean_errors(2))
    if expected_mean_errors(4) != (Fraction(3, 4), Fraction(5, 4)):
        raise AssertionError(expected_mean_errors(4))
    for n in (2, 4):
        if expected_mean_errors(n) != (Fraction(n - 1, n), Fraction(n + 1, n)):
            raise AssertionError(n)
        if Fraction(n + 1, n) - Fraction(n - 1, n) != Fraction(2, n):
            raise AssertionError("gap")
    if Fraction(0**2 + 2**2, 2) != 2:
        raise AssertionError("irreducible second moment")


def check_minimal_experiment() -> None:
    # 大设定沿用预训练笔记。小核对把 N 和 D 各缩小 100 倍。
    big = 6 * 10**8 * 2 * 10**9
    small = 6 * 10**6 * 2 * 10**7
    if (big, small) != (1_200_000_000_000_000_000, 120_000_000_000_000):
        raise AssertionError((big, small))
    if big // small != 10_000:
        raise AssertionError(big // small)
    if Fraction(small, 10**14) != Fraction(6, 5):
        raise AssertionError("1.2 seconds")
    if (10**8) // (10**6) != 100 or (2 * 10**9) // (2 * 10**7) != 100:
        raise AssertionError("param and data ratios")

    # d = 64，两层，S = 256，V = 512。注意力占 2/3，输出投影占 1/3，合计是 6ND 的 2 倍。
    d_model, n_layer, seq, vocab, tokens = 64, 2, 256, 512, 10**7
    n_params = 12 * n_layer * d_model**2
    if n_params != 98_304:
        raise AssertionError(n_params)
    if Fraction(seq, 6 * d_model) != Fraction(2, 3):
        raise AssertionError("attention ratio")
    if Fraction(vocab * d_model, n_params) != Fraction(1, 3):
        raise AssertionError("output ratio")
    body = 6 * n_params * tokens
    with_attention = body * 5 // 3
    output = body // 3
    total = with_attention + output
    if (body, with_attention, output, total) != (
        5_898_240_000_000,
        9_830_400_000_000,
        1_966_080_000_000,
        11_796_480_000_000,
    ):
        raise AssertionError((body, with_attention, output, total))
    if total != body * 2 or Fraction(1) + Fraction(2, 3) + Fraction(1, 3) != 2:
        raise AssertionError("body plus attention plus output")

    baseline = (1, 0, 1, 0, 0, 1, 0, 1, 0, 1)
    changed = (1, 1, 1, 0, 1, 1, 0, 0, 1, 1)
    if (sum(baseline), sum(changed), sum(changed) - sum(baseline)) != (5, 7, 2):
        raise AssertionError("correct counts")
    if not sum(changed) - sum(baseline) < 3:
        raise AssertionError("a lead of 2 does not meet +3")
    both = only_changed = only_baseline = neither = 0
    for left, right in zip(baseline, changed):
        if left and right:
            both += 1
        elif right:
            only_changed += 1
        elif left:
            only_baseline += 1
        else:
            neither += 1
    if (both, only_changed, only_baseline, neither) != (4, 3, 1, 2):
        raise AssertionError((both, only_changed, only_baseline, neither))
    # 第 2 题和第 5 题是只改动对的两格。改成错之后两边都是 5。
    if (changed[1], changed[4]) != (1, 1) or (baseline[1], baseline[4]) != (0, 0):
        raise AssertionError("items 2 and 5")
    if sum(changed) - 2 != sum(baseline):
        raise AssertionError("flipping two cells ties the table")

    diff = Fraction(only_changed - only_baseline, 10)
    second = Fraction(only_changed + only_baseline, 10)
    var = second - diff**2
    if (diff, second, var) != (Fraction(1, 5), Fraction(2, 5), Fraction(9, 25)):
        raise AssertionError((diff, second, var))
    se = math.sqrt(float(var) / 10)
    half = 1.96 * se
    if (round(se, 4), round(half, 4), round(float(diff) - half, 4), round(float(diff) + half, 4)) != (
        0.1897,
        0.3719,
        -0.1719,
        0.5719,
    ):
        raise AssertionError((round(se, 4), round(half, 4)))
    if not (float(diff) - half < 0 < float(diff) + half):
        raise AssertionError("paired interval should contain 0")

    # 四格是为了拆差而设的整数：只换部件 3，只换数据 2，两样都换 8，叠在一起多出来 3。
    only_component = 9 - 6
    only_data = 8 - 6
    joint = 14 - 6
    leftover = joint - (only_component + only_data)
    if (only_component, only_data, joint, leftover) != (3, 2, 8, 3):
        raise AssertionError((only_component, only_data, joint, leftover))

    if Fraction(1, 2) ** 4 != Fraction(1, 16):
        raise AssertionError("four independent misses")
    if 1 - Fraction(1, 2) ** 4 != Fraction(15, 16):
        raise AssertionError("at least one of four trials")

    contaminated = Fraction(3 + 2, 10)
    clean = Fraction(2, 7)
    if contaminated != Fraction(1, 2) or not clean < Fraction(1, 2):
        raise AssertionError((contaminated, clean))
    if round(float(clean), 4) != 0.2857:
        raise AssertionError(round(float(clean), 4))


def check_statistics() -> None:
    # Y ∈ {0, 2} 各 1/2。n = 2 的四个独立样本，以及第二条是副本的情形。
    samples = ((0, 0), (0, 2), (2, 0), (2, 2))

    def expectation(values):
        return sum((Fraction(v) for v in values), Fraction(0)) / len(values)

    def variance(values):
        mu = expectation(values)
        return expectation([(v - mu) ** 2 for v in values])

    means = []
    squared_dev = []
    sse = []
    mle = []
    unbiased = []
    for sample in samples:
        mean = Fraction(sum(sample), 2)
        sq = sum((Fraction(y) - mean) ** 2 for y in sample)
        means.append(mean)
        squared_dev.append((mean - 1) ** 2)
        sse.append(sq)
        mle.append(sq / 2)
        unbiased.append(sq / 1)
    if means != [0, 1, 1, 2] or squared_dev != [1, 0, 0, 1]:
        raise AssertionError((means, squared_dev))
    if sse != [0, 2, 2, 0] or mle != [0, 1, 1, 0] or unbiased != [0, 2, 2, 0]:
        raise AssertionError((sse, mle, unbiased))
    if expectation(means) != 1 or variance(means) != Fraction(1, 2):
        raise AssertionError((expectation(means), variance(means)))
    if expectation([(m - 1) ** 2 for m in means]) != Fraction(1, 2):
        raise AssertionError("mean mse")
    if round(math.sqrt(0.5), 4) != 0.7071:
        raise AssertionError(round(math.sqrt(0.5), 4))

    # 副本：协方差为 1，样本均值的方差不再是 σ²/n。
    if expectation((0, 2)) != 1 or variance((0, 2)) != 1:
        raise AssertionError("copied observations")
    if Fraction(1 + 1 + 2 * 1, 4) != 1 or Fraction(1 + 1, 4) != Fraction(1, 2):
        raise AssertionError("covariance identity")

    if expectation(mle) != Fraction(1, 2) or expectation(mle) - 1 != Fraction(-1, 2):
        raise AssertionError("mle bias")
    if variance(mle) != Fraction(1, 4):
        raise AssertionError(variance(mle))
    if variance(mle) + (expectation(mle) - 1) ** 2 != Fraction(1, 2):
        raise AssertionError("mle mse")
    if expectation([(v - 1) ** 2 for v in mle]) != Fraction(1, 2):
        raise AssertionError("mle mse direct")
    if (expectation(unbiased), variance(unbiased)) != (1, 1):
        raise AssertionError("unbiased variance")
    if expectation([(v - 1) ** 2 for v in unbiased]) != 1:
        raise AssertionError("unbiased mse")
    # (0, 2) 这一次打中 σ²，四个样本的平均仍是 1/2。
    if mle[1] != 1 or unbiased[1] != 2:
        raise AssertionError("single sample")

    # Bernoulli p = 1/2，n = 2。p̂ = k/n，不重推导数。
    bernoulli = ((0, 0), (0, 1), (1, 0), (1, 1))
    p_hat = [Fraction(sum(sample), 2) for sample in bernoulli]
    if p_hat != [0, Fraction(1, 2), Fraction(1, 2), 1]:
        raise AssertionError(p_hat)
    if expectation(p_hat) != Fraction(1, 2) or variance(p_hat) != Fraction(1, 8):
        raise AssertionError((expectation(p_hat), variance(p_hat)))
    if Fraction(1, 2) * Fraction(1, 2) / 2 != Fraction(1, 8):
        raise AssertionError("p(1-p)/n")

    se = math.sqrt((0.5 * 0.5) / 2)
    half = 1.96 * se
    if (round(se, 4), round(half, 4)) != (0.3536, 0.6930):
        raise AssertionError((round(se, 4), round(half, 4)))
    if (round(0.5 - half, 4), round(0.5 + half, 4)) != (-0.1930, 1.1930):
        raise AssertionError("wald at 1/2")
    if (round(0.0 - half, 4), round(0.0 + half, 4)) != (-0.6930, 0.6930):
        raise AssertionError("wald at 0")
    if (round(1.0 - half, 4), round(1.0 + half, 4)) != (0.3070, 1.6930):
        raise AssertionError("wald at 1")

    # 0, 2, 4, 6。前两条训练，后两条测试。
    data = (0, 2, 4, 6)
    train, test = data[:2], data[2:]
    mu_train = Fraction(sum(train), len(train))
    mu_test = Fraction(sum(test), len(test))
    mu_all = Fraction(sum(data), len(data))
    if (mu_train, mu_test, mu_all) != (1, 5, 3):
        raise AssertionError((mu_train, mu_test, mu_all))
    if mu_all != (2 * mu_train + 2 * mu_test) / 4:
        raise AssertionError("pooled mean")
    if mu_all - mu_test != Fraction(len(train), len(data)) * (mu_train - mu_test):
        raise AssertionError("shrink toward the test mean")

    def sse(points, center) -> Fraction:
        return sum((Fraction(y) - center) ** 2 for y in points)

    if sse(test, mu_test) != 2:
        raise AssertionError("test residual about its own mean")
    if sse(test, mu_train) != 2 + 2 * (mu_test - mu_train) ** 2:
        raise AssertionError("identity")
    if (sse(test, mu_train), sse(test, mu_all)) != (34, 10):
        raise AssertionError("test sse")
    if (sse(train, mu_train), sse(train, mu_all)) != (2, 10):
        raise AssertionError("train sse")
    if (sse(data, mu_train), sse(data, mu_all)) != (36, 20):
        raise AssertionError("total sse")
    if not sse(test, mu_all) < sse(test, mu_train):
        raise AssertionError("full mean should lower the test sum of squares")

    train_var = sse(train, mu_train) / len(train)
    full_var = sse(data, mu_all) / len(data)
    if (train_var, full_var) != (1, 5):
        raise AssertionError((train_var, full_var))
    if tuple((y - mu_train) / 1 for y in test) != (3, 5):
        raise AssertionError("train standardization")
    std = math.sqrt(5)
    scaled = tuple((y - 3) / std for y in test)
    assert_close(scaled[0], 1 / math.sqrt(5))
    assert_close(scaled[1], 3 / math.sqrt(5))
    if (round(std, 4), round(scaled[0], 4), round(scaled[1], 4)) != (2.2361, 0.4472, 1.3416):
        raise AssertionError((round(std, 4), round(scaled[0], 4), round(scaled[1], 4)))

    # σ² = 1 时平均负对数似然的梯度是 μ - x̄。学习率 1 从 0 走到均值。
    if (0 - mu_train, 0 - mu_all) != (-1, -3):
        raise AssertionError("gradients")
    if (0 - 1 * (0 - mu_train), 0 - 1 * (0 - mu_all)) != (1, 3):
        raise AssertionError("one step")

    # c = 1 时单条梯度是 1 或 -1。
    example_grad = {0: 1, 2: -1}
    batch_mean = [
        Fraction(example_grad[a] + example_grad[b], 2) for a, b in samples
    ]
    if batch_mean != [1, 0, 0, -1]:
        raise AssertionError(batch_mean)
    if expectation(batch_mean) != 0 or variance(batch_mean) != Fraction(1, 2):
        raise AssertionError("independent minibatch")
    copied_grad = (1, -1)
    if expectation(copied_grad) != 0 or variance(copied_grad) != 1:
        raise AssertionError("duplicated minibatch")

    lr = Fraction(1, 2)
    updated = [1 - lr * g for g in batch_mean]
    copied_updated = [1 - lr * g for g in copied_grad]
    if updated != [Fraction(1, 2), 1, 1, Fraction(3, 2)]:
        raise AssertionError(updated)
    if copied_updated != [Fraction(1, 2), Fraction(3, 2)]:
        raise AssertionError(copied_updated)
    if expectation(updated) != 1 or variance(updated) != Fraction(1, 8):
        raise AssertionError((expectation(updated), variance(updated)))
    if expectation(copied_updated) != 1 or variance(copied_updated) != Fraction(1, 4):
        raise AssertionError(variance(copied_updated))
    if lr**2 * variance(batch_mean) != Fraction(1, 8):
        raise AssertionError("learning rate scales independent variance")
    if lr**2 * variance(copied_grad) != Fraction(1, 4):
        raise AssertionError("learning rate scales duplicated variance")


def gpipe_idle_slots(stages: int, microbatches: int) -> tuple[int, ...]:
    """GPipe：先全部前向，再全部反向。前向和反向各占 1 拍，阶段工作量相同。"""
    occupied = [set() for _ in range(stages)]
    for micro in range(microbatches):
        for stage in range(stages):
            occupied[stage].add(stage + micro)
    backward_start = microbatches + stages - 1
    for step in range(microbatches):
        for stage in range(stages):
            occupied[stage].add(backward_start + (stages - 1 - stage) + step)
    wall = 2 * (microbatches + stages - 1)
    return tuple(sum(1 for t in range(wall) if t not in busy) for busy in occupied)


def check_distributed() -> None:
    # 两个 worker，本地对样本取平均，再对 worker 取平均。
    local = ((6 + 2) / 2, (0 + -4) / 2)
    if local != (4, -2):
        raise AssertionError(local)
    averaged = (local[0] + local[1]) / 2
    if averaged != 1 or (6 + 2 + 0 + -4) / 4 != 1:
        raise AssertionError(averaged)
    if local[0] + local[1] != 2:
        raise AssertionError("sum of local means")
    summed = ((6 + 2) + (0 + -4)) / 2
    if summed != 2 or (6 + 2 + 0 + -4) != 4 or 4 / summed != 2:
        raise AssertionError(summed)

    per_worker, workers, steps = 32, 4, 1000
    tokens_dp = workers * per_worker * steps
    tokens_one = per_worker * steps
    if (tokens_dp, tokens_one, tokens_dp // per_worker) != (128_000, 32_000, 4_000):
        raise AssertionError((tokens_dp, tokens_one))
    if Fraction(tokens_one, tokens_dp) != Fraction(1, 4):
        raise AssertionError("token ratio")
    microbatches = 4
    if workers * microbatches * per_worker != 512 or microbatches * per_worker != 128:
        raise AssertionError("effective batch")

    lr = Fraction(1, 10)
    if 0 - lr * 1 != Fraction(-1, 10) or 0 - lr * 2 != Fraction(-1, 5):
        raise AssertionError("learning-rate scale")

    # 混合精度 Adam：fp16 参数、fp16 梯度、fp32 副本和两份矩。
    psi = 2**20
    nd = 4
    if 2 + 2 + 4 + 4 + 4 != 16:
        raise AssertionError("bytes per parameter")
    naive = 16 * psi
    zero1 = 4 * psi + (12 * psi) // nd
    zero2 = 2 * psi + (14 * psi) // nd
    zero3 = (16 * psi) // nd
    if (naive, zero1, zero2, zero3) != (16_777_216, 7_340_032, 5_767_168, 4_194_304):
        raise AssertionError((naive, zero1, zero2, zero3))
    if (naive, zero1, zero3) != (2**24, 7 * 2**20, 2**22):
        raise AssertionError("powers of two")
    if zero2 != Fraction(11, 2) * psi:
        raise AssertionError(zero2)
    ratios = (Fraction(naive, zero1), Fraction(naive, zero2), Fraction(naive, zero3))
    if ratios != (Fraction(16, 7), Fraction(32, 11), 4):
        raise AssertionError(ratios)
    if not (ratios[0] < 4 and ratios[1] < 8):
        raise AssertionError("finite Nd is below the large-Nd limits")
    layer = 2 * psi // 4
    if layer != 2**19 or layer != 524_288:
        raise AssertionError(layer)
    # 论文的 1.5B 模型：16 字节是 24 GB，fp32 缓冲是 6 GB，都按 10^9 字节。
    if 15 * 10**8 * 16 != 24 * 10**9 or 4 * 15 * 10**8 != 6 * 10**9:
        raise AssertionError("paper byte counts")

    def matvec(weight: tuple[tuple[int, ...], ...], vector: tuple[int, ...]) -> tuple[int, ...]:
        # y = x @ W.T，W 的每一行是一个输出通道。
        return tuple(sum(x * w for x, w in zip(vector, row)) for row in weight)

    x = (1, 2, 3, 4)
    w1 = (
        (1, 0, 0, 0),
        (0, 1, 0, 0),
        (1, 1, 0, 0),
        (0, 0, 1, 1),
    )
    full_y = matvec(w1, x)
    shard_y = (matvec(w1[:2], x), matvec(w1[2:], x))
    if full_y != (1, 2, 3, 7) or shard_y != ((1, 2), (3, 7)):
        raise AssertionError((full_y, shard_y))
    if shard_y[0] + shard_y[1] != full_y:
        raise AssertionError("column shards concatenate")

    w2 = (
        (1, 2, 3, 4),
        (0, 1, 0, 1),
    )
    full_z = matvec(w2, full_y)
    partials = (
        matvec(tuple(row[:2] for row in w2), full_y[:2]),
        matvec(tuple(row[2:] for row in w2), full_y[2:]),
    )
    reduced = tuple(a + b for a, b in zip(*partials))
    if full_z != (42, 9) or partials != ((5, 2), (37, 7)) or reduced != full_z:
        raise AssertionError((full_z, partials, reduced))
    bias = (1, 1)
    once = tuple(z + b for z, b in zip(reduced, bias))
    twice = tuple(a + b + 2 * c for a, b, c in zip(*partials, bias))
    if once != (43, 10) or twice != (44, 11):
        raise AssertionError((once, twice))
    if tuple(t - o for t, o in zip(twice, once)) != bias:
        raise AssertionError("extra bias")

    grad_y = (1, 1, 1, 1)
    # 列切分的反向：每个 rank 用自己的输出梯度和权重行，对输入梯度求和。
    def input_grad(rows: tuple[tuple[int, ...], ...], grad_out: tuple[int, ...]) -> tuple[int, ...]:
        width = len(rows[0])
        return tuple(sum(grad_out[i] * rows[i][j] for i in range(len(rows))) for j in range(width))

    parts = (input_grad(w1[:2], grad_y[:2]), input_grad(w1[2:], grad_y[2:]))
    grad_x = tuple(a + b for a, b in zip(*parts))
    if parts != ((1, 1, 0, 0), (1, 1, 1, 1)) or grad_x != (2, 2, 1, 1):
        raise AssertionError((parts, grad_x))
    if grad_x != input_grad(w1, grad_y):
        raise AssertionError("backward all-reduce")

    stages, micros = 4, 4
    idle = gpipe_idle_slots(stages, micros)
    wall = 2 * (micros + stages - 1)
    useful = 2 * micros
    if idle != (6, 6, 6, 6) or wall != 14 or useful != 8:
        raise AssertionError((idle, wall, useful))
    if Fraction(idle[0], wall) != Fraction(3, 7):
        raise AssertionError("bubble over wall")
    if Fraction(useful, wall) != Fraction(4, 7):
        raise AssertionError("efficiency")
    if Fraction(idle[0], useful) != Fraction(3, 4):
        raise AssertionError("bubble over useful")
    if Fraction(stages - 1, micros + stages - 1) != Fraction(3, 7):
        raise AssertionError("closed form")
    # 反向每拍按前向的 2 倍计，比例不变。
    duration_wall = 3 * (micros + stages - 1)
    duration_useful = 3 * micros
    if (duration_wall, duration_useful, duration_wall - duration_useful) != (21, 12, 9):
        raise AssertionError("weighted slots")
    if Fraction(9, 21) != Fraction(3, 7):
        raise AssertionError("weighted bubble")
    if Fraction(stages - 1, 16 + stages - 1) != Fraction(3, 19):
        raise AssertionError("sixteen microbatches")
    if Fraction(16, 16 + stages - 1) != Fraction(16, 19):
        raise AssertionError("efficiency at M = 4K")
    if Fraction(stages - 1, 1 + stages - 1) != Fraction(3, 4):
        raise AssertionError("single microbatch")
    if gpipe_idle_slots(stages, 1) != (6, 6, 6, 6):
        raise AssertionError(gpipe_idle_slots(stages, 1))
    if Fraction(1 + 3 + 5 + 7, 4) != 4 or (1 + 3 + 5 + 7) != 16:
        raise AssertionError("microbatch mean")

    factor = Fraction(nd - 1, nd)
    if factor != Fraction(3, 4):
        raise AssertionError(factor)
    dp_elements = 2 * psi * factor
    dp_bytes = 2 * dp_elements
    if (dp_elements, dp_bytes) != (1_572_864, 3_145_728):
        raise AssertionError((dp_elements, dp_bytes))
    zero2_elements = 2 * (psi * factor)
    if zero2_elements != dp_elements:
        raise AssertionError(zero2_elements)
    one_gather = psi * factor
    zero3_elements = 3 * one_gather
    zero3_bytes = 2 * zero3_elements
    if (one_gather, zero3_elements, zero3_bytes) != (786_432, 2_359_296, 4_718_592):
        raise AssertionError((one_gather, zero3_elements, zero3_bytes))
    if Fraction(zero3_elements, dp_elements) != Fraction(3, 2):
        raise AssertionError("ZeRO-3 ratio")
    message = 64 * 256
    tp_elements = 2 * message * Fraction(1, 2)
    if (message, tp_elements, 2 * tp_elements) != (16_384, 16_384, 32_768):
        raise AssertionError((message, tp_elements))
    if dp_elements / tp_elements != 96:
        raise AssertionError(dp_elements / tp_elements)
    if 4 * message != 65_536 or 2 * 4 * message != 131_072:
        raise AssertionError("four all-reduces")


def main() -> None:
    check_softmax()
    check_perplexity()
    check_least_squares()
    check_eigen_and_condition()
    check_rank_one_and_lora()
    check_logistic()
    check_generalization()
    check_backprop()
    check_bpe()
    check_algorithms()
    check_training_step()
    check_classification_counts()
    check_probability()
    check_statistics()
    check_optimizer()
    check_attention()
    check_structures()
    check_activation_norm()
    check_pretraining()
    check_minimal_experiment()
    check_alignment()
    check_evaluation()
    check_rag()
    check_inference()
    check_distributed()
    print("ok")


if __name__ == "__main__":
    main()
