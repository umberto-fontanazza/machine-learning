from itertools import product
from pprint import pprint
from typing import cast

from load import load_data, split_data
from numpy import array, dtype, float64, full, hstack, log, ndarray, ones, uint8, zeros
from scipy.special import logsumexp
from sklearn.preprocessing import normalize

type F64Matrix = ndarray[tuple[int, int], dtype[float64]]
type U8Array = ndarray[tuple[int], dtype[uint8]]


def word_counts(train_data: list[str]) -> tuple[dict[str, int], int]:
    count = 0
    occurrencies: dict[str, int] = {}
    for line in train_data:
        for word in line.split():
            count += 1
            occurrencies[word] = occurrencies.get(word, 0) + 1
    return occurrencies, count


def train(train_data: list[list[str]], epsilon=0.001) -> list[dict[str, float64]]:
    occurrencies: list[dict[str, int]] = []
    counts: list[int] = []
    all_words: set[str] = set()
    for td in train_data:
        occ, count = word_counts(td)
        occurrencies.append(occ)
        counts.append(count)
        all_words = all_words.union(set(occ.keys()))
    log_frequencies: list[dict[str, float64]] = []
    for occ, count in zip(occurrencies, counts):
        freq = {}

        for word in all_words:
            val = log((occ.get(word, 0) + epsilon), dtype=float64)
            val -= log(count + epsilon * len(all_words))
            freq[word] = val

        log_frequencies.append(freq)
    return log_frequencies


def class_conditional_ll(
    data: list[str] | str, log_frequencies: list[dict[str, float64]]
) -> F64Matrix:
    if isinstance(data, list):
        return hstack([class_conditional_ll(entry, log_frequencies) for entry in data])
    if not isinstance(data, str):
        raise ValueError("Data must be a str!")
    words: list[str] = data.split()
    totals = zeros(len(log_frequencies), dtype=float64)
    for word, (cls, log_freq) in product(words, enumerate(log_frequencies)):
        totals[cls] += log_freq.get(word, 0)
    return totals.reshape(-1, 1)


def classify(
    data: list[str],
    log_frequencies: list[dict[str, float64]],
    target: U8Array | None = None,
):
    prior = normalize(ones((len(log_frequencies), 1), dtype=float64), axis=0, norm="l1")
    cls_ll = class_conditional_ll(data, log_frequencies)
    joint_ll = cls_ll + prior
    marginal_ll = cast(F64Matrix, logsumexp(joint_ll, axis=0)).reshape((1, -1))
    cls_posterior_ll = joint_ll - marginal_ll
    predictions = cls_posterior_ll.argmax(axis=0)
    if target is None:
        return predictions
    return (predictions != target).sum() / target.size


def main():
    hell, purgatory, heaven = load_data()
    train_hell, test_hell = split_data(hell)
    train_purgatory, test_purgatory = split_data(purgatory)
    train_heaven, test_heaven = split_data(heaven)

    binary_problems = [
        [train_hell, test_hell, train_heaven, test_heaven],
        [train_hell, test_hell, train_purgatory, test_purgatory],
        [train_purgatory, test_purgatory, train_heaven, test_heaven],
    ]
    for train_0, test_0, train_1, test_1 in binary_problems:
        model = train([train_0, train_1])
        test_data = [*test_0, *test_1]
        test_labels = array(
            [0 if index < len(test_0) else 1 for index in range(len(test_data))]
        )
        accuracy = 1 - classify(test_data, model, test_labels)
        print(f"{accuracy=}")


if __name__ == "__main__":
    main()
