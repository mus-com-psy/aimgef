import pandas as pd
from tqdm import tqdm
import numpy as np
import scipy.stats
import json


def mean_confidence_interval(x, confidence=0.95):
    a = 1.0 * np.array(x)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n - 1)
    return m, m - h, m + h


def maia_markov():
    length = 7
    results = []
    values_w_16 = [[] for _ in range(length)]
    values_w_8 = [[] for _ in range(length * 2)]
    for i in tqdm(range(25)):
        with open("./results/maia/{}.json".format(26 + i)) as jf:
            maia = json.load(jf)
        try:
            for j in range(len(values_w_16)):
                values_w_16[j].append(1 - maia["wo-16-8"]["maxSimilarities"][j]["maxSimilarity"])
            for j in range(len(values_w_8)):
                values_w_8[j].append(1 - maia["wo-8-4"]["maxSimilarities"][j]["maxSimilarity"])
        except IndexError:
            continue

    for i, v in enumerate(values_w_16):
        ci = mean_confidence_interval(v)
        results.append([i * 2 + 1, ci[0], ci[1], ci[2], "long"])
    for i, v in enumerate(values_w_8):
        ci = mean_confidence_interval(v)
        results.append([i + 1, ci[0], ci[1], ci[2], "short"])

    df = pd.DataFrame(data=results, columns=["Step", "Mean", "Min", "Max", "WO"])
    df.to_csv("./os_maia.csv", index=False)


def transformer():
    length = 7
    results = []
    values_w_16 = [[] for _ in range(length)]
    values_w_8 = [[] for _ in range(length * 2)]
    for i in tqdm(range(30)):
        with open("./results/transformer/{}.json".format(90 + i)) as jf:
            maia = json.load(jf)
        try:
            for j in range(len(values_w_16)):
                values_w_16[j].append(1 - maia["wo-16-8"]["maxSimilarities"][j]["maxSimilarity"])
            for j in range(len(values_w_8)):
                values_w_8[j].append(1 - maia["wo-8-4"]["maxSimilarities"][j]["maxSimilarity"])
        except IndexError:
            continue

    for i, v in enumerate(values_w_16):
        ci = mean_confidence_interval(v)
        results.append([i * 2 + 1, ci[0], ci[1], ci[2], "long"])
    for i, v in enumerate(values_w_8):
        ci = mean_confidence_interval(v)
        results.append([i + 1, ci[0], ci[1], ci[2], "short"])

    df = pd.DataFrame(data=results, columns=["Step", "Mean", "Min", "Max", "WO"])
    df.to_csv("./os_transformer.csv", index=False)


if __name__ == '__main__':
    maia_markov()
    transformer()
