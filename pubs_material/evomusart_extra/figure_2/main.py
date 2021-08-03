import pandas as pd
import numpy as np
import scipy.stats


def mean_confidence_interval(x, confidence=0.95):
    a = 1.0 * np.array(x)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n - 1)
    return m, m - h, m + h


def baseline():
    results = []
    df = pd.read_csv("./simGen-winS_8-binS_0.1-jit_0.csv")
    df = df.fillna(df.mean()).loc[:, df.columns != "name"].apply(lambda x: 1 - x / 100)
    for i, column in enumerate(df):
        ci = mean_confidence_interval(df[column].to_list())
        results.append([i + 1, ci[0], ci[1], ci[2]])
    df = pd.DataFrame(data=results, columns=["Step", "Mean", "Min", "Max"])
    df.to_csv("./oriGen.csv", index=False)


if __name__ == '__main__':
    baseline()
