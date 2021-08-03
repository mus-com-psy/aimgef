import pandas as pd
import numpy as np
import scipy.stats


def mean_confidence_interval(x, confidence=0.95):
    a = 1.0 * np.array(x)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n - 1)
    return m, m - h, m + h


def main():
    files = [
        "./simBinScale-binS_0.1.csv",
        "./simBinScale-binS_0.5.csv",
        "./simBinScale-binS_1.csv",
        "./simBinScale-binS_5.csv"
    ]
    bin_size = [0.1, 0.5, 1, 5]
    scale = [0.75, 0.9, 1.1, 1.25]
    for i, file in enumerate(files):
        results = []
        df = pd.read_csv(file, header=None)
        df = df[[3, 5, 7, 9]].dropna().apply(lambda x: x / 100)
        for j, c in enumerate([3, 5, 7, 9]):
            for v in df[c].to_list():
                results.append([str(scale[j]), v])

        df = pd.DataFrame(data=results, columns=["Scale", "Value"])
        df.to_csv(f'{file}_cleaned.csv', index=False)


if __name__ == '__main__':
    main()
