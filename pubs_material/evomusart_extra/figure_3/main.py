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
        "./simTrain-winS_8-binS_0.1-jit_0.001.csv",
        "./simTrain-winS_8-binS_0.1-jit_0.005.csv",
        "./simTrain-winS_8-binS_0.1-jit_0.01.csv",
        "./simTrain-winS_8-binS_0.1-jit_0.05.csv",
        "./simTrain-winS_8-binS_0.1-jit_0.1.csv",
    ]
    jit = [0.001, 0.005, 0.01, 0.05, 0.1]
    results = []
    for i, file in enumerate(files):
        df = pd.read_csv(file, header=None)
        df = df[[2, 3]].dropna(subset=[2, 3], how='all').apply(lambda x: x / 100)
        diff = df.loc[df[2] != df[3]]
        success = 1 - len(diff) / len(df)
        if i < 3:
            ci = mean_confidence_interval(df[2].to_list())
        else:
            ci = mean_confidence_interval(df[3].to_list())
        results.append([jit[i], ci[0], ci[1], ci[2], success])
        df = pd.DataFrame(data=results, columns=["Jitter", "Mean", "Min", "Max", "Success"])
        df.to_csv("./simJit.csv", index=False)

def jitter():
    files = [
        "./simTrain-winS_8-binS_0.1-jit_0.001.csv",
        "./simTrain-winS_8-binS_0.1-jit_0.005.csv",
        "./simTrain-winS_8-binS_0.1-jit_0.01.csv",
        "./simTrain-winS_8-binS_0.1-jit_0.05.csv",
        "./simTrain-winS_8-binS_0.1-jit_0.1.csv",
    ]
    jit = [0.001, 0.005, 0.01, 0.05, 0.1]
    results = []
    for i, file in enumerate(files):
        df = pd.read_csv(file, header=None)
        df = df[[2, 3]].dropna(subset=[2, 3], how='all').apply(lambda x: x / 100)
        diff = df.loc[df[2] != df[3]]
        if i < 3:
            for v in df[2].to_list():
                results.append([str(jit[i]), v])
        else:
            for v in df[3].to_list():
                results.append([str(jit[i]), v])
    df = pd.DataFrame(data=results, columns=["Jitter", "Value"])
    df.to_csv("./simJit_all.csv", index=False)


if __name__ == '__main__':
    jitter()
    main()
