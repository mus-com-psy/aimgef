import pandas as pd
from tqdm import tqdm
import json

if __name__ == '__main__':
    # for wo in ["wo-16-8", "wo-8-4", "wo-4-2"]:
    #     results = []
    #     for i in range(10):
    #         train_values = []
    #         valid_values = []
    #         for j in tqdm(range(30)):
    #             with open("./results/train/{}.json".format(i * 30 + j)) as jf:
    #                 train = json.load(jf)
    #             with open("./results/valid/{}.json".format(i * 30 + j)) as jf:
    #                 valid = json.load(jf)
    #             train_maxSimilarities = train[wo]["maxSimilarities"]
    #             valid_maxSimilarities = valid[wo]["maxSimilarities"]
    #             try:
    #                 for s in train_maxSimilarities:
    #                     train_values.append(1 - s["maxSimilarity"])
    #                 for s in valid_maxSimilarities:
    #                     valid_values.append(1 - s["maxSimilarity"])
    #             except KeyError:
    #                 continue
    #         results.append([i, "Train", sum(train_values) / len(train_values), min(train_values)])
    #         results.append([i, "Validation", sum(valid_values) / len(valid_values), min(valid_values)])
    #     df = pd.DataFrame(data=results, columns=["Epoch", "Split", "Mean", "Min"])
    #     df.to_csv("./os_{}.csv".format(wo), index=False)

    results = []
    values = [[] for _ in range(8)]
    for i in tqdm(range(25)):
        with open("./results/maia/{}.json".format(26 + i)) as jf:
            maia = json.load(jf)
        for j in range(8):
            values[j].append(1 - maia["wo-16-8"]["maxSimilarities"][j]["maxSimilarity"])
    for i, v in enumerate(values):
        results.append([i + 1, sum(v) / len(v), min(v)])
    df = pd.DataFrame(data=results, columns=["Epoch", "Mean", "Min"])
    df.to_csv("./os_maia.csv", index=False)
