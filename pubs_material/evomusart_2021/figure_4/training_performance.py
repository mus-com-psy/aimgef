import pandas as pd
import numpy as np

# os-train
os = pd.read_csv("train-os.csv", index_col=0).sort_index()
result = []
for i in range(30):
    tmp = os.iloc[i*30:(i+1)*30]
    mu = tmp[tmp["os"] >= 0.00 * i].max()["os"]
    # result.append([int(i), "t", 1 - tmp[tmp["os"] >= mu].mean()["os"]])
    result.append([int(i), "train", 1 - tmp.max()["os"]])
    if i == 13:
        print(1 - tmp.mean()["os"])
pd.DataFrame(np.array(result), columns=["checkpoint", "split", "value"]).to_csv("os_train.csv", index=False)

# os-valid
os = pd.read_csv("valid-os.csv", index_col=0).sort_index()
result = []
for i in range(30):
    tmp = os.iloc[i*30:(i+1)*30]
    mu = tmp[tmp["os"] >= 0.00 * i].max()["os"]
    # result.append([int(i), "t", 1 - tmp[tmp["os"] >= mu].mean()["os"]])
    result.append([int(i), "validation", 1 - tmp.max()["os"]])
    if i == 13:
        print(1 - tmp.mean()["os"])

pd.DataFrame(np.array(result), columns=["checkpoint", "split", "value"]).to_csv("os_valid.csv", index=False)

# loss
v = pd.read_csv("valid-loss.csv")
t = pd.read_csv("train-loss.csv")
result = []
for i, row in enumerate(v.iterrows()):
    x = row[1]["Step"]
    result.append([i, "validation", row[1]["Value"]])
    r = t.iloc[(t["Step"] - x).abs().argsort()[0]]["Value"]
    result.append([i, "train", r])
data = pd.DataFrame(np.array(result), columns=["checkpoint", "split", "value"])
data.to_csv("loss.csv", index=False)


# loss
v = pd.read_csv("valid-acc.csv")
t = pd.read_csv("train-acc.csv")
result = []
for i, row in enumerate(v.iterrows()):
    x = row[1]["Step"]
    result.append([i, "validation", row[1]["Value"]])
    r = t.iloc[(t["Step"] - x).abs().argsort()[0]]["Value"]
    result.append([i, "train", r])
data = pd.DataFrame(np.array(result), columns=["checkpoint", "split", "value"])
data.to_csv("acc.csv", index=False)