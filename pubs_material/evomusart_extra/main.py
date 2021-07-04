import json
import glob
import os
import pickle
import errno
import numpy as np
import matplotlib.pyplot as plt
# from scipy.ndimage.interpolation import rotate
import math


def rotate(origin, point, angle):
    """
    Rotate a point counterclockwise by a given angle around a given origin.

    The angle should be given in radians.
    """
    ox, oy = origin
    px, py = point

    qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
    return qx, qy


def create_hash_entry(values, mode, ctime, filename, t_min, t_max):
    s = ""
    if mode == "duple":
        apd = int(abs(values[0]))
        if apd >= 100:
            raise ValueError("Invalid absolute pitch difference.")
        if values[0] >= 0:
            s += "+"
        else:
            s += "-"
        if apd < 10:
            s += "0"
        s += str(apd)
        if values[1] >= t_max or values[1] < t_min:
            raise ValueError("Invalid time difference.")
        s += f'{values[1]:.1f}'
    return {"hash": s, "ctime": ctime, "filename": filename}


class Hasher:
    def __init__(self, load=None):
        self.load = load
        if load:
            with open(load) as json_file:
                self.lookup = json.load(json_file)
        else:
            self.lookup = {}

    def contains(self, key):
        if key in self.lookup.keys():
            return self.lookup[key]
        else:
            return None

    def create_hash_entries(self, pts, filename, mode="duple",
                            t_min=0.1, t_max=10, p_min=1, p_max=12):
        nh = 0
        if mode == "duple":
            for i in range(len(pts) - 1):
                v0 = pts[i]
                j = i + 1
                while j < len(pts):
                    v1 = pts[j]
                    td = v1[0] - v0[0]
                    apd = abs(v1[1] - v0[1])
                    if t_min < td < t_max and p_min <= apd <= p_max:
                        he = create_hash_entry([v1[1] - v0[1], td], mode, v0[0], filename, t_min, t_max)
                        self.insert(he)
                        nh += 1
                    if td >= t_max:
                        j = len(pts) - 1
                    j += 1
        return nh

    def match_hash_entries(self, pts, filename, mode="duple", t_min=0.1, t_max=10, p_min=1, p_max=12):
        results = {}
        nh = 0
        if mode == "duple":
            for i in range(len(pts) - 1):
                v0 = pts[i]
                j = i + 1
                while j < len(pts):
                    v1 = pts[j]
                    td = v1[0] - v0[0]
                    apd = abs(v1[1] - v0[1])
                    if t_min < td < t_max and p_min <= apd <= p_max:
                        he = create_hash_entry([v1[1] - v0[1], td], mode, v0[0], filename, t_min, t_max)
                        match = self.contains(he["hash"])
                        if match:
                            for key, value in match.items():
                                if not (key in results.keys()):
                                    results[key] = []
                                results[key] += [[he["ctime"], v] for v in value]
                        nh += 1
                    if td >= t_max:
                        j = len(pts) - 1
                    j += 1
        return {"nosHashes": nh, "results": results}

    def insert(self, entry):
        key = entry["hash"]
        ctime = entry["ctime"]
        filename = entry["filename"]
        if key in self.lookup.keys():
            if filename in self.lookup[key].keys():
                self.lookup[key][filename].append(ctime)
            else:
                self.lookup[key][filename] = [ctime]
        else:
            self.lookup[key] = {filename: [ctime]}


def build(hasher, path, mode="duple"):
    for file in glob.glob(path + "/*.json"):
        print(f'Hashing {file}')
        with open(file) as json_file:
            points = json.load(json_file)
        hasher.create_hash_entries(points, os.path.basename(file).split(".")[0], mode)
    with open("./out/lookup.json", "w") as fp:
        json.dump(hasher.lookup, fp)


def matching():
    h = Hasher("./out/lookup.json")
    # build(h, "./original/train")
    ori_path = "./original/train"
    # can_path = "./candidates/transformer_train"
    can_path = "./original/validation"

    for f in glob.glob(can_path + "/*.json"):
        with open(f) as j_file:
            point_set = json.load(j_file)
        print(f'Matching {f} {len(point_set)}')
        for i in range((len(point_set) // 100) + 1):
            matches = h.match_hash_entries(point_set[i * 100:(i + 1) * 100], "")
            fname = f'./out/baseline/{os.path.basename(f).split(".")[0]}/{i * 100}-{(i + 1) * 100}.json'
            if not os.path.exists(os.path.dirname(fname)):
                try:
                    os.makedirs(os.path.dirname(fname))
                except OSError as exc:  # Guard against race condition
                    if exc.errno != errno.EEXIST:
                        raise
            with open(fname, "w") as j_file:
                json.dump(matches, j_file)
                print(f'\t[DONE] {i * 100}-{(i + 1) * 100}')


def check_similarity(can, ori):
    match = []
    for f in sorted(glob.glob(f'./out/baseline/{can}/*.json'), key=os.path.getmtime):
        print(f'[{can}] [{ori}] \t{f}')
        with open(f) as j_file:
            try:
                match.extend(json.load(j_file)["results"][ori])
            except KeyError:
                continue
    fname = f'./out/baseline/{can}/{ori}.pkl'
    print(f'[LEN] {len(match)}')
    with open(fname, "wb") as f:
        pickle.dump(match, f)


def baseline(w=16, o=8):
    for i in ["1211", "1219", "1240", "1827", "1893", "2322", "2368"]:
        out = []
        for f in glob.glob(f'./out/baseline/{i}/*.pkl'):
            with open(f, 'rb') as pkl_file:
                out.extend(pickle.load(pkl_file))
                print(f'[{i}] [{f}]')
        # out = sorted(out, key=lambda x: x[0])
        out = rebase(out)
        print("[DONE] Rebase.")
        ctime = 0
        max_time = np.max(out[:, 0]) - w
        print(np.max(np.abs(out[:, 1] - out[:, 0])))
        # while ctime <= max_time:
        #     tmp = out[ctime <= out[:, 0] & out[:, 0] < ctime + w]
        #     tmp = np.abs(tmp[:, 1] - tmp[:, 0])
        #     bins = np.arange(0, np.max(tmp), np.max(tmp) / 10)
        #     inds = np.digitize(tmp, bins)
        #     np.bincount(inds)
        #     ctime += (w - o)


def rebase(x):
    x = np.array(x)
    x -= np.mean(x, axis=0)
    # plt.xlim([-20, 20])
    # plt.ylim([-20, 20])
    # plt.scatter(x[:, 0], x[:, 1])
    # plt.show()
    cov_mat = np.cov(x, rowvar=False)
    eigen_values, eigen_vectors = np.linalg.eigh(cov_mat)
    sorted_index = np.argsort(eigen_values)[::-1]
    sorted_eigenvalue = eigen_values[sorted_index]
    sorted_eigenvectors = eigen_vectors[:, sorted_index]
    x = np.dot(sorted_eigenvectors.transpose(), x.transpose()).transpose()

    # plt.scatter(x[:, 0], x[:, 1])
    # plt.xlim([-20, 20])
    # plt.ylim([-20, 20])
    # plt.show()

    result = []
    for p in x:
        result.append(rotate([0, 0], p, 45))
    result = np.array(result)
    # plt.scatter(result[:, 0], result[:, 1])
    # plt.xlim([-20, 20])
    # plt.ylim([-20, 20])
    # plt.show()
    # print(result[:, 0] - result[:, 1])
    return result


if __name__ == '__main__':
    baseline()

    # for i in sorted(glob.glob('./original/validation/*.json')):
    #     for j in sorted(glob.glob('./original/train/*.mid')):
    #         check_similarity(os.path.basename(i).split(".")[0], os.path.basename(j))
