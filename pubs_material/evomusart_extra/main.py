import json
import glob
import os
import pickle
import errno
import numpy as np
import matplotlib.pyplot as plt
# from scipy.ndimage.interpolation import rotate
import math
import argparse
from tqdm import tqdm


def mkdir(filename):
    if not os.path.exists(os.path.dirname(filename)):
        try:
            os.makedirs(os.path.dirname(filename))
        except OSError as exc:  # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise


def build(src_path, tgt_path, mode="triple", t_min=0.1, t_max=4, p_min=1, p_max=12):
    assert os.path.isdir(src_path)
    assert os.path.isdir(tgt_path)
    if mode == "triple":
        for file in glob.glob(f'{src_path}/*.json'):
            filename = os.path.basename(file).split(".")[0]
            print(f'[HASH] {filename}')
            with open(file) as json_file:
                pts = json.load(json_file)
                pts = sorted([list(x) for x in set(tuple(x) for x in pts)], key=lambda x: x[0])
            for i in tqdm(range(len(pts) - 2)):
                v0 = pts[i]
                j = i + 1
                while j < len(pts) - 1:
                    v1 = pts[j]
                    td1 = v1[0] - v0[0]
                    pd1 = v1[1] - v0[1]
                    apd1 = abs(pd1)
                    if t_min < td1 < t_max and p_min <= apd1 <= p_max:
                        k = j + 1
                        while k < len(pts):
                            v2 = pts[k]
                            td2 = v2[0] - v1[0]
                            pd2 = v2[1] - v1[1]
                            apd2 = abs(pd2)
                            if t_min < td2 < t_max and p_min <= apd2 <= p_max:
                                if pd1 < 0:
                                    s1 = f'-{apd1}'
                                else:
                                    s1 = f'+{apd1}'
                                if pd2 < 0:
                                    s2 = f'-{apd2}'
                                else:
                                    s2 = f'+{apd2}'
                                if td1 >= td2:
                                    tdr = round((td1 / td2) * 10) / 10
                                    s3 = f'+{tdr:.1f}'
                                else:
                                    tdr = round((td2 / td1) * 10) / 10
                                    s3 = f'-{tdr:.1f}'
                                filepath = f'{tgt_path}/{s1}/{s2}/{s3}/{filename}.npy'
                                # print(f'\t[ENTRY] {s1}{s2}{s3}-{filename}')
                                mkdir(filepath)
                                if os.path.isfile(filepath):
                                    np.save(filepath, np.append(np.load(filepath), v0[0]))
                                else:
                                    np.save(filepath, np.array([v0[0]]))
                            k += 1
                    j += 1


def match(lookup_path, src_path, tgt_path, mode="triple",
          w=16, o=8, t_min=0.1, t_max=4, p_min=1, p_max=12):
    assert os.path.isdir(lookup_path)
    assert os.path.isdir(src_path)
    assert os.path.isdir(tgt_path)
    if mode == "triple":
        for file in glob.glob(f'{src_path}/*.json'):
            filename = os.path.basename(file).split(".")[0]

            with open(file) as json_file:
                points = json.load(json_file)
                points = sorted([list(x) for x in set(tuple(x) for x in points)], key=lambda x: x[0])
            t = 0
            max_t = max([p[0] for p in points]) - w
            while t <= max_t:
                print(f'[LOOK] {filename} {t}/{max_t}')
                results = {}
                pts = [p for p in points if t <= p[0] < t + w]
                nh = 0
                for i in tqdm(range(len(pts) - 2)):
                    v0 = pts[i]
                    j = i + 1
                    while j < len(pts) - 1:
                        v1 = pts[j]
                        td1 = v1[0] - v0[0]
                        pd1 = v1[1] - v0[1]
                        apd1 = abs(pd1)
                        if t_min < td1 < t_max and p_min <= apd1 <= p_max:
                            k = j + 1
                            while k < len(pts):
                                v2 = pts[k]
                                td2 = v2[0] - v1[0]
                                pd2 = v2[1] - v1[1]
                                apd2 = abs(pd2)
                                if t_min < td2 < t_max and p_min <= apd2 <= p_max:
                                    if pd1 < 0:
                                        s1 = f'-{apd1}'
                                    else:
                                        s1 = f'+{apd1}'
                                    if pd2 < 0:
                                        s2 = f'-{apd2}'
                                    else:
                                        s2 = f'+{apd2}'
                                    if td1 >= td2:
                                        tdr = round((td1 / td2) * 10) / 10
                                        s3 = f'+{tdr:.1f}'
                                    else:
                                        tdr = round((td2 / td1) * 10) / 10
                                        s3 = f'-{tdr:.1f}'
                                    if os.path.isdir(f'{lookup_path}/{s1}/{s2}/{s3}'):
                                        nh += 1
                                        for found in glob.glob(f'{lookup_path}/{s1}/{s2}/{s3}/*.npy'):
                                            tgt_name = os.path.basename(found).split(".")[0]
                                            tmp = np.load(found)
                                            on = np.repeat(v0[0], tmp.shape[0])
                                            to_concat = np.vstack((on, tmp)).T
                                            if tgt_name in results.keys():
                                                results[tgt_name] = np.concatenate(
                                                    (results[tgt_name], to_concat), axis=0)
                                            else:
                                                results[tgt_name] = to_concat
                                k += 1
                        j += 1
                for k, v in results.items():
                    x = rebase(v)
                    x = x[:, 0] - x[:, 1]
                    np.save(f'{tgt_path}/{w}-{o}/{filename}/{k}/{t}.npy', x)
                t += w - o


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


def _build(hasher, path, mode="duple"):
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
        for j, f in enumerate(glob.glob(f'./out/baseline/{i}/*.pkl')):
            with open(f, 'rb') as pkl_file:
                if j == 0:
                    out = np.array(pickle.load(pkl_file))
                else:
                    out = np.concatenate((out, np.array(pickle.load(pkl_file))), axis=0)
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
    # x = np.array(x)
    x -= np.mean(x, axis=0)

    plt.scatter(x[:, 0], x[:, 1])
    plt.xlim([-200, 200])
    plt.ylim([-200, 400])
    plt.show()

    cov_mat = np.cov(x, rowvar=False)
    eigen_values, eigen_vectors = np.linalg.eigh(cov_mat)
    sorted_index = np.argsort(eigen_values)[::-1]
    sorted_eigenvalue = eigen_values[sorted_index]
    sorted_eigenvectors = eigen_vectors[:, sorted_index]
    x = np.dot(sorted_eigenvectors.transpose(), x.transpose()).transpose()

    plt.scatter(x[:, 0], x[:, 1])
    plt.xlim([-200, 200])
    plt.ylim([-200, 400])
    plt.show()

    result = []
    for p in x:
        result.append(rotate([0, 0], p, 45))
    result = np.array(result)
    plt.scatter(result[:, 0], result[:, 1])
    plt.xlim([0, 200])
    plt.ylim([0, 200])
    plt.show()
    # print(result[:, 0] - result[:, 1])
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("job", type=str, choices=["build", "match"])
    parser.add_argument("--mode", type=str, choices=["triple"])
    parser.add_argument("--lookup", type=str)
    parser.add_argument("--src", type=str)
    parser.add_argument("--tgt", type=str)
    args = parser.parse_args()
    if args.job == "build":
        build(args.src, args.tgt, args.mode)
    elif args.job == "match":
        match(args.lookup, args.src, args.tgt, args.mode)

    # for i in sorted(glob.glob('./original/validation/*.json')):
    #     for j in sorted(glob.glob('./original/train/*.mid')):
    #         check_similarity(os.path.basename(i).split(".")[0], os.path.basename(j))
