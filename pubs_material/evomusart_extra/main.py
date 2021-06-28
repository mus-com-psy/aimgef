import json
import glob
import os
import errno


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


if __name__ == '__main__':
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
