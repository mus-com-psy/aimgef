from multiprocessing import Pool, cpu_count
import os
import errno
import numpy as np
import glob
import json


def mkdir(filename):
    if not os.path.exists(os.path.dirname(filename)):
        try:
            os.makedirs(os.path.dirname(filename))
        except OSError as exc:  # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise


def entry(pts, mode, target, t_min=0.1, t_max=4, p_min=1, p_max=12):
    """
    :param pts: a list of points
    :param mode: build or match
    :param target: target file
    :param t_min: minimum time difference
    :param t_max: maximum time difference
    :param p_min: minimum pitch difference
    :param p_max: maximum pitch difference
    :return: a list of entries
    """
    match = {"nm": 0, "match": {}}
    for i in range(len(pts) - 2):
        v_0 = pts[i]
        for j in range(i + 1, len(pts) - 1):
            v_1 = pts[j]
            td_0 = v_1[0] - v_0[0]
            pd_0 = v_1[1] - v_0[1]
            apd_0 = abs(pd_0)
            if t_min < td_0 < t_max and p_min <= apd_0 <= p_max:
                for k in range(j + 1, len(pts)):
                    v_2 = pts[k]
                    td_1 = v_2[0] - v_1[0]
                    pd_1 = v_2[1] - v_1[1]
                    apd_1 = abs(pd_1)
                    if t_min < td_1 < t_max and p_min <= apd_1 <= p_max:
                        if pd_0 < 0:
                            s_0 = f'-{int(apd_0)}'
                        else:
                            s_0 = f'+{int(apd_0)}'
                        if pd_1 < 0:
                            s_1 = f'-{int(apd_1)}'
                        else:
                            s_1 = f'+{int(apd_1)}'
                        if td_0 >= td_1:
                            tdr = float(round((td_0 / td_1) * 10) / 10)
                            s_2 = f'+{tdr:.1f}'
                        else:
                            tdr = float(round((td_1 / td_0) * 10) / 10)
                            s_2 = f'-{tdr:.1f}'
                        if mode == "build":
                            filename = f'./data/lookup/{s_0}/{s_1}/{s_2}/{target}.npy'
                            mkdir(filename)
                            if os.path.isfile(filename):
                                np.save(filename, np.append(np.load(filename), v_0[0]))
                            else:
                                np.save(filename, np.array([v_0[0]]))
                        elif mode == "match":
                            if os.path.isdir(f'./data/lookup/{s_0}/{s_1}/{s_2}'):
                                match["nm"] += 1
                                for f in glob.glob(f'./data/lookup/{s_0}/{s_1}/{s_2}/*.npy'):
                                    name = os.path.basename(f).split(".")[0]
                                    on = np.load(f)
                                    if name in match["match"].keys():
                                        match["match"][name] += [[v_0[0], o] for o in on.tolist()]
                                    else:
                                        match["match"][name] = [[v_0[0], o] for o in on.tolist()]
                        else:
                            print("[ERROR] Invalid model.")
    if mode == "match":
        mkdir(f'./data/match/{target}.json')
        with open(f'./data/match/{target}.json', "w") as fp:
            json.dump(match, fp)


if __name__ == '__main__':
    job_list = []
    with open("./maestro-v3.0.0/maestro-v3.0.0.json") as json_file:
        maestro = json.load(json_file)
    for key, value in maestro["split"].items():
        if value == "train":
            src = f'{os.path.splitext(maestro["midi_filename"][key])[0]}.json'
            with open(f'./maestro-v3.0.0/{src}') as json_file:
                points = json.load(json_file)
                points = sorted([list(x) for x in set(tuple(x) for x in points)], key=lambda x: x[0])
                job_list.append([points, "build", key])

    with Pool(cpu_count() - 1) as p:
        p.starmap(entry, points)
    print("[DONE]")
