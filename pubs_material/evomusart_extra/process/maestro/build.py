from multiprocessing import Pool, cpu_count
import os
import errno
import numpy as np
import glob
import json
import sqlite3
import argparse

tMin = 0.25
tMax = 1
pMin = 1
pMax = 7
# print([f'+{x}.{y}' for x in range(1, 40) for y in range(0, 10)] + ["+40.0"] + [f'-{x}.{y}' for x in range(1, 41) for y in range(0, 10)] + ["-40.0"])
# print([f'+{x}' for x in range(1, 13)] + [f'-{x}' for x in range(1, 13)])
entry2index = {}
index2entry = {}
count = 0
for s0 in [f'+{x}' for x in range(pMin, pMax + 1)] + [f'-{x}' for x in range(pMin, pMax + 1)]:
    for s1 in [f'+{x}' for x in range(pMin, pMax + 1)] + [f'-{x}' for x in range(pMin, pMax + 1)]:
        for s2 in [f'+{x}.{y}' for x in range(1, int(tMax // tMin)) for y in range(0, 10)] + \
                  [f'+{int(tMax // tMin)}.0'] + \
                  [f'-{x}.{y}' for x in range(1, int(tMax // tMin)) for y in range(0, 10)] + \
                  [f'-{int(tMax // tMin)}.0']:
            entry2index[s0 + s1 + s2] = count
            index2entry[count] = s0 + s1 + s2
            count += 1

with open("./maestro-v3.0.0/maestro-v3.0.0.json") as json_file:
    maestro = json.load(json_file)
match_table = []
for key, value in maestro["split"].items():
    if value == "train":
        match_table.append(key)


def mkdir(filename):
    if not os.path.exists(os.path.dirname(filename)):
        try:
            os.makedirs(os.path.dirname(filename))
        except OSError as exc:  # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise


def entry(pts, mode, target, t_min=tMin, t_max=tMax, p_min=pMin, p_max=pMax):
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
    con = sqlite3.connect("./data/lookup.db", timeout=30.0)
    m_con = sqlite3.connect("./data/match.db", timeout=30.0)
    cur = con.cursor()
    m_cur = m_con.cursor()
    nh = 0
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
                            """
                            Version 1
                            """
                            # filename = f'./data/lookup/{s_0}/{s_1}/{s_2}/{target}.npy'
                            # mkdir(filename)
                            # if os.path.isfile(filename):
                            #     np.save(filename, np.append(np.load(filename), v_0[0]))
                            # else:
                            #     np.save(filename, np.array([v_0[0]]))
                            """
                            Version 2
                            """
                            cur.execute(f'INSERT INTO _{target} VALUES ({entry2index[s_0 + s_1 + s_2]}, {v_0[0]})')
                            con.commit()
                        elif mode == "match":
                            """
                            Version 1
                            """
                            # if os.path.isdir(f'./data/lookup/{s_0}/{s_1}/{s_2}'):
                            #     match["nm"] += 1
                            #     for f in glob.glob(f'./data/lookup/{s_0}/{s_1}/{s_2}/*.npy'):
                            #         name = os.path.basename(f).split(".")[0]
                            #         on = np.load(f)
                            #         if name in match["match"].keys():
                            #             match["match"][name] += [[v_0[0], o] for o in on.tolist()]
                            #         else:
                            #             match["match"][name] = [[v_0[0], o] for o in on.tolist()]
                            """
                            Version 2
                            """
                            for t in match_table:
                                cur.execute(f'SELECT * FROM _{t} WHERE entry = {entry2index[s_0 + s_1 + s_2]};')
                                result = cur.fetchall()
                                if result:
                                    nh += 1
                                    to_insert = [(entry2index[s_0 + s_1 + s_2], t, v_0[0], x[1]) for x in result]
                                    m_cur.executemany(f'INSERT INTO _{target} VALUES (?,?,?,?)', to_insert)
                                    m_con.commit()

                        else:
                            print("[ERROR] Invalid model.")
        if mode == "match":
            print(f'\t[REPORT] {target}\t{i}/{len(pts)}')
    # if mode == "match":
    #     mkdir(f'./data/match/{target}.json')
    #     with open(f'./data/match/{target}.json', "w") as fp:
    #         json.dump(match, fp)
    if mode == "match":
        m_cur.execute(f'INSERT INTO entry_count VALUES ({target}, {nh})')
    print(f'[DONE]\t{target}')
    cur.close()
    m_cur.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("job", type=str, choices=["build", "match"])
    args = parser.parse_args()
    if args.job == "build":
        """
        Version 1
        """
        # job_list = []
        # with open("./maestro-v3.0.0/maestro-v3.0.0.json") as json_file:
        #     maestro = json.load(json_file)
        # for key, value in maestro["split"].items():
        #     if value == "train":
        #         src = f'{os.path.splitext(maestro["midi_filename"][key])[0]}.json'
        #         with open(f'./maestro-v3.0.0/{src}') as json_file:
        #             points = json.load(json_file)
        #             points = sorted([list(x) for x in set(tuple(x) for x in points)], key=lambda x: x[0])
        #             job_list.append([points, "build", key])
        #
        # with Pool(cpu_count() - 1) as p:
        #     p.starmap(entry, job_list)

        """
        Version 2
        """
        with open("./data/tmp.txt", "r") as txt_file:
            done = [x[7:-1] for x in txt_file.readlines()]

        connection = sqlite3.connect("./data/lookup.db")
        cursor = connection.cursor()
        with open("./maestro-v3.0.0/maestro-v3.0.0.json") as json_file:
            maestro = json.load(json_file)
        job_list = []
        for key, value in maestro["split"].items():
            if value == "train":
                if not (key in done):
                    src = f'{os.path.splitext(maestro["midi_filename"][key])[0]}.json'
                    with open(f'./maestro-v3.0.0/{src}') as json_file:
                        # cursor.execute(f'CREATE TABLE IF NOT EXISTS _{key}(entry INTEGER, ontime REAL)')
                        points = json.load(json_file)
                        points = sorted([list(x) for x in set(tuple(x) for x in points)], key=lambda x: (x[0], x[1]))
                    job_list.append([points, "build", key])
        connection.commit()
        cursor.close()

        with Pool(cpu_count() - 1) as p:
            p.starmap(entry, job_list)
    elif args.job == "match":
        connection = sqlite3.connect("./data/match.db")
        cursor = connection.cursor()
        with open("./maestro-v3.0.0/maestro-v3.0.0.json") as json_file:
            maestro = json.load(json_file)
        job_list = []
        for key, value in maestro["split"].items():
            if value in ["validation", "test"]:
                src = f'{os.path.splitext(maestro["midi_filename"][key])[0]}.json'
                with open(f'./maestro-v3.0.0/{src}') as json_file:
                    cursor.execute(f'CREATE TABLE IF NOT EXISTS _{key}(entry INTEGER, target INTEGER, q_ontime REAL, t_ontime REAL)')
                    points = json.load(json_file)
                    points = sorted([list(x) for x in set(tuple(x) for x in points)], key=lambda x: (x[0], x[1]))
                job_list.append([points, "match", key])
        cursor.execute(f'CREATE TABLE IF NOT EXISTS entry_count(excerpt INTEGER, entry_count INTEGER)')
        connection.commit()
        cursor.close()
        with Pool(cpu_count() - 1) as p:
            p.starmap(entry, job_list)
