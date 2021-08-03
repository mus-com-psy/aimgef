from multiprocessing import Pool, cpu_count
import os
import errno
import json
import sqlite3
import argparse
import pickle
import time

tMin = 0.5
tMax = 2
pMin = 1
pMax = 6
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
lookup_tables = []
query_tables = []
for key, value in maestro["split"].items():
    if value == "train":
        lookup_tables.append(key)
    else:
        query_tables.append(key)


def check_similarity(query, trans, bin_size=0.25):
    trans = sorted(trans, key=lambda x: x[1])
    tmp = []
    for i, t in enumerate(trans):
        tmp.append((i, len([x for x in trans if t <= x[1] <= t + bin_size])))
    max_i = max(tmp, key=lambda x: x[1])
    trans = trans[max_i[0]:max_i[0] + max_i[1]]
    trans_count = {}
    for i in trans:
        trans_count[i[0]] = trans_count.get(i[0], 0) + 1
    num_unique_entries = sum([item[1] for item in query.items()])
    result = 0
    for k, v in query.items():
        c = trans_count.get(k, 0)
        if c > v:
            result += v / num_unique_entries
        else:
            result += k / num_unique_entries
    return result



def mkdir(filename):
    if not os.path.exists(os.path.dirname(filename)):
        try:
            os.makedirs(os.path.dirname(filename))
        except OSError as exc:  # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise


def entry(pts, mode, target, sp, t_min=tMin, t_max=tMax, p_min=pMin, p_max=pMax):
    """
    :param pts: a list of points
    :param mode: build or match
    :param target: target file
    :param sp: split
    :param t_min: minimum time difference
    :param t_max: maximum time difference
    :param p_min: minimum pitch difference
    :param p_max: maximum pitch difference
    :return: a list of entries
    """
    l_con = sqlite3.connect("./data/maestro.db", timeout=30.0)
    l_cur = l_con.cursor()
    if mode == "build":
        for i in range(len(pts) - 2):
            v_0 = pts[i]
            entries = []
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
                            entries.append((entry2index[s_0 + s_1 + s_2], v_0[0], target))
            l_cur.executemany(f'INSERT INTO {sp} VALUES (?,?,?);', entries)
            l_con.commit()

    elif mode == "match":
        start = time.time()
        print(f'[SQL] Querying {target}')
        target_sql = f'SELECT t.entry, (t.ontime - q.ontime) AS time FROM ' \
                     f'(SELECT entry, ontime from train WHERE excerpt = 0) AS t ' \
                     f'LEFT JOIN (SELECT entry, ontime from {sp} WHERE excerpt = {target} AND ontime > 0 AND ontime < 8) AS q USING(entry) ' \
                     f'UNION ALL SELECT t.entry, (t.ontime - q.ontime) AS time FROM ' \
                     f'(SELECT entry, ontime from {sp} WHERE excerpt = {target} AND ontime > 0 AND ontime < 8) AS q ' \
                     f'LEFT JOIN (SELECT entry, ontime from train WHERE excerpt = 0) AS t USING(entry) WHERE t.entry IS NULL;'
        l_cur.execute(target_sql)
        print(f'[SQL] Fetching {target}')
        trans = l_cur.fetchall()
        print(f'[SQL] Fetched {target}, len = {len(trans)}')
        l_cur.execute(f'SELECT entry, ontime from {sp} WHERE excerpt = {target} AND ontime > 0 AND ontime < 8')
        query_entries = l_cur.fetchall()
        query = {}
        for i in query_entries:
            query[i[0]] = query.get(i[0], 0) + 1
        print(f'[SIM] Check similarity of {target}')
        sim = check_similarity(query, trans)
        # os.system(f'sqlite3 -header -csv ./data/maestro.db "{target_sql}" > ./data/match/{target}.csv')
        # l_cur.execute(target_sql)
        # matched = l_cur.fetchall()
        # with open(f'./data/match/{target}.json', "w") as fp:
        #     json.dump(matched, fp)
        end = time.time()
        print(f'[ELAPSED]\t{sim:.2f}\t{end - start}s')

    else:
        print("[ERROR] Invalid model.")

    print(f'[DONE]\t{target}')
    l_cur.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("job", type=str, choices=["build", "match"])
    args = parser.parse_args()
    job_list = []
    part = 100
    with open("./maestro-v3.0.0/maestro-v3.0.0.json") as json_file:
        maestro = json.load(json_file)
    if args.job == "build":
        connection = sqlite3.connect(f'./data/maestro-{part}.db')
        cursor = connection.cursor()
        sp_count = {"train": 0, "validation": 0, "test": 0}
        for split in ["train", "validation", "test"]:
            if sp_count["train"] > part or \
                    sp_count["validation"] > int(part / 20) or \
                    sp_count["validation"] > int(part / 20):
                break
            cursor.execute(f'CREATE TABLE IF NOT EXISTS {split}(entry INTEGER, ontime REAL, excerpt INTEGER)')
            sp_count[split] += 1
        connection.commit()
        cursor.close()
        for key, value in maestro["midi_filename"].items():
            src = f'{os.path.splitext(value)[0]}.pkl'
            with open(f'./maestro-v3.0.0/{src}', 'rb') as f:
                points = pickle.load(f)
                job_list.append([points, "build", key, maestro["split"][key]])

    elif args.job == "match":
        # connection = sqlite3.connect("./data/match.db")
        # cursor = connection.cursor()
        for key, value in maestro["split"].items():
            if value in ["validation", "test"]:
                src = f'{os.path.splitext(maestro["midi_filename"][key])[0]}.json'
                with open(f'./maestro-v3.0.0/{src}') as json_file:
                    # sql = f'CREATE TABLE IF NOT EXISTS _{key}' \
                    #       f'(entry INTEGER, target INTEGER, q_ontime REAL, t_ontime REAL)'
                    # cursor.execute(sql)
                    points = json.load(json_file)
                    points = sorted([list(x) for x in set(tuple(x) for x in points)], key=lambda x: (x[0], x[1]))
                job_list.append([points, "match", key, maestro["split"][key]])
        # cursor.execute(f'CREATE TABLE IF NOT EXISTS entry_count(excerpt INTEGER, entry_count INTEGER)')
        # connection.commit()
        # cursor.close()

    entry(job_list[0][0], job_list[0][1], job_list[0][2], job_list[0][3])
    # for job in job_list:
    #     entry(job[0], job[1], job[2], job[3])
    # with Pool(cpu_count() - 1) as p:
    #     p.starmap(entry, job_list)
