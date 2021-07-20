from multiprocessing import Pool, cpu_count
import os
import errno
import json
import sqlite3
import argparse

tMin = 0.25
tMax = 1
pMin = 1
pMax = 7
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
    l_con = sqlite3.connect("./data/lookup.db", timeout=30.0)
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
                            entries.append((entry2index[s_0 + s_1 + s_2], v_0[0]))
            l_cur.executemany(f'INSERT INTO _{target} VALUES (?,?);', entries)
            l_con.commit()

    elif mode == "match":
        match = {}
        q_con = sqlite3.connect("./data/query.db", timeout=30.0)
        q_cur = q_con.cursor()
        m_con = sqlite3.connect("./data/match.db", timeout=30.0)
        m_cur = m_con.cursor()
        # m_cur.execute(f'INSERT INTO entry_count VALUES ({target},{len(query)});')
        for i in lookup_tables:
            match[i] = 0
            q_cur.execute(f'SELECT * FROM _{target};')
            for q in q_cur:
                # l_cur.execute(f'SELECT * FROM _{i} WHERE entry = {q[0]}')
                l_cur.execute(f'SELECT COUNT(*) '
                              f'FROM _{i} '
                              f'WHERE entry = {q[0]} AND ontime >= {q[1] * 0.75} AND ontime <= {q[1] * 1.25};')
                # m_cur.executemany(f'INSERT INTO _{target} VALUES (?,?,?,?);', [(q[0], i, q[1], x[1]) for x in l_cur])
                # m_con.commit()
                c = l_cur.fetchall()
                match[i] += c[0][0]
            print(f'\t[PROGRESS]\t{target} - {i}')
        with open(f'./data/match/{target}.json', "w") as fp:
            json.dump(match, fp)

        m_cur.close()
        q_cur.close()

    else:
        print("[ERROR] Invalid model.")

    print(f'[DONE]\t{target}')
    l_cur.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("job", type=str, choices=["build", "match"])
    args = parser.parse_args()
    job_list = []
    with open("./maestro-v3.0.0/maestro-v3.0.0.json") as json_file:
        maestro = json.load(json_file)
    if args.job == "build":
        connection = sqlite3.connect("./data/lookup.db")
        # connection = sqlite3.connect("./data/query.db")
        cursor = connection.cursor()

        for key, value in maestro["midi_filename"].items():
            # for key, value in maestro["split"].items():
            #     if value in ["validation", "test"]:
            #         src = f'{os.path.splitext(maestro["midi_filename"][key])[0]}.json'
            src = f'{os.path.splitext(value)[0]}.json'
            with open(f'./maestro-v3.0.0/{src}') as json_file:
                cursor.execute(f'CREATE TABLE IF NOT EXISTS _{key}(entry INTEGER, ontime REAL)')
                points = json.load(json_file)
                points = sorted([list(x) for x in set(tuple(x) for x in points)], key=lambda x: (x[0], x[1]))
                job_list.append([points, "build", key])
        connection.commit()
        cursor.close()

    elif args.job == "match":
        connection = sqlite3.connect("./data/match.db")
        cursor = connection.cursor()
        for key, value in maestro["split"].items():
            if value in ["validation", "test"]:
                src = f'{os.path.splitext(maestro["midi_filename"][key])[0]}.json'
                with open(f'./maestro-v3.0.0/{src}') as json_file:
                    sql = f'CREATE TABLE IF NOT EXISTS _{key}' \
                          f'(entry INTEGER, target INTEGER, q_ontime REAL, t_ontime REAL)'
                    cursor.execute(sql)
                    points = json.load(json_file)
                    points = sorted([list(x) for x in set(tuple(x) for x in points)], key=lambda x: (x[0], x[1]))
                job_list.append([points, "match", key])
        cursor.execute(f'CREATE TABLE IF NOT EXISTS entry_count(excerpt INTEGER, entry_count INTEGER)')
        connection.commit()
        cursor.close()

    # for job in job_list:
    #     entry(job[0], job[1], job[2])
    with Pool(cpu_count() - 1) as p:
        p.starmap(entry, job_list)
