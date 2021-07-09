from get_entry import entry
from multiprocessing import Pool, cpu_count
import json
import glob
import argparse
import os
import errno


def mkdir(filename):
    if not os.path.exists(os.path.dirname(filename)):
        try:
            os.makedirs(os.path.dirname(filename))
        except OSError as exc:  # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=str)
    args = parser.parse_args()
    points = []
    for file in glob.glob(f'{args.src}/*.json'):
        name = os.path.basename(file).split(".")[0]
        with open(file) as json_file:
            pts = json.load(json_file)
            pts = sorted([list(x) for x in set(tuple(x) for x in pts)], key=lambda x: x[0])
        t = 0
        max_t = max([p[0] for p in pts]) - 8
        while t <= max_t:
            points.append([[p for p in pts if t <= p[0] < t + 8], "match", f'{name}/{t}'])
            t += 8

    with Pool(cpu_count() - 1) as p:
        p.starmap(entry, points)
    print("[DONE]")
