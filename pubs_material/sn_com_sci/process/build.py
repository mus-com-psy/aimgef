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
        print(f'[HASH] {name}')
        with open(file) as json_file:
            pts = json.load(json_file)
            pts = sorted([list(x) for x in set(tuple(x) for x in pts)], key=lambda x: x[0])
        points.append([pts, "build", name])

    with Pool(cpu_count() - 1) as p:
        p.starmap(entry, points)
    print("[DONE]")
