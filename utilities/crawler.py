import os
import json
import copy
import errno
import requests
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from shutil import copyfile
from bs4 import BeautifulSoup

kern_dict = {
    "ID": "missing",
    "Composer": "missing",
    "Piece": "missing",
    "Opus": "missing",
    "No": "missing",
    "Movement": "missing",
    "Tempo": "missing",
    "Scholarly category": "missing",
    "URL": "missing",
    "mid": "missing",
    "krn": "missing",
}


def mkdir(path):
    if not os.path.exists(path):
        try:
            os.makedirs(path)
        except OSError as exc:
            if exc.errno != errno.EEXIST:
                raise


def get_data(composers):
    index = 0
    id_kern = []
    for suffix in composers:
        composer_url = "http://kern.humdrum.org/search?s=t&keyword=" + suffix
        url_content = requests.get(composer_url).content
        soup = BeautifulSoup(url_content, "html.parser")
        url_pool = soup.find_all(src="https://kern.humdrum.org/img/button-M.gif")
        composer_name = soup.find("i").text[1:]

        # print("Getting data of: " + composer_name)
        for link in tqdm(url_pool):
            midi_content = requests.get(link.parent.get("href")).content
            if midi_content[:4] == b"MThd":
                midi_file = Path.cwd() / "dataset/KernScores/unprocessed/midi/{}/{}.mid".format(composer_name, index)
                midi_file.parent.mkdir(parents=True, exist_ok=True)
                # mkdir("../../dataset/KernScores/unprocessed/midi/" + composer_name + "/")
                with midi_file.open(mode="wb") as f:
                    f.write(midi_content)
                kern_content = requests.get(link.parent.get("href")[:-4] + "kern&o=fullrep").content
                kern_file = Path.cwd() / "dataset/KernScores/unprocessed/kern/{}/{}.krn".format(composer_name, index)
                kern_file.parent.mkdir(parents=True, exist_ok=True)
                # mkdir("../../dataset/KernScores/unprocessed/kern/" + composer_name + "/")
                with kern_file.open(mode="wb") as f:
                    f.write(kern_content)
                krn = copy.deepcopy(kern_dict)
                with kern_file.open(mode="rb") as f:
                    for line in f.readlines():
                        line = line.decode("utf-8", "backslashreplace")
                        # if "!!!COM:" in line:
                        #     krn["Composer"] = line[8:].rstrip()
                        if "!!!OTL:" in line:
                            krn["Piece"] = line[8:].rstrip()
                        elif "!!!OPS:" in line:
                            krn["Opus"] = line[8:].rstrip()
                        elif "!!!ONM:" in line:
                            krn["No"] = line[8:].rstrip()
                        elif "!!!OMV:" in line:
                            krn["Movement"] = line[8:].rstrip()
                        elif "!!!OMD:" in line:
                            krn["Tempo"] = line[8:].rstrip()
                        elif "!!!SCT:" in line:
                            krn["Scholarly category"] = line[8:].rstrip()
                        elif "!!!SCT1:" in line:
                            krn["Scholarly category"] = line[9:].rstrip()
                krn["ID"] = str(index)
                krn["Composer"] = composer_name
                krn["URL"] = link.parent.get("href")[:-4] + "info"
                krn["mid"] = midi_file.as_posix()
                krn["krn"] = kern_file.as_posix()
                id_kern.append(krn)
                index += 1
    id_kern = json.dumps(id_kern, indent=4)
    index_file = Path.cwd() / "dataset/KernScores/unprocessed/index.json"
    index_file.parent.mkdir(parents=True, exist_ok=True)
    with index_file.open(mode="w") as j:
        j.write(id_kern)


def filtering(file_path):
    (Path.cwd() / "dataset/KernScores/CSQ/unfiltered/kern").mkdir(parents=True, exist_ok=True)
    (Path.cwd() / "dataset/KernScores/CSQ/unfiltered/midi").mkdir(parents=True, exist_ok=True)
    (Path.cwd() / "dataset/KernScores/CSQ/filtered/kern").mkdir(parents=True, exist_ok=True)
    (Path.cwd() / "dataset/KernScores/CSQ/filtered/midi").mkdir(parents=True, exist_ok=True)

    composers = ["Beethoven", "Haydn", "Mozart"]
    tempo_list = ["Allegro", "Allegretto", "Vivace", "Presto", "Moderato",
                  "allegro", "allegretto", "vivace", "presto", "moderato"]
    data = pd.read_json(file_path)
    data = data[data["Composer"].str.contains("|".join(composers)) & data["Piece"].str.contains("String Quartet")]
    filtered = data[data["Movement"].str.contains("1") & data["Tempo"].str.contains("|".join(tempo_list))]
    print("Number of string quartets of first movement and fast tempo: %d" % len(filtered))
    for _, row in filtered.iterrows():
        midi_file = Path(row["mid"])
        kern_file = Path(row["krn"])
        copyfile(midi_file, Path.cwd() / "dataset/KernScores/CSQ/filtered/midi/{}.mid".format(row["ID"]))
        copyfile(kern_file, Path.cwd() / "dataset/KernScores/CSQ/filtered/kern/{}.krn".format(row["ID"]))

    filtered_json = json.dumps(list(filtered.T.to_dict().values()), indent=4)
    with (Path.cwd() / "dataset/KernScores/CSQ/filtered/index.json").open(mode="w") as j:
        j.write(filtered_json)

    unfiltered = data[~data["ID"].isin(filtered["ID"].to_list())]
    print("Number of remaining string quartets: %d" % len(unfiltered))
    for _, row in unfiltered.iterrows():
        midi_file = Path(row["mid"])
        kern_file = Path(row["krn"])
        copyfile(midi_file, Path.cwd() / "dataset/KernScores/CSQ/unfiltered/midi/{}.mid".format(row["ID"]))
        copyfile(kern_file, Path.cwd() / "dataset/KernScores/CSQ/unfiltered/kern/{}.krn".format(row["ID"]))

    unfiltered_json = json.dumps(list(unfiltered.T.to_dict().values()), indent=4)
    with (Path.cwd() / "dataset/KernScores/CSQ/unfiltered/index.json").open(mode="w") as j:
        j.write(unfiltered_json)


def main():
    composers_suffix = [
        "Adam+de+la+Halle",
        "Alkan",
        "Bach+Johann",
        "Banchieri",
        "Beethoven",
        "Billings",
        "Bossi",
        "Brahms",
        "Buxtehude",
        "Byrd",
        "Chopin",
        "Clementi",
        "Corelli",
        "Dufay",
        "Dunstable",
        "Field+John",
        "Flecha",
        "Foster",
        "Frescobaldi",
        "Gershwin",
        "Giovannelli",
        "Grieg",
        "Haydn",
        "Himmel+Friedrich",
        "Hummel+Johann",
        "Isaac",
        "\Ives%20Charles",
        "Joplin",
        "Josquin",
        "Landini",
        "Lassus",
        "Liszt",
        "MacDowell",
        "Mendelssohn",
        "Monteverdi",
        "Mozart",
        "Pachelbel",
        "Prokofiev",
        "Ravel",
        "Scarlatti",
        "Schubert",
        "Schumann",
        "Scriabin",
        "Sinding",
        "Sousa",
        "Turpin",
        "Scarlatti",
        "Vecchi",
        "Victoria",
        "Vivaldi",
        "Weber+Carl",
    ]
    candidates = ["Beethoven", "Haydn", "Mozart"]
    get_data(composers_suffix)
    filtering(Path.cwd() / "dataset/KernScores/unprocessed/index.json")
