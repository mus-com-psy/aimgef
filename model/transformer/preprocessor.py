import json
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from midi_io import MIDI


def get_maestro_midi_list(split):
    maestro_json = pd.read_json((Path.cwd() / "dataset/maestro-v2.0.0/maestro-v2.0.0.json").as_posix())
    midi_list = maestro_json.loc[maestro_json['split'] == split]['midi_filename'].values.tolist()
    return midi_list


def main(style, representation, length, step_size, time_granularity, ignore_velocity, merge_tracks, transposition):
    if representation == "token":
        if style == "CSQ":
            train_index = 0
            valid_index = 0
            index_filename = Path.cwd() / "dataset/KernScores/CSQ/filtered/index.json"
            with index_filename.open(mode="r") as f:
                midi_list = json.load(f)
            for midi in tqdm(midi_list):
                mid = MIDI(midi['mid'], decoder="pretty_midi")
                if mid.num_tracks != 4:
                    continue
                move = range(-5, 7) if transposition else [0]
                valid_set = [1211, 1219, 1240, 1827, 1893, 2322, 2368]  # 1207, 1227, 1836, 1901, 1908, 2340, 2325
                split = "train" if midi["ID"] not in valid_set else "validation"
                for i in move:
                    excerpts = mid.slice_events(length,
                                                step_size,
                                                time_granularity,
                                                ignore_velocity,
                                                merge_tracks,
                                                mid.transposition(i)[1])
                    for j, e in enumerate(excerpts):
                        file_index = train_index if split == "train" else valid_index
                        filename = Path.cwd() / "dataset/{}/{}/{}/{}/{}/{}.npy".format(style,
                                                                                       representation,
                                                                                       length,
                                                                                       split,
                                                                                       file_index // 1000,
                                                                                       file_index)
                        filename.parent.mkdir(parents=True, exist_ok=True)
                        # MIDI.plot_tokens(e, time_granularity)
                        np.save(filename.as_posix(), np.array(e))
                        if split == "train":
                            train_index += 1
                        else:
                            valid_index += 1
        elif style == "CPI":
            for split in ["test", "validation", "train"]:
                file_index = 0
                midi_list = get_maestro_midi_list(split)
                for midi in tqdm(midi_list):
                    mid = MIDI((Path.cwd() / "dataset/maestro-v2.0.0" / midi).as_posix(), decoder="pretty_midi")
                    if mid.num_tracks != 1:
                        continue
                    move = range(-5, 7) if transposition else [0]
                    for i in move:
                        excerpts = mid.slice_events(length,
                                                    step_size,
                                                    time_granularity,
                                                    ignore_velocity,
                                                    merge_tracks,
                                                    mid.transposition(i)[1])
                        for e in excerpts:
                            filename = Path.cwd() / "dataset/{}/{}/{}/{}/{}/{}.npy".format(style,
                                                                                           representation,
                                                                                           length,
                                                                                           split,
                                                                                           file_index // 1000,
                                                                                           file_index)
                            filename.parent.mkdir(parents=True, exist_ok=True)
                            # MIDI.plot_tokens(e[0], time_granularity)
                            # print(midi)
                            np.save(filename.as_posix(), np.array(e))
                            file_index += 1


if __name__ == '__main__':
    durations = [0, 1 / 32, 1 / 16, 1 / 12, 1 / 8, 1 / 6, 3 / 16, 1 / 4, 1 / 3, 3 / 8, 1 / 2, 2 / 3, 3 / 4, 1]
    main(style='CSQ',
         representation='token',
         length=2048,
         step_size=1024,
         time_granularity=durations,
         ignore_velocity=True,
         merge_tracks=True,
         transposition=True)
    # main(style='CPI',
    #      representation='token',
    #      length=2048,
    #      step_size=512,
    #      time_granularity=False,
    #      ignore_velocity=False,
    #      merge_tracks=False,
    #      transposition=False)
