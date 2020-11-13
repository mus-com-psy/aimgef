import copy
import datetime
import errno
import json
import math
import os
from pathlib import Path
import random

from easydict import EasyDict as edict
import matplotlib.pyplot as plt
import mido
import numpy as np
import pandas as pd
import pretty_midi
from tqdm import tqdm


class MIDI:
    def __init__(self, filename, decoder):
        self.filename = filename
        self.decoder = decoder
        if decoder == "pretty_midi":
            self.source = pretty_midi.PrettyMIDI(filename)
            self.resolution = self.source.resolution * 4
            tracks = self.source.instruments
            self.num_tracks = len(tracks)
            self.notes = [self.to_note(i, note) for i, track in enumerate(tracks)
                          for note in track.notes]
            self.events = [event for i, track in enumerate(tracks)
                           for note in track.notes
                           for event in self.to_events(i, note)]
            self.events.sort(key=lambda x: (x.time, x.track, x.pitch))

    def to_tick(self, time):
        return int(self.source.time_to_tick(time))

    def to_time(self, tick):
        return float(self.source.tick_to_time(tick))

    def to_note(self, i, note):
        return edict({"track": i,
                      "pitch": note.pitch,
                      "start": self.to_tick(note.start),
                      "end": self.to_tick(note.end),
                      "velocity": note.velocity})

    def to_events(self, i, note):
        on = edict({"track": i,
                    "action": "on",
                    "pitch": note.pitch,
                    "time": self.to_tick(note.start),
                    "velocity": note.velocity})
        off = edict({"track": i,
                     "action": "off",
                     "pitch": note.pitch,
                     "time": self.to_tick(note.end),
                     "velocity": note.velocity})
        return on, off

    def transposition(self, move):
        notes = copy.deepcopy(self.notes)
        events = copy.deepcopy(self.events)
        for note in notes:
            note["pitch"] += move
            self.validate_pitch(note["pitch"])
        for event in events:
            event["pitch"] += move
            self.validate_pitch(event["pitch"])
        return notes, events

    def validate_pitch(self, pitch):
        assert pitch >= 0, "Encountered invalid pitch transposition: \n\tFILE: {file}".format(file=self.filename)
        assert pitch <= 127, "Encountered invalid pitch transposition: \n\tFILE: {file}".format(file=self.filename)

    def get_time_tokens(self, ticks, time_granularity):
        output = []
        if time_granularity:
            quotient, remainder = divmod(ticks, self.resolution)
            for _ in range(quotient):
                output.append(len(time_granularity) - 1)
            if remainder:
                closest = min(range(len(time_granularity)),
                              key=lambda i: abs(time_granularity[i] - remainder / self.resolution))
                if closest:
                    output.append(closest)
        else:
            quotient, remainder = divmod(self.to_time(ticks), 1)
            for _ in range(int(quotient)):
                output.append(100)
            if remainder:
                closest = int(remainder / 0.01)
                if closest:
                    output.append(closest)
        return output

    def slice_events(self, length, step_size, time_granularity, ignore_velocity, merge_tracks, tracks):
        """
        :param length:
        :param step_size:
        :param time_granularity:
        :param ignore_velocity:
        :param merge_tracks:
        :param tracks:
        :return:

        padding = 0
        on: 1 - 128
        off: 129 - 256
        if time_granularity:
            durations: 257 - 269
            velocity: 270 - 301
        else:
            durations: 257 - 356
            velocity: 357 - 388
        """
        outputs = []
        start_token_on = 1
        start_token_off = 129
        start_token_time = 257
        start_token_velocity = 270 if time_granularity else 357
        if merge_tracks:
            for i in range(len(tracks) // step_size):
                num_tracks = self.num_tracks
                tokens = []
                velocity = 0
                keyboard = [[False for _ in range(128)] for _ in range(num_tracks)]
                current_time = tracks[i * step_size].time
                for event in tracks[i * step_size:]:
                    if event.action == "on":
                        if not keyboard[event.track][event.pitch]:
                            if not tokens:
                                current_time = event.time
                            keyboard[event.track][event.pitch] = True
                            time_tokens = self.get_time_tokens(event.time - current_time, time_granularity)
                            tokens += [token + start_token_time - 1 for token in time_tokens]
                            if not ignore_velocity:
                                assert event.velocity != 0, "Velocity cannot be 0."
                                if math.ceil(event.velocity / 4) != velocity:
                                    velocity = math.ceil(event.velocity / 4)
                                    tokens.append(start_token_velocity + velocity - 1)
                            tokens.append(start_token_on + event.pitch)
                            current_time = event.time
                        else:
                            continue
                    if event.action == "off":
                        if keyboard[event.track][event.pitch]:
                            keyboard[event.track][event.pitch] = False
                            time_tokens = self.get_time_tokens(event.time - current_time, time_granularity)
                            tokens += [token + start_token_time - 1 for token in time_tokens]
                            tokens.append(start_token_off + event.pitch)
                            current_time = event.time
                        else:
                            continue
                    if len(tokens) >= length:
                        tokens = tokens[:length]
                        break
                if len(tokens) < length:
                    for _ in range(length - len(tokens)):
                        tokens.append(0)
                outputs.append(tokens)
        else:
            for i in range(len(tracks) // step_size):
                num_tracks = self.num_tracks
                tokens = [[] for _ in range(num_tracks)]
                velocity = [0 for _ in range(num_tracks)]
                keyboard = [[False for _ in range(128)] for _ in range(num_tracks)]
                current_time = [tracks[i * step_size].time for _ in range(num_tracks)]
                for event in tracks[i * step_size:]:
                    if event.action == "on":
                        if not keyboard[event.track][event.pitch]:
                            if not tokens[event.track]:
                                current_time[event.track] = event.time
                            keyboard[event.track][event.pitch] = True
                            time_tokens = self.get_time_tokens(event.time - current_time[event.track], time_granularity)
                            tokens[event.track] += [token + start_token_time - 1 for token in time_tokens]
                            if not ignore_velocity:
                                assert event.velocity != 0, "Velocity cannot be 0."
                                if math.ceil(event.velocity / 4) != velocity[event.track]:
                                    velocity[event.track] = math.ceil(event.velocity / 4)
                                    tokens[event.track] += [start_token_velocity + velocity[event.track] - 1]
                            tokens[event.track].append(start_token_on + event.pitch)
                            current_time[event.track] = event.time
                        else:
                            continue
                    if event.action == "off":
                        if keyboard[event.track][event.pitch]:
                            keyboard[event.track][event.pitch] = False
                            time_tokens = self.get_time_tokens(event.time - current_time[event.track], time_granularity)
                            tokens[event.track] += [token + start_token_time - 1 for token in time_tokens]
                            tokens[event.track].append(start_token_off + event.pitch)
                            current_time[event.track] = event.time
                        else:
                            continue
                    if len(tokens[event.track]) >= length:
                        tokens[event.track] = tokens[event.track][:length]
                        break
                for track_tokens in tokens:
                    if len(track_tokens) < length:
                        for _ in range(length - len(track_tokens)):
                            track_tokens.append(0)
                outputs.append(tokens)
        return outputs

    @staticmethod
    def plot_tokens(tokens, time_granularity):
        if time_granularity:
            scalar = 16
            total_time = sum([time_granularity[t - 256] for t in tokens if 257 <= t <= 257 + len(time_granularity) - 1])
            image = np.zeros((128, int(total_time * scalar)))
            keyboard = np.zeros(128)
            time = 0
            for token in tokens:
                if 1 <= token <= 128:
                    keyboard[token] = 1
                elif 129 <= token <= 256:
                    keyboard[token - 128] = 0
                elif 257 <= token <= 269:
                    for note_on in np.where(keyboard == 1)[0]:
                        image[note_on, time:time + int(time_granularity[token - 256] * scalar)] = 1
                    time += int(time_granularity[token - 256] * scalar)

            plt.imshow(image, origin='lower')
            plt.show()
        else:
            scalar = 1
            total_time = sum([(t - 256) for t in tokens if 257 <= t <= 356])
            image = np.zeros((128, int(total_time * scalar) + 1))
            keyboard = np.zeros(128)
            time = 0
            for token in tokens:
                if 1 <= token <= 128:
                    keyboard[token] = 1
                elif 129 <= token <= 256:
                    keyboard[token - 128] = 0
                elif 257 <= token <= 356:
                    for note_on in np.where(keyboard == 1)[0]:
                        image[note_on, time:time + int((token - 256) * scalar)] = 1
                    time += int((token - 256) * scalar)
            plt.imshow(image, origin='lower', aspect="auto")
            plt.show()

    @staticmethod
    def to_midi(tracks, time_granularity):
        midi = pretty_midi.PrettyMIDI(resolution=480, initial_tempo=120)
        on_range = range(1, 129)
        off_range = range(129, 257)
        time_range = range(257, 270) if time_granularity else range(257, 357)
        velocity_range = range(270, 302) if time_granularity else range(357, 389)
        for track in tracks:
            instrument = pretty_midi.Instrument(program=0)
            onset = {n: [] for n in range(128)}
            offset = {n: [] for n in range(128)}
            time = 0
            velocity = 64
            for token in track:
                if token in on_range:
                    onset[token - 1].append((time, velocity))
                elif token in off_range:
                    offset[token - 129].append(time)
                elif token in time_range:
                    time += time_granularity[token - 256] if time_granularity else (token - 256) * 0.01
                elif token in velocity_range:
                    velocity = (token - (269 if time_granularity else 356)) * 4 - 1

            for i in range(128):
                while onset[i] and offset[i]:
                    on = onset[i].pop(0)
                    while offset[i][0] - on[0] <= 0:
                        _ = offset[i].pop(0)
                        if not offset[i]:
                            break
                    if not offset[i]:
                        break
                    if 0 < offset[i][0] - on[0] <= (4 if time_granularity else 2):
                        off = offset[i].pop(0)
                        start = midi.tick_to_time(int(on[0] * 480 * 4)) if time_granularity else on[0]
                        end = midi.tick_to_time(int(off * 480 * 4)) if time_granularity else off
                        note = pretty_midi.Note(velocity=on[1], pitch=i, start=start, end=end)
                        instrument.notes.append(note)
            midi.instruments.append(instrument)
        return midi
