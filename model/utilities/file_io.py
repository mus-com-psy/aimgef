import math
import matplotlib.pyplot as plt
import numpy as np
import pretty_midi
import torchaudio


class MusicBox:
    def __init__(self, midi_filename, audio_filename, time_set=None):
        if midi_filename:
            self._midi = pretty_midi.PrettyMIDI(midi_filename)
            if time_set is None:
                self._time_set = [(t / 100) for t in range(0, 101)]
            else:
                self._time_set = time_set
            self._vocab = {
                "pad": 0,
                "on": list(range(1, 129)),
                "off": list(range(129, 257)),
                "time": list(range(257, 257 + len(self._time_set) - 1)),
                "vel": list(range(257 + len(self._time_set) - 1, 257 + len(self._time_set) - 1 + 32))
            }
        if audio_filename:
            self._audio = torchaudio.load(audio_filename)

    def get_audio(self):
        return self._audio

    def clean(self, events):
        output = []
        keyboard = [False for _ in range(128)]
        for event in events:
            if event in self._vocab["on"]:
                pitch = event - self._vocab["on"][0]
                if not keyboard[pitch]:
                    keyboard[pitch] = True
                else:
                    continue
            if event in self._vocab["off"]:
                pitch = event - self._vocab["off"][0]
                if keyboard[pitch]:
                    keyboard[pitch] = False
                else:
                    continue
            output.append(event)
        while not (output[0] in self._vocab["on"] or output[0] in self._vocab["vel"]):
            output.pop(0)
        return output

    def get_sliced_events(self, length, stride, vel=True):
        output = []
        events = self.get_events(vel)
        start_idx = 0
        while start_idx < len(events):
            if len(events) - start_idx < length:
                break
            output.append(self.clean(events[start_idx:start_idx + length]))
            start_idx += stride
        return output

    def get_events(self, vel=True):
        notes = []
        for i, track in enumerate(self._midi.instruments):
            for note in track.notes:
                notes.append({
                    "track": i,
                    "act": "on",
                    "pitch": note.pitch,
                    "time": note.start,
                    "vel": note.velocity
                })
                notes.append({
                    "track": i,
                    "act": "off",
                    "pitch": note.pitch,
                    "time": note.end,
                    "vel": note.velocity
                })
        notes.sort(key=lambda x: (x["time"], x["pitch"]))
        events = []
        cur_vel = 0
        cur_time = notes[0]["time"]
        keyboard = [False for _ in range(128)]
        for note in notes:
            if note["act"] == "on" and not keyboard[note["pitch"]]:
                keyboard[note["pitch"]] = True
                time_tokens = self.get_time_tokens(note["time"] - cur_time)
                events += time_tokens
                if vel:
                    assert note["vel"] != 0, "Velocity shouldn't be 0."
                    vel_token = math.ceil(note["vel"] / 4)
                    if vel_token != cur_vel:
                        cur_vel = vel_token
                        events.append(self._vocab["vel"][0] + cur_vel - 1)
                events.append(self._vocab["on"][0] + note["pitch"])
                cur_time = note["time"]
            if note["act"] == "off" and keyboard[note["pitch"]]:
                keyboard[note["pitch"]] = False
                time_tokens = self.get_time_tokens(note["time"] - cur_time)
                events += time_tokens
                events.append(self._vocab["off"][0] + note["pitch"])
                cur_time = note["time"]
        return events

    def get_time_tokens(self, time):
        output = []
        quotient, remainder = divmod(time, self._time_set[-1])
        for _ in range(int(quotient)):
            output.append(self._vocab["time"][-1])
        if remainder:
            closest = min(range(len(self._time_set)), key=lambda i: abs(self._time_set[i] - remainder))
            if closest != 0:
                output.append(closest + self._vocab["time"][0] - 1)
        return output

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
