import math
from easydict import EasyDict as edict
import numpy as np
import pretty_midi
import itertools


def identity(x):
    return x


def validate_pitch(pitch):
    return 0 <= pitch <= 127


class MIDI:
    def __init__(
        self,
        time_quantization,
        time_unit
    ):
        if time_unit == "time":
            self.time_base_func = self.tick2time
            self.time_revert_func = identity
        elif time_unit == "crotchet":
            self.time_base_func = self.tick2crotchet
            self.time_revert_func = self.crotchet2time
        else:
            raise KeyError("Invalid time unit.")
        self.time_quantization = time_quantization

        self.source = pretty_midi.PrettyMIDI(resolution=480, initial_tempo=120)

    @property
    def vocab(self):
        return edict(
            {
                "pad": 0,
                "on": list(range(1, 129)),
                "off": list(range(129, 257)),
                "time": list(range(257, 257 + len(self.time_quantization))),
                "vel": list(range(257 + len(self.time_quantization), 257 + len(self.time_quantization) + 32))
            }
        )

    def time2tick(self, time):
        return self.source.time_to_tick(time)

    def tick2time(self, tick):
        return self.source.tick_to_time(int(tick))

    def crotchet2tick(self, crotchet):
        return crotchet * self.source.resolution

    def tick2crotchet(self, tick):
        return tick / self.source.resolution

    def crotchet2time(self, crotchet):
        return self.tick2time(self.crotchet2tick(crotchet))

    def time2crotchet(self, time):
        return self.tick2crotchet(self.time2tick(time))

    def note2events(self, note):
        on = edict(
            {
                "action": "on",
                "pitch": note.pitch,
                "time": self.time2tick(note.start),
                "velocity": note.velocity
            }
        )
        off = edict(
            {
                "action": "off",
                "pitch": note.pitch,
                "time": self.time2tick(note.end),
                "velocity": note.velocity
            }
        )
        return on, off

    def get_time_tokens(self, ticks):
        output = []
        quotient, remainder = divmod(
            self.time_base_func(ticks),
            self.time_quantization[-1]
        )
        assert quotient.is_integer()
        for _ in range(int(quotient)):
            output.append(self.vocab.time[-1])
        if remainder:
            time_quantization = [0] + self.time_quantization
            closest = min(
                range(len(time_quantization)),
                key=lambda i: abs(time_quantization[i] - remainder)
            )
            if closest != 0:
                output.append(closest - 1 + self.vocab.time[0])
        return output

    def process(self, filename, ignore_velocity, merge_tracks, pitch_shift=0, time_scale=1):
        self.source = pretty_midi.PrettyMIDI(filename)
        # ticks per semibreve
        tracks = self.source.instruments
        num_tracks = len(tracks)
        if merge_tracks:
            tracks_of_events = [
                sorted(
                    [
                        event
                        for track in tracks
                        for note in track.notes
                        for event in self.note2events(note)
                    ],
                    key=lambda x: (x.time, x.pitch)
                )
            ]
        else:
            tracks_of_events = [
                sorted(
                    [
                        event
                        for note in track.notes
                        for event in self.note2events(note)
                    ],
                    key=lambda x: (x.time, x.pitch)
                )
                for track in tracks
            ]

        sequence = [[] for _ in range(len(tracks_of_events))]
        keyboard = [[False for _ in range(128)]
                    for _ in range(num_tracks)]
        for i, track in enumerate(tracks_of_events):
            cur_vel = -1
            cur_time = 0
            for event in track:
                event.pitch += pitch_shift
                event.time *= time_scale
                if not validate_pitch(event.pitch):
                    return None
                if event.action == "on" and not keyboard[i][event.pitch]:
                    keyboard[i][event.pitch] = True
                    time_tokens = self.get_time_tokens(event.time - cur_time)
                    sequence[i] += time_tokens
                    if not ignore_velocity:
                        vel_token = math.floor(event.velocity / 4)
                        if vel_token != cur_vel:
                            sequence[i].append(self.vocab.vel[0] + vel_token)
                            cur_vel = vel_token
                    sequence[i].append(self.vocab.on[0] + event.pitch)
                    cur_time = event.time
                if event.action == "off" and keyboard[i][event.pitch]:
                    keyboard[i][event.pitch] = False
                    time_tokens = self.get_time_tokens(event.time - cur_time)
                    sequence[i] += time_tokens
                    sequence[i].append(self.vocab.off[0] + event.pitch)
                    cur_time = event.time

        sequence = list(
            zip(*itertools.zip_longest(*sequence, fillvalue=self.vocab.pad))
        )
        return np.array(sequence, dtype=int)

    def to_midi(self, sequence, resolution=480, tempo=120):
        self.source = pretty_midi.PrettyMIDI(resolution=resolution, initial_tempo=tempo)
        for track in sequence:
            instrument = pretty_midi.Instrument(program=0)
            onset = {n: [] for n in range(128)}
            offset = {n: [] for n in range(128)}
            time = 0
            velocity = 64
            for token in track:
                if token in self.vocab.on:
                    onset[token - self.vocab.on[0]].append((time, velocity))
                elif token in self.vocab.off:
                    offset[token - self.vocab.off[0]].append(time)
                elif token in self.vocab.time:
                    time += self.time_revert_func(
                        self.time_quantization[token - self.vocab.time[0]]
                    )
                elif token in self.vocab.vel:
                    velocity = (token - self.vocab.vel[0]) * 4

            for i in range(128):
                while len(onset[i]) > 0 and len(offset[i]) > 0:
                    on_time, on_vel = onset[i].pop(0)
                    # Remove all off events that happan before the on event
                    while len(offset[i]) > 0 and offset[i][0] - on_time <= 0:
                        _ = offset[i].pop(0)
                    if len(offset[i]) == 0:
                        break
                    off_time = offset[i].pop(0)
                    note = pretty_midi.Note(
                        velocity=on_vel,
                        pitch=i,
                        start=on_time,
                        end=off_time
                    )
                    instrument.notes.append(note)
            self.source.instruments.append(instrument)
        return self.source
