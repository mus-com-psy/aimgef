import pretty_midi as pm
import pickle
import json
import os

if __name__ == '__main__':
    with open("./maestro-v3.0.0/maestro-v3.0.0.json") as json_file:
        maestro = json.load(json_file)
        for i, filename in maestro["midi_filename"].items():
            midi = pm.PrettyMIDI(f'./maestro-v3.0.0/{filename}')
            assert len(midi.instruments) == 1
            notes = sorted(midi.instruments[0].notes, key=lambda x: (x.start, x.pitch))
            head = notes[0].start
            results = [[n.start - head, n.pitch] for n in notes]
            with open(f'./maestro-v3.0.0/{os.path.splitext(filename)[0]}.pkl', "wb") as f:
                pickle.dump(results, f)
            print(f'[DONE]\t{i.zfill(4)}\t{filename}')
