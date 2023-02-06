import os
import errno
import pandas as pd

def mkdir(filename):
    if not os.path.exists(os.path.dirname(filename)):
        try:
            os.makedirs(os.path.dirname(filename))
        except OSError as exc:  # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise


def get_maestro_midi_list(index_path, split):
    maestro_json = pd.read_json(index_path)
    midi_list = maestro_json.loc[maestro_json['split'] == split]['midi_filename'].values.tolist()
    return midi_list
