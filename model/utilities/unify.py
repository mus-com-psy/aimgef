import numpy as np


def correlation(x):
    assert x.shape[0] == 12
    keys = ["C major", "Db major", "D major", "Eb major", "E major", "F major",
            "Gb major", "G major", "Ab major", "A major", "Bb major", "B major",
            "C minor", "C# minor", "D minor", "Eb minor", "E minor", "F minor",
            "F# minor", "G minor", "G# minor", "A minor", "Bb minor", "B minor"]
    major = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
    minor = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
    max_r = 0
    max_key = ''
    key_index = 0
    output = {}
    for i in range(12):
        r = np.corrcoef(x, np.roll(major, i))[0, 1]
        output[keys[i]] = r
        if max_r <= r:
            max_r = r
            max_key = keys[i]
            key_index = i

    for i in range(12):
        r = np.corrcoef(x, np.roll(minor, i))[0, 1]
        output[keys[i + 12]] = r
        if max_r <= r:
            max_r = r
            max_key = keys[i + 12]
            key_index = i + 12

    return max_r, max_key, key_index, output


def count_key(x):
    tmp = np.remainder(x, 12)
    output = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    for i in tmp:
        output[i] += 1
    return output


def unify(x):
    _, max_key, key_index, _ = correlation(count_key(x))
    to_minus = 0
    if 0 <= key_index < 12:
        to_minus = key_index
    elif 12 <= key_index < 24:
        to_minus = key_index - 3

    pitch_sum = x.sum()
    minus = 0
    for i in range(11):
        abs_sum_value = np.abs(np.sum(x - i * 12 - to_minus))
        if abs_sum_value <= pitch_sum:
            pitch_sum = abs_sum_value
            minus = i * 12 + to_minus
    if 'minor' in max_key:
        return x - minus + 60
    elif 'major' in max_key:
        return x - minus + 69
    else:
        raise ValueError('Invalid key signature.')
