import numpy as np
import matplotlib.pyplot as plt

p1999 = np.array([
    [
        612,
        53
    ],
    [
        612,
        67
    ],
    [
        612,
        69
    ],
    [
        612.25,
        55
    ],
    [
        612.5,
        57
    ],
    [
        612.75,
        59
    ],
    [
        613,
        60
    ],
    [
        613.25,
        62
    ],
    [
        613.5,
        64
    ],
    [
        613.75,
        62
    ],
    [
        614,
        50
    ],
    [
        614,
        63
    ],
    [
        614,
        68
    ],
    [
        614,
        73
    ],
    [
        614.25,
        52
    ],
    [
        614.5,
        54
    ],
    [
        614.75,
        56
    ],
    [
        615,
        57
    ],
    [
        615.25,
        59
    ],
    [
        615.5,
        61
    ],
    [
        615.75,
        63
    ],
    [
        616,
        60
    ],
    [
        616,
        62
    ],
    [
        616,
        67
    ],
    [
        616,
        69
    ],
    [
        616,
        74
    ],
    [
        617,
        46
    ],
    [
        617,
        53
    ],
    [
        617,
        57
    ],
    [
        617,
        62
    ],
    [
        617,
        64
    ],
    [
        617,
        67
    ],
    [
        617,
        69
    ],
    [
        618,
        46
    ],
    [
        618,
        53
    ],
    [
        618,
        57
    ],
    [
        618,
        62
    ],
    [
        618,
        64
    ],
    [
        618,
        67
    ],
    [
        618,
        69
    ]
])
p1999[:, 0] -= p1999[0, 0]

p2322 = np.array([
    [
        468,
        54
    ],
    [
        468,
        61
    ],
    [
        468,
        63
    ],
    [
        468,
        68
    ],
    [
        468.25,
        70
    ],
    [
        468.5,
        54
    ],
    [
        468.5,
        72
    ],
    [
        468.75,
        70
    ],
    [
        469,
        57
    ],
    [
        469,
        61
    ],
    [
        469,
        66
    ],
    [
        469,
        69
    ],
    [
        469.25,
        71
    ],
    [
        469.5,
        57
    ],
    [
        469.5,
        59
    ],
    [
        469.5,
        64
    ],
    [
        469.5,
        73
    ],
    [
        469.75,
        62
    ],
    [
        469.75,
        71
    ],
    [
        470,
        58
    ],
    [
        470,
        61
    ],
    [
        470,
        70
    ],
    [
        470.25,
        72
    ],
    [
        470.5,
        58
    ],
    [
        470.5,
        70
    ],
    [
        470.75,
        68
    ],
    [
        471,
        51
    ],
    [
        471,
        62
    ],
    [
        471,
        64
    ],
    [
        471,
        69
    ],
    [
        471.25,
        71
    ],
    [
        471.5,
        51
    ],
    [
        471.5,
        69
    ],
    [
        471.75,
        67
    ],
    [
        472,
        54
    ],
    [
        472,
        61
    ],
    [
        472,
        63
    ],
    [
        472,
        68
    ],
    [
        473,
        54
    ],
    [
        473,
        58
    ],
    [
        473,
        61
    ],
    [
        473,
        63
    ],
    [
        473,
        65
    ],
    [
        473,
        68
    ],
    [
        473,
        70
    ],
    [
        474,
        54
    ],
    [
        474,
        58
    ],
    [
        474,
        61
    ],
    [
        474,
        63
    ],
    [
        474,
        65
    ],
    [
        474,
        68
    ],
    [
        474,
        70
    ]
])
p2322[:, 0] -= p2322[0, 0]

t_min = 0.1
t_max = 4
p_min = 1
p_max = 12

lookup = {}
pts = p1999
pts = sorted([list(x) for x in set(tuple(x) for x in pts)], key=lambda x: x[0])
p1999_index = []
for i in range(len(pts) - 2):
    v0 = pts[i]
    j = i + 1
    while j < len(pts) - 1:
        v1 = pts[j]
        td1 = v1[0] - v0[0]
        pd1 = v1[1] - v0[1]
        apd1 = abs(pd1)
        if t_min < td1 < t_max and p_min <= apd1 <= p_max:
            k = j + 1
            while k < len(pts):
                v2 = pts[k]
                td2 = v2[0] - v1[0]
                pd2 = v2[1] - v1[1]
                apd2 = abs(pd2)
                if t_min < td2 < t_max and p_min <= apd2 <= p_max:
                    if pd1 < 0:
                        s1 = f'-{int(apd1)}'
                    else:
                        s1 = f'+{int(apd1)}'
                    if pd2 < 0:
                        s2 = f'-{int(apd2)}'
                    else:
                        s2 = f'+{int(apd2)}'
                    if td1 >= td2:
                        tdr = round((td1 / td2) * 10) / 10
                        s3 = f'+{tdr:.1f}'
                    else:
                        tdr = round((td2 / td1) * 10) / 10
                        s3 = f'-{tdr:.1f}'
                    key = f'{s1}{s2}{s3}'
                    p1999_index.append([key, (v0, v1, v2)])
                    if key in lookup.keys():
                        lookup[key].append((v0, v1, v2))
                    else:
                        lookup[key] = [(v0, v1, v2)]
                k += 1
        j += 1

print(lookup)

nh = 0
index = []
results = []
pts = p2322
pts = sorted([list(x) for x in set(tuple(x) for x in pts)], key=lambda x: x[0])
p2322_index = []
for i in range(len(pts) - 2):
    v0 = pts[i]
    j = i + 1
    while j < len(pts) - 1:
        v1 = pts[j]
        td1 = v1[0] - v0[0]
        pd1 = v1[1] - v0[1]
        apd1 = abs(pd1)
        if t_min < td1 < t_max and p_min <= apd1 <= p_max:
            k = j + 1
            while k < len(pts):
                v2 = pts[k]
                td2 = v2[0] - v1[0]
                pd2 = v2[1] - v1[1]
                apd2 = abs(pd2)
                if t_min < td2 < t_max and p_min <= apd2 <= p_max:
                    nh += 1
                    if pd1 < 0:
                        s1 = f'-{int(apd1)}'
                    else:
                        s1 = f'+{int(apd1)}'
                    if pd2 < 0:
                        s2 = f'-{int(apd2)}'
                    else:
                        s2 = f'+{int(apd2)}'
                    if td1 >= td2:
                        tdr = round((td1 / td2) * 10) / 10
                        s3 = f'+{tdr:.1f}'
                    else:
                        tdr = round((td2 / td1) * 10) / 10
                        s3 = f'-{tdr:.1f}'
                    key = f'{s1}{s2}{s3}'
                    p2322_index.append([key, (v0, v1, v2)])
                    if key in lookup.keys():
                        results += [[v0[0], p[0][0]] for p in lookup[key]]
                        for p in lookup[key]:
                            if v0[0] == p[0][0]:
                                index.append([key, (v0, v1, v2), p])

                k += 1
        j += 1

print(len(pts))
print(pts)
results = np.array(results)
print(results)
print(f'{np.count_nonzero(np.abs(results[:, 0] - results[:, 1]) == 0)} / {nh}')
print(index)
