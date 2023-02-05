#!/usr/bin/env python3
import os


def main():
    composer = ['Beethoven', 'Haydn', 'Mozart']
    for c in composer:
        n_first = 0
        n_first_allegro = 0
        n_files = len(os.listdir('kern/' + c + '/'))

        tempo_1 = []
        tempo_0 = []
        for i in range(n_files):
            with open('kern/' + c + '/' + str(i) + '.krn', 'r') as f:
                sq = False
                first = False
                tempo = False
                for line in f.readlines():
                    if '!!!OTL: String Quartet' in line:
                        sq = True
                    if '!!!OMV' in line and '1' in line:
                        first = True
                    if '!!!OMD' in line:
                        tmp_tempo = line[8:]
                        if 'Allegro' in line or 'Allegretto' in line or 'Vivace' in line or 'Presto' in line \
                                or 'allegro' in line or 'allegretto' in line or 'vivace' in line or 'presto' in line:
                            tempo = True

                if sq and first:
                    n_first += 1
                    if tempo:
                        tempo_1.append(tmp_tempo)
                        n_first_allegro += 1
                    else:
                        tempo_0.append(tmp_tempo)
        print(c + ':\n', '   String Quartet 1st movements: ' + str(n_first), '\n',
              '        Allegro tempo: ' + str(n_first_allegro), '\n')

        print('          selected tempo:')
        for t in tempo_1:
            print('                 ' + t, end='')
        print('          unselected tempo:')
        for t in tempo_0:
            print('                 ' + t, end='')
    return


if __name__ == '__main__':
    main()
