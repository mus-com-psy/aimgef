import mido
import datetime


def pitch_midi(x):
    for sequence in x:
        current_time = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
        mid = mido.MidiFile(charset='latin1', clip=False, debug=False, ticks_per_beat=384, type=1)
        info = mido.MidiTrack()
        info.append(mido.MetaMessage(type='set_tempo', tempo=500000, time=0))
        info.append(mido.MetaMessage(type='time_signature', clocks_per_click=24, denominator=4,
                                     notated_32nd_notes_per_beat=8, numerator=4, time=0))
        info.append(mido.MetaMessage(type='end_of_track', time=1))
        track = mido.MidiTrack()
        mid.tracks.append(info)
        mid.tracks.append(track)
        track.append(mido.Message('program_change', program=0, time=0))
        for pitch in sequence:
            _, index = pitch.max(0)

            track.append(mido.Message('note_on', note=index.item(), velocity=64, time=100))
            track.append(mido.Message('note_on', note=index.item(), velocity=0, time=100))
        track.append(mido.MetaMessage(type='end_of_track', time=1))
        mid.save('log/%s.mid' % current_time)
