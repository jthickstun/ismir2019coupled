from collections import defaultdict

import numpy as np

notes = ['c', 'd', 'd', 'e', 'e', 'f', 'g', 'g', 'a', 'a', 'b', 'b']
flats = ['','-','','-','','','-','','-','','-','']

def create_value_map(dataset):
    value_map = dict()
    dot_amounts = [3./2.,7./4.,15./16.]
    for idx,dur in enumerate(dataset.dur_map):
        if dur == 0: continue
        base = 4./dur
        dots = 0
        while round(base) != round(base,2):
            base *= dot_amounts[dots]
            dots += 1
            if dots == 3:
                base = dots = 0

        if base == 0: out = 'U'
        else: out = str(int(base)) + '.'*dots
        value_map[idx] = out
    return value_map

value_map = None

def decode_duration(t):
    if np.sum(t) > 1: return 'INVALID'
    if np.sum(t) == 0: return 'O' # masked
    idx = np.argmax(t)
    if idx > 2: 
        out = value_map[idx]
    elif idx == 2: out = '-' # start
    elif idx == 1: out = 'x' # empty
    elif idx == 0: out = '.' # null
    return out

def decode_notes(e, offset=24):
    if np.sum(e) == 0: return 'r'
    
    out = []
    for n in range(len(e)):
        if e[n] == 1:
            octave,idx = divmod(offset+n-60,12)
            note = notes[idx]
            if octave < 0:
                octave -= 1
                note = note.upper()
            note *= abs(1+octave)
            note += flats[idx]
                
            out.append(note)
    return ' '.join(out)

def decode_event(e,t):
    out = []
    for p in range(6-int(np.sum(t[:,1]))):
        dur = decode_duration(t[p])
        if dur == '.': out.append(dur)
        else: out.append(dur + decode_notes(e[p]))
    out = '\t'.join(out)
    out += '\n'
    return out

def data_to_str(e,t,f,y,yt,yf,loc,corpus,pos):
    out = ['**kern']*int(6-np.sum(t[0,:,1]))
    out = '\t'.join(out)
    out += '\n'
    
    for j in range(e.shape[0]):
        out += decode_event(e[j],t[j])

    out += '\t'.join(['*-']*int(6-np.sum(t[0,:,1])))
    out += '\n'
    out += decode_event(y,yt)
    return out#.expandtabs(16)

