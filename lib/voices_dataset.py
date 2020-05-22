import sys,os,pickle
from collections import defaultdict
from enum import Enum

import numpy as np
from torch.utils.data import Dataset

from . import config
from .corpus import checkout,find_scores,parse_raw

DatasetSplit = Enum('DatasetSplit', 'total train test')

class VoicesDataset(Dataset):
    """KernScores Voices Dataset.
    """

    data_dir = 'data'
    data_file = 'voices_data.npz'

    NULL = 0
    EMPTY = 1
    START = 2

    def __init__(self, split=DatasetSplit.total, context=10, corpora=tuple(config.corpora.keys()), pitch_shift=False, shuffle=False, instr=False):
        self.context = context
        self.corpora = corpora
        self.shuffle = shuffle
        self.instrument_labels = instr

        # data augmentation
        self.pitch_shift = pitch_shift

        self.data = self.create_dataset()

        if split == DatasetSplit.train:
            self.data = { k : v for k,v in self.data.items() if k.split(':')[0] not in config.test_ids }
        elif split == DatasetSplit.test:
            self.data = { k : v for k,v in self.data.items() if k.split(':')[0] in config.test_ids }

        self.data = { k : v for k,v in self.data.items() if k.startswith(self.corpora) }

        self.precompute()

    def precompute(self):
        self.size = 0
        self.base_idx = dict()
        self.cumsize = dict()
        for score_id,(_,_,_,index) in sorted(self.data.items()):
            for v in range(6):
                if len(index[v]) == 0: continue
                self.base_idx[(score_id,v)] = self.size
                self.cumsize[self.size] = (score_id,v)
                self.size += len(index[v])
        self.sorted_base = sorted(self.cumsize.keys())

        beats = 0
        events = 0
        for score_id,(e,t,f,index) in self.data.items():
            beats += sum(self.dur_map[t[:,0].astype(np.int32)])
            events += sum([len(index[v]) for v in range(6)])
        self.events_per_beat = float(events/beats)

    def access(self, score_id, voice, i):
        events,durs,flow,index = self.data[score_id]
        j = index[voice][i][0] # index in score of i'th item in the specified voice
        pos = index[voice][i][1] # position of i'th item in the specified voice

        ps = 0
        if self.pitch_shift:
            ps = np.random.randint(-6,5)

        e = np.zeros([self.context,self.max_parts,self.m],dtype=np.float32)
        t = np.full([self.context,6],self.START,dtype=np.int32)
        f = np.repeat(np.eye(6,6,dtype=np.float32)[None,:,:],self.context,axis=0)
        if self.context > j: # need to do some temporal padding
            if ps == 0: e[self.context-j:] = events[0:j].astype(np.float32)
            elif ps > 0: e[self.context-j:,:,ps:] = events[0:j,:,:-ps].astype(np.float32)
            else: e[self.context-j:,:,:ps] = events[0:j,:,-ps:].astype(np.float32)
            t[self.context-j:] = durs[0:j]
            f[self.context-j:] = flow[0:j].astype(np.float32) 
        else:
            if ps == 0: e[:] = events[j-self.context:j].astype(np.float32)
            elif ps > 0: e[:,:,ps:] = events[j-self.context:j,:,:-ps].astype(np.float32)
            else: e[:,:,:ps] = events[j-self.context:j,:,-ps:].astype(np.float32)
            t[:] = durs[j-self.context:j].astype(np.int32)
            f[:] = flow[j-self.context:j].astype(np.float32)
        
        t_out = np.zeros([self.context,self.max_parts,self.maxdur+3],dtype=np.float32)
        for v in range(self.max_parts):
            for k in range(self.context): t_out[k, v, t[k,v]] = 1
        t = t_out

        y = np.zeros([self.max_parts,self.m],dtype=np.float32)
        if ps == 0: y[:voice+1] = events[j,:voice+1].astype(np.float32)
        elif ps > 0: y[:voice+1,ps:] = events[j,:voice+1,:-ps].astype(np.float32)
        else: y[:voice+1,:ps] = events[j,:voice+1,-ps:].astype(np.float32)

        yt = np.zeros([self.max_parts,self.maxdur+3],dtype=np.float32)
        for v in range(6):
            if v <= voice: # if we're conditioning on this part
                yt[v, int(durs[j,v])] = 1
            else: # just fill in the deterministic bits
                if durs[j,v] == self.NULL:
                    yt[v,self.NULL] = 1
                    y[v] = events[j,v]
                elif durs[j,v] == self.EMPTY:
                    yt[v,self.EMPTY] = 1
        yf = flow[j].copy().astype(np.float32)

        loc = np.zeros(48,dtype=np.float32)
        loc[pos % 48] = 1

        corpus = np.zeros(len(self.corpora),dtype=np.float32)
        for i,c in enumerate(sorted(self.corpora)):
            if score_id.startswith(c):
                corpus[i] = 1
                break
        else: raise KeyError

        if self.instrument_labels:
            instr = np.zeros(3,dtype=np.float32)
            if score_id.startswith(tuple(config.piano_corpora)): instr[0] = 1
            elif score_id.startswith(tuple(config.string_corpora)): instr[1] = 1
            else: instr[2] = 1

        pos = np.sum(self.dur_map[durs[:j].astype(np.int32)],axis=0)

        # voice we're predicting goes first
        e[:,[0,voice]] = e[:,[voice,0]]
        t[:,[0,voice]] = t[:,[voice,0]]
        y[[0,voice]] = y[[voice,0]]
        yt[[0,voice]] = yt[[voice,0]]
        pos[[0,voice]] = pos[[voice,0]]

        # apply change of basis to the flows (swap in both indices; PUP^)
        f[:,[0,voice]] = f[:,[voice,0]]; f[:,:,[0,voice]] = f[:,:,[voice,0]]
        yf[[0,voice]] = yf[[voice,0]]; yf[:,[0,voice]] = yf[:,[voice,0]]

        if self.shuffle:
            o = range(self.max_parts) # original ordering
            p = np.random.permutation(range(1,self.max_parts))
            p = np.insert(p,0,0) # first part is fixed
            e[:,o] = e[:,p]
            t[:,o] = t[:,p]
            y[o] = y[p]
            yt[o] = yt[p]
            pos[o] = pos[p]
            f[:,o] = f[:,p]; f[:,:,o] = f[:,:,p]
            yf[o] = yf[p]; yf[:,o] = yf[:,p]

        if self.instrument_labels: return e,t,f,y,yt,yf,loc,instr,pos
        else: return e,t,f,y,yt,yf,loc,corpus,pos

    def location_to_index(self, score_id, voice, i):
        return self.base_idx[(score_id,voice)] + i

    def index_to_location(self, index):
        base = self.sorted_base[np.searchsorted(self.sorted_base,index,'right')-1]
        score_id,voice = self.cumsize[base]
        i = index - base
        return score_id,voice,i

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        return self.access(*self.index_to_location(index))

    def to_raster(self, e, t, f, y, yt, yf, loc, corpus, pos):
        e = np.pad(e,[(0,0),(0,0),(self.offset,128-(self.offset+self.m))],'constant')
        t = (48*self.dur_map[np.argmax(t,axis=2)]).astype(np.int32)
        y = np.pad(y[:,:,None],[(0,0),(self.offset,128-(self.offset+self.m)),(0,1)],'constant')
        yt = (48*self.dur_map[np.argmax(yt,axis=1)]).astype(np.int32)

        x = np.zeros([np.max(np.sum(t,axis=0)),e.shape[1],e.shape[2],2])
        offset = (48*(max(pos)-pos)).astype(np.int32)
        loc = x.shape[0] - offset
        for i in reversed(range(e.shape[0])):
            for p in range(e.shape[1]):
                if t[i,p] == 0: continue
                if loc[p]-t[i,p] >= 0: x[loc[p]-t[i,p],p,:,1] = e[i,p]
                x[loc[p]-t[i,p]:loc[p],p,:,0] = e[i,p]
                loc[p] -= t[i,p]
        return x,y,yt

    def decode_duration(self,t):
        if np.sum(t) > 1: return 'INVALID'
        if np.sum(t) == 0: return 'O' # masked
        idx = np.argmax(t)
        if idx > 2: out = str(round(self.dur_map[idx],2))
        elif idx == 2: out = '-' # start
        elif idx == 1: out = 'x' # empty
        elif idx == 0: out = '*' # null
        return out

    def decode_notes(self,e):
        out = ''
        for n in range(len(e)):
            if e[n] == 1: out += str(self.offset+n) + ' '
        return out

    def decode_flow(self,f):
        if not np.any(f): return 'NOFLOW'

        out = ''
        for i in range(f.shape[0]):
            for o in range(f.shape[1]):
                if i == o and f[i,o] == 1: # suppress the diagonal
                    continue
            
                if f[i,o] != 0: out += '{}->{},'.format(i,o)
        return out

    def decode_event(self,e,t,f):
        out = ''
        for p in range(len(e)):
            out += self.decode_duration(t[p]) + ' : ' + self.decode_notes(e[p]) + '\t'
    
        out += self.decode_flow(f)
        out += '\n'   
        return out

    def data_to_str(self,e,t,f,y,yt,yf,loc,corpus,pos):
        out = ''
        for j in range(e.shape[0]):
            out += self.decode_event(e[j],t[j],f[j])

        out += '---\n'
        out += self.decode_event(y,yt,yf)
        return out.expandtabs(16)

    def create_dataset(self):
        data_path = os.path.join(self.data_dir,self.data_file)
        if os.path.isfile(data_path):
            data = dict(np.load(data_path, allow_pickle=True))
            self.dur_map = data.pop('_dur_map')
            self.maxdur = len(self.dur_map)-3
            self.offset = int(data.pop('_min_note'))
            self.m = int(data.pop('_note_range'))
            self.max_parts = int(data.pop('_max_parts'))
            print('Found cached voices datafile at {}'.format(data_path,len(data)))
            return data

        scores_path = os.path.join(self.data_dir,'kernscores')
        checkout(scores_path)
        parsed_scores,min_note,max_note,pickups = parse_raw(find_scores(scores_path),event_rep=True)

        scores_data = dict()
        dur_map = dict()
        for score_id,score in parsed_scores.items():
            pickup = pickups[score_id] % 1

        scores_data = dict()
        dur_map = dict()
        dur_map[0] = self.NULL
        dur_map[-4] = self.EMPTY
        dur_map[-8] = self.START
        m = max_note-min_note+1

        for score_id,score in parsed_scores.items():
            pickup = pickups[score_id] % 1
            events = []
            durs = []
            flow = []
            voice_index = [[] for v in range(6)]
            loc = -48*pickup
            for i in range(len(score)):
                events.append(score[i][0][:,min_note:min_note+m])

                step = 9999
                for v in range(6):
                    dur = 4*score[i][1][v]
                    if not (dur in dur_map): dur_map[dur] = len(dur_map)
                    if dur > 0:
                        step = min(step,dur)
                        voice_index[v].append((i,int(loc)))
                    elif dur == 0: # null
                        assert i > 0 # first event should never be null
                        events[-1][v] = events[-2][v]

                durs.append(np.vectorize(dur_map.__getitem__)(4*score[i][1]))
                flow.append(score[i][2])
        
                loc += 48*step
        
            events = np.stack(events).astype(np.int8)
            durs = np.stack(durs).astype(np.int8)
            flow = np.stack(flow).astype(np.int8)
            scores_data[score_id] = (events,durs,flow,voice_index)

        inv_dur_map = np.zeros(len(dur_map))
        for dur,idx in sorted(dur_map.items()):
            inv_dur_map[idx] = max(dur,0)

        scores_data['_dur_map'] = inv_dur_map
        scores_data['_min_note'] = min_note
        scores_data['_note_range'] = m
        scores_data['_max_parts'] = 6
         
        np.savez(data_path,**scores_data)
    
        self.dur_map = scores_data.pop('_dur_map')
        self.maxdur = len(self.dur_map)-3
        self.offset = int(scores_data.pop('_min_note'))
        self.m = int(scores_data.pop('_note_range'))
        self.max_parts = int(scores_data.pop('_max_parts'))
        return scores_data

