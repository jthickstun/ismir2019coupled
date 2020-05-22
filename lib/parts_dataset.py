import sys,os,errno,pickle
from collections import defaultdict
from enum import Enum
from subprocess import call

import numpy as np
from torch.utils.data import Dataset

from . import config
from .corpus import checkout,parse_parts

DatasetSplit = Enum('DatasetSplit', 'total train test')

class PartsDataset(Dataset):
    """KernScores Parts Dataset.
    """

    data_dir = 'data'
    data_file = 'parts_data.npz'

    def __init__(self, split=DatasetSplit.total, context=10):
        self.context = context

        self.data = self.create_dataset()

        if split == DatasetSplit.train:
            self.data = { k : v for k,v in self.data.items() if k.split(':')[0] not in config.test_ids }
        elif split == DatasetSplit.test:
            self.data = { k : v for k,v in self.data.items() if k.split(':')[0] in config.test_ids }

        self.precompute()

    def precompute(self):
        self.size = 0
        self.base_idx = dict()
        self.cumsize = dict()
        for part_id,part in sorted(self.data.items()):
            self.base_idx[part_id] = self.size
            self.cumsize[self.size] = part_id
            self.size += len(part)
        self.sorted_base = sorted(self.cumsize.keys())

    def access(self, part_id, i):
        if self.context > i: # need to do some padding
            e = np.zeros([self.context,self.m],dtype=np.float32)
            e[self.context-i:] = self.data[part_id][0:i,:self.m].astype(np.float32)
            t = np.zeros(self.context,dtype=np.int32)
            t[self.context-i:] = self.data[part_id][0:i,self.m]
        else:
            e = self.data[part_id][i-self.context:i,:self.m].astype(np.float32)
            t = self.data[part_id][i-self.context:i,self.m]

        t_out = np.zeros([self.context,self.maxdur],dtype=np.float32)
        for k in range(self.context): t_out[k, t[k]] = 1
        t = t_out

        y = self.data[part_id][i][:self.m].astype(np.float32)
        yt = np.zeros(self.maxdur,dtype=np.float32)
        yt[self.data[part_id][i][self.m]] = 1

        loc = np.zeros(48,dtype=np.float32)
        loc[self.data[part_id][i][self.m+1]] = 1

        return e,t,y,yt,loc

    def location_to_index(self, part_id, i):
        return self.base_idx[part_id] + i

    def index_to_location(self, index):
        base = self.sorted_base[np.searchsorted(self.sorted_base,index,'right')-1]
        part_id = self.cumsize[base]
        i = index - base
        return part_id,i

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        return self.access(*self.index_to_location(index))

    def to_raster(self, e, t, y, yt, loc):
        e = np.pad(e,[(0,0),(self.offset,128-(self.offset+self.m))],'constant')
        t = (48*self.dur_map[np.argmax(t,axis=1)]).astype(np.int32)
        y = np.pad(y[:,None],[(self.offset,128-(self.offset+self.m)),(0,1)],'constant')
        yt = int(48*self.dur_map[np.argmax(yt)])

        x = np.zeros([np.sum(t),e.shape[1],2])
        k = 0
        for i in range(e.shape[0]):
            x[k,:,0] = x[k,:,1] = e[i]
            for j in range(t[i]):
                x[k+j,:,0] = e[i]
            k += t[i]
        return x[:,None],y,yt

    def create_dataset(self):
        data_path = os.path.join(self.data_dir,self.data_file)
        if os.path.isfile(data_path):
            data = dict(np.load(data_path))
            self.dur_map = data.pop('_dur_map')
            self.maxdur = len(self.dur_map)
            self.offset = int(data.pop('_min_note'))
            self.m = int(data.pop('_note_range'))
            print('Found cached parts datafile at {} ({} parts)'.format(data_path,len(data)))
            return data

        scores_path = os.path.join(self.data_dir,'kernscores')
        checkout(scores_path)
        parts_path = os.path.join(self.data_dir,self.data_file)
        parts_data,self.offset,self.m,self.dur_map = parse_parts(scores_path,cache=parts_path)
        self.maxdur = len(self.dur_map)
        return parts_data

