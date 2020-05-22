import numpy as np

import torch
from torch.autograd import Variable
import torch.nn.functional as F

from sklearn.metrics import average_precision_score
from sklearn.metrics import accuracy_score

from .model import BaseModel

class PartsModel(BaseModel):
    def __init__(self, *args, dur_map=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.dur_map = dur_map

        self.register_statistic('apn_ts',True,'{:<8.2f}')
        self.register_statistic('apn_tr',True,'{:<8.2f}')
        self.register_statistic('acct_ts',True,'{:<8.2f}')
        self.register_statistic('acct_tr',True,'{:<8.2f}')
        self.register_statistic('ll_test',True,'{:<8.2f}')
        self.register_statistic('ll_tr',True,'{:<8.2f}')
        self.register_statistic('lln_ts',True,'{:<8.2f}')
        self.register_statistic('lln_tr',False,'{:<8.2f}')
        self.register_statistic('llt_ts',True,'{:<8.2f}')
        self.register_statistic('llt_tr',False,'{:<8.2f}')

    def loss(self, yhat, z):
        # overall loss is sum of two constituent losses: duration loss and notes loss
        return self._loss(yhat, z).sum()

    def _loss(self, yhat, z):
        # unpack the predictions
        yht,yhn = yhat
        _,_,y,yt,_ = z

        events_per_beat = 1.219 # see parts_summary.ipynb

        time_nats = torch.nn.functional.cross_entropy(yht,yt.max(1)[1])
        note_nats = self.m*torch.nn.functional.binary_cross_entropy_with_logits(yhn,y)
        nats_per_event = torch.stack([time_nats,note_nats])

        # (events/beat) x (nats/event) x (bits/nat) = bits/beat
        return events_per_beat*nats_per_event*(1/np.log(2))

    def prepare_data(self, x):
        e,t,y,yt,loc = x

        # ship everything over to the gpu
        e = Variable(e.cuda(), requires_grad=False)
        t = Variable(t.cuda(), requires_grad=False)
        y = Variable(y.cuda(), requires_grad=False)
        yt = Variable(yt.cuda(), requires_grad=False) 
        loc = Variable(loc.cuda(), requires_grad=False)

        return e,t,y,yt,loc

    def compute_stats(self, loader):
        loss = torch.zeros(2).cuda()
        batch = loader.batch_size
        note_predictions = np.empty([len(loader)*batch,self.m])
        time_predictions = np.empty(len(loader)*batch,dtype=np.int32)
        note_ground = np.empty([len(loader)*batch,self.m])
        time_ground = np.empty(len(loader)*batch,dtype=np.int32)
        for i, x in enumerate(loader):
            _,_,yn,yt,_ = x
            x = self.prepare_data(x)
            yhat = self(x)
            loss += self._loss(yhat,x).data
            yht,yhn = yhat
            time_predictions[i*batch:(i+1)*batch] = np.argmax(yht.data.cpu().numpy(),axis=1)
            note_predictions[i*batch:(i+1)*batch] = yhn.data.cpu().numpy()
            time_ground[i*batch:(i+1)*batch] = np.argmax(yt.numpy(),axis=1)
            note_ground[i*batch:(i+1)*batch] = yn.numpy()
        loss /= len(loader)

        avp = average_precision_score(note_ground.ravel(),note_predictions.ravel(),average=None)
        acc = accuracy_score(time_ground,time_predictions)
        return loss, avp, acc

    def update_status(self, train_loader, test_loader, last_time, update_time):
        (llt, lln), avp, acc = self.compute_stats(test_loader)
        self._tmp_stats['ll_test'] = llt + lln
        self._tmp_stats['llt_ts'] = llt
        self._tmp_stats['lln_ts'] = lln
        self._tmp_stats['apn_ts'] = 100*avp
        self._tmp_stats['acct_ts'] = 100*acc

        (llt, lln), avp, acc = self.compute_stats(train_loader)
        self._tmp_stats['ll_tr'] = llt + lln
        self._tmp_stats['llt_tr'] = llt
        self._tmp_stats['lln_tr'] = lln
        self._tmp_stats['apn_tr'] = 100*avp
        self._tmp_stats['acct_tr'] = 100*acc

        super().update_status(train_loader, test_loader, last_time, update_time)

    def sample(self, parts=100, notes=50, batch_size=100, raster=True, raster_resolution=48, init_data=None):
        events = []
        times = []
        for i in range(parts//batch_size + 1):
            part_count = min(batch_size,parts - i*batch_size)
            if part_count == 0: break
            e, t = self._sample_batch(part_count,notes,raster,raster_resolution,init_data)
            events.append(e)
            times.append(t)

        events = np.concatenate(events)
        times = np.concatenate(times)
        if not raster: return events,times
        
        # rasterize
        part_list = []
        for part in range(parts):
            dts = [int(dt) for dt in raster_resolution*self.dur_map[np.argmax(times[part],axis=1)]]
            x = np.zeros([np.sum(dts),128,2])
            k = 0
            for i in range(notes):
                x[k,:,1] = events[part,i]
                x[k:k+dts[i],:,0] = events[part,i]
                k += dts[i]
                
            part_list.append(x[:,None,:])

        return part_list

    def _sample_batch(self, parts, notes, raster, raster_resolution, init_data):
        events = Variable(torch.zeros([parts,self.context+notes,self.m]).cuda(),requires_grad=False)
        times = Variable(torch.zeros([parts,self.context+notes,self.maxdur]).cuda(),requires_grad=False)
        locs = Variable(torch.zeros([parts,raster_resolution]).cuda(),requires_grad=False)

        if init_data == None: # default init
            for k in range(0,self.context): times[:,k,0] = 1
        else: # init from a supplied sequence
            init_ids = np.random.choice(list(init_data.data.keys()),parts,replace=True)
            for part in range(parts):
                e,t,_,_,loc = init_data.access(init_ids[part],self.context)
                events[part,:self.context],times[part,:self.context],locs[part] = torch.Tensor(e).cuda(),torch.Tensor(t).cuda(),loc

        T = np.zeros(parts)
        for k in range(self.context,self.context+notes):
            locs.fill_(0)
            for part in range(parts):
                locs[part,round(raster_resolution*T[part]) % raster_resolution] = 1

            p = F.softmax(self.predict_rhythm(events[:,k-self.context:k].contiguous(),
                                              times[:,k-self.context:k].contiguous(),
                                              locs),dim=1)
            dts = torch.multinomial(p).data[:,0]
            for part in range(parts):
                times[part,k,dts[part]] = 1
                T[part] += self.dur_map[dts[part]]

            for note in range(self.m):
                p = F.sigmoid(self.predict_notes(events[:,k-self.context:k].contiguous(),
                                                 times[:,k-self.context:k].contiguous(),
                                                 events[:,k].contiguous(),
                                                 times[:,k].contiguous(),
                                                 locs))[:,note]
                events[:,k,note] = torch.bernoulli(p).data
       
        events = np.pad(events[:,self.context:].data.cpu().numpy(),[(0,0),(0,0),(self.offset,128-(self.offset+self.m))],'constant')
        times = times[:,self.context:].data.cpu().numpy()

        return events,times
