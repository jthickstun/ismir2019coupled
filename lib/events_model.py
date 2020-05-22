import numpy as np

import torch
from torch.autograd import Variable
import torch.nn.functional as F

from sklearn.metrics import average_precision_score
from sklearn.metrics import accuracy_score

from .model import BaseModel
from .config import corpora

class EventsModel(BaseModel):
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
        _,_,y,yt,_,_,_ = z

        events_per_beat = 4.775 # see voices_summary.ipynb

        time_nats = torch.nn.functional.cross_entropy(yht,yt[:,0,:self.maxdur].max(1)[1])
        note_nats = self.m*torch.nn.functional.binary_cross_entropy_with_logits(yhn,y[:,0])
        nats_per_event = torch.cat([time_nats,note_nats])

        # (events/beat) x (nats/event) x (bits/nat) = bits/beat
        return events_per_beat*nats_per_event*(1/np.log(2))

    def prepare_data(self, x):
        e,t,y,yt,flow,loc,corpus,_ = x

        # ship everything over to the gpu
        e = Variable(e.cuda(), requires_grad=False)
        t = Variable(t.cuda(), requires_grad=False)
        y = Variable(y.cuda(), requires_grad=False)
        yt = Variable(yt.cuda(), requires_grad=False) 
        flow = Variable(flow.cuda(), requires_grad=False) 
        loc = Variable(loc.cuda(), requires_grad=False) 
        corpus = Variable(corpus.cuda(), requires_grad=False) 

        return e,t,y,yt,flow,loc,corpus

    def compute_stats(self, loader):
        loss = 0
        batch = loader.batch_size
        note_predictions = np.empty([len(loader)*batch,self.m])
        time_predictions = np.empty(len(loader)*batch,dtype=np.int32)
        note_ground = np.empty([len(loader)*batch,self.m])
        time_ground = np.empty(len(loader)*batch,dtype=np.int32)
        for i, x in enumerate(loader):
            _,_,yn,yt,_,_,_,_ = x
            x = self.prepare_data(x)
            yhat = self(x)
            loss += self._loss(yhat,x).data
            yht,yhn = yhat
            time_predictions[i*batch:(i+1)*batch] = np.argmax(yht.data.cpu().numpy(),axis=1)
            note_predictions[i*batch:(i+1)*batch] = yhn.data.cpu().numpy()
            time_ground[i*batch:(i+1)*batch] = np.argmax(yt[:,0,:self.maxdur].numpy(),axis=1)
            note_ground[i*batch:(i+1)*batch] = yn[:,0].numpy()
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

    def sample(self, scores=100, num_events=50, parts=4, corpus=None):
        self.NULL = self.maxdur
        self.EMPTY = self.maxdur+1

        if corpus == None:
            corpus = np.random.randint(0,len(corpora),scores)

        events = torch.zeros([scores,self.context+num_events,self.parts,self.m]).cuda()
        times = torch.zeros([scores,self.context+num_events,self.parts,self.dur_features]).cuda()
        corps = torch.zeros([scores,len(corpora)]).cuda()

        for part in range(self.parts):
            if part < parts:
                for k in range(0,self.context): times[:,k,:,0] = 1
            else:
                times[:,:,part,self.EMPTY] = 1

        for score in range(scores):
            corps[score,corpus[score]] = 1

        e = Variable(torch.zeros([scores,self.context+1,self.parts,self.m]).cuda(),requires_grad=False)
        t = Variable(torch.zeros([scores,self.context+1,self.parts,self.dur_features]).cuda(),requires_grad=False)
        loc = Variable(torch.zeros([scores,48]).cuda(),requires_grad=False)
        flow = Variable(torch.zeros([scores,self.context,6,6]).cuda(),requires_grad=False)
        c = Variable(torch.zeros([scores,len(corpora)]).cuda(),requires_grad=False)

        flow[:,:] = torch.eye(6)

        frontier = np.zeros([scores,parts])
        for k in range(self.context,self.context+num_events):
            print('.', end='')
            rear = np.min(frontier,axis=1)
            for score in range(scores):
                for part in range(parts):
                    if frontier[score,part] > rear[score]:
                        events[score,k,part] = events[score,k-1,part]
                        times[score,k,part,self.NULL] = 1

            for part in range(parts):
                loc.fill_(0)
                updates = torch.LongTensor([score for score in range(scores) if frontier[score,part] == rear[score]]).cuda()
                e[:len(updates)] = events[updates][:,k-self.context:k+1]
                e[:len(updates),:,0] = events[updates][:,k-self.context:k+1,part]
                e[:len(updates),:,part] = events[updates][:,k-self.context:k+1,0]
                t[:len(updates)] = times[updates][:,k-self.context:k+1]
                t[:len(updates),:,0] = times[updates][:,k-self.context:k+1,part]
                t[:len(updates),:,part] = times[updates][:,k-self.context:k+1,0]
                c[:len(updates)] = corps[updates]
                for i,score in enumerate(updates):
                    loc[i,48*frontier[score,part]%48] = 1

                p = F.softmax(self.predict_rhythm(e[:len(updates),:self.context].contiguous(),
                                                  t[:len(updates),:self.context].contiguous(),
                                                  e[:len(updates),-1].contiguous(),
                                                  t[:len(updates),-1].contiguous(),
                                                  loc[:len(updates)],
                                                  c[:len(updates)]),dim=1)
                dts = torch.multinomial(p).data[:,0]
                for i,score in enumerate(updates):
                    t[i,-1,0,dts[i]] = 1
                    frontier[score,part] += self.dur_map[dts[i]]

                for note in range(self.m):
                    p = F.sigmoid(self.predict_notes(e[:len(updates),:self.context].contiguous(),
                                                     t[:len(updates),:self.context].contiguous(),
                                                     e[:len(updates),-1].contiguous(),
                                                     t[:len(updates),-1].contiguous(),
                                                     loc[:len(updates)],
                                                     c[:len(updates)]))[:,note]
                    e[:len(updates),-1,0,note] = torch.bernoulli(p).data

                for i,score in enumerate(updates):
                    events[score,k,part] = e[i,-1,0].data
                    times[score,k,part] = t[i,-1,0].data

        events = events[:,self.context:].cpu().numpy()
        times = times[:,self.context:].cpu().numpy()
        corps = np.argmax(corps.cpu().numpy(),axis=1)
        return events,times,corps
