import numpy as np

import torch
from torch.autograd import Variable
import torch.nn.functional as F

from sklearn.metrics import average_precision_score
from sklearn.metrics import accuracy_score

from .model import BaseModel
from .config import corpora

class VoicesModel(BaseModel):
    def __init__(self, *args, dataset=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.dur_map = dataset.dur_map
        self.events_per_beat = dataset.events_per_beat

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
        _,_,_,y,yt,yf,_,_ = z

        time_nats = torch.nn.functional.cross_entropy(yht,yt[:,0,3:].max(1)[1])
        note_nats = self.m*torch.nn.functional.binary_cross_entropy_with_logits(yhn,y[:,0])
        nats_per_event = torch.stack([time_nats,note_nats])

        # (events/beat) x (nats/event) x (bits/nat) = bits/beat
        return self.events_per_beat*nats_per_event*(1/np.log(2))

    def prepare_data(self, x):
        e,t,f,y,yt,yf,loc,corpus,_ = x

        # ship everything over to the gpu
        e = Variable(e.cuda(), requires_grad=False)
        t = Variable(t.cuda(), requires_grad=False)
        f = Variable(f.cuda(), requires_grad=False) 
        y = Variable(y.cuda(), requires_grad=False)
        yt = Variable(yt.cuda(), requires_grad=False) 
        yf = Variable(yf.cuda(), requires_grad=False) 
        loc = Variable(loc.cuda(), requires_grad=False) 
        corpus = Variable(corpus.cuda(), requires_grad=False) 

        return e,t,f,y,yt,yf,loc,corpus

    def compute_stats(self, loader):
        loss = torch.zeros(2).cuda()
        batch = loader.batch_size
        note_predictions = np.empty([len(loader)*batch,self.m])
        time_predictions = np.empty(len(loader)*batch,dtype=np.int32)
        note_ground = np.empty([len(loader)*batch,self.m])
        time_ground = np.empty(len(loader)*batch,dtype=np.int32)
        for i, x in enumerate(loader):
            _,_,_,yn,yt,yf,_,_,_ = x
            x = self.prepare_data(x)
            yhat = self(x)
            loss += self._loss(yhat,x).data
            yht,yhn = yhat
            time_predictions[i*batch:(i+1)*batch] = np.argmax(yht.data.cpu().numpy(),axis=1)
            note_predictions[i*batch:(i+1)*batch] = yhn.data.cpu().numpy()
            time_ground[i*batch:(i+1)*batch] = np.argmax(yt[:,0,3:].numpy(),axis=1)
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

    def sample(self, scores=100, num_events=50, parts=4, corpus=None, debug=None):
        self.NULL = 0
        self.EMPTY = 1
        self.START = 2

        if corpus is None:
            corpus = np.random.randint(0,len(corpora),scores)

        events = torch.zeros([scores,self.context+num_events,self.parts,self.m]).cuda()
        times = torch.zeros([scores,self.context+num_events,self.parts,self.dur_features]).cuda()
        flow = torch.zeros([scores,self.context+num_events,self.parts,self.parts]).cuda()
        corps = torch.zeros([scores,int(max(corpus))+1]).cuda()

        for k in range(0,self.context):
            times[:,k,:,self.START] = 1

        for part in range(parts,self.parts):
            times[:,self.context:,part,self.EMPTY] = 1

        flow[:,:self.context+1] = torch.eye(self.parts)[None,None,:,:].repeat(scores,self.context+1,1,1)
        flow[:,self.context+1:,:parts,:parts] = torch.eye(parts)[None,None,:,:].repeat(scores,num_events-1,1,1)

        for score in range(scores):
            corps[score,corpus[score]] = 1

        e = Variable(torch.zeros([scores,self.context+1,self.parts,self.m]).cuda(),requires_grad=False)
        t = Variable(torch.zeros([scores,self.context+1,self.parts,self.dur_features]).cuda(),requires_grad=False)
        f = Variable(torch.zeros([scores,self.context+1,self.parts,self.parts]).cuda(),requires_grad=False)
        loc = Variable(torch.zeros([scores,48]).cuda(),requires_grad=False)
        c = Variable(torch.zeros(corps.shape).cuda(),requires_grad=False)

        frontier = np.zeros([scores,parts],dtype=np.int32)
        for k in range(self.context,self.context+num_events):
            print('.', end='')
            rear = np.min(frontier,axis=1)
            for score in range(scores):
                for part in range(parts):
                    if frontier[score,part] > rear[score]:
                        events[score,k,part] = events[score,k-1,part]
                        times[score,k,part,self.NULL] = 1

            if debug != None: debug(events[0,:k+1],times[0,:k+1],flow[0,:k+1],loc[0].data)

            for part in range(parts):
                loc.fill_(0)
                updates = torch.LongTensor([score for score in range(scores) if frontier[score,part] == rear[score]]).cuda()
                e[:len(updates)] = events[updates][:,k-self.context:k+1]
                e[:len(updates),:,0] = events[updates][:,k-self.context:k+1,part]
                e[:len(updates),:,part] = events[updates][:,k-self.context:k+1,0]
                t[:len(updates)] = times[updates][:,k-self.context:k+1]
                t[:len(updates),:,0] = times[updates][:,k-self.context:k+1,part]
                t[:len(updates),:,part] = times[updates][:,k-self.context:k+1,0]
                f[:len(updates)] = flow[updates][:,k-self.context:k+1] # no need to swap for now because all flow is hardcoded to identity
                c[:len(updates)] = corps[updates]
                for i,score in enumerate(updates):
                    loc[i,frontier[score,part]%48] = 1

                #if debug != None and frontier[0,part] == rear[0]: debug(e[0].data,t[0].data,f[0].data,loc[0].data)

                p = F.softmax(self.predict_rhythm(e[:len(updates),:self.context].contiguous(),
                                                  t[:len(updates),:self.context].contiguous(),
                                                  f[:len(updates),:self.context].contiguous(),
                                                  e[:len(updates),-1].contiguous(),
                                                  t[:len(updates),-1].contiguous(),
                                                  f[:len(updates),-1].contiguous(),
                                                  loc[:len(updates)],
                                                  c[:len(updates)]),dim=1)
                dts = torch.multinomial(p, num_samples=1).data[:,0]
                for i,score in enumerate(updates):
                    t[i,-1,0,3+dts[i]] = 1
                    frontier[score,part] += 48*self.dur_map[3+dts[i]]

                for note in range(self.m):
                    p = F.sigmoid(self.predict_notes(e[:len(updates),:self.context].contiguous(),
                                                     t[:len(updates),:self.context].contiguous(),
                                                     f[:len(updates),:self.context].contiguous(),
                                                     e[:len(updates),-1].contiguous(),
                                                     t[:len(updates),-1].contiguous(),
                                                     f[:len(updates),-1].contiguous(),
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
