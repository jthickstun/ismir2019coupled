import os,signal,copy
from time import time
from contextlib import contextmanager

import numpy as np
import torch

def worker_init(args):
    signal.signal(signal.SIGINT, signal.SIG_IGN) # ignore signals so parent can handle them
    np.random.seed(os.getpid() ^ int(time())) # approximately random seed for workers

def optimize(model, train_set, test_set, learning_rate=0.01, momentum=.95, batch_size=1000, epochs=40, workers=4, update_rate=1000, l2=0, sample_size=50000):
    kwargs = {'num_workers': workers, 'pin_memory': True, 'worker_init_fn': worker_init}
    train_loader = torch.utils.data.DataLoader(dataset=train_set,batch_size=batch_size,shuffle=True,**kwargs)

    prng = np.random.RandomState(999)
    test_sampler = torch.utils.data.sampler.SubsetRandomSampler(prng.choice(range(len(test_set)),min(sample_size,len(test_set)),replace=False))
    test_loader = torch.utils.data.DataLoader(dataset=test_set,batch_size=batch_size,sampler=test_sampler,drop_last=True,**kwargs)
    train_sampler = torch.utils.data.sampler.SubsetRandomSampler(prng.choice(range(len(train_set)),min(sample_size,len(train_set)),replace=False))
    train_sample_loader = torch.utils.data.DataLoader(dataset=train_set,batch_size=batch_size,sampler=train_sampler,drop_last=True,**kwargs)

    print('Initiating optimizer, {} iterations/epoch.'.format(len(train_loader)))
    model.restore_checkpoint()
    print(model.status_header())

    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=l2)

    try:
        t = time()
        for epoch in range(epochs):
            for i, x in enumerate(train_loader):
                if i % update_rate == 0:
                    with model.iterate_averaging():
                        model.update_status(train_sample_loader, test_loader, t, time())
                        model.checkpoint()
                        print(model.status())
                    t = time()

                optimizer.zero_grad()
                x = model.prepare_data(x)
                loss = model.loss(model(x),x)
                loss.backward()
                optimizer.step()
                model.average_iterates()

    except KeyboardInterrupt:
        print('Graceful Exit')
    else:
        print('Finished')

def terminal_error(model, dataset, batch_size=1000, workers=4):
    kwargs = {'num_workers': workers, 'pin_memory': True, 'worker_init_fn': worker_init}
    dataloader = torch.utils.data.DataLoader(dataset=dataset,batch_size=batch_size,drop_last=True,**kwargs)
    (llt, lln), avp, acc = model.compute_stats(dataloader)
    return llt+lln,llt,lln,100*avp,100*acc
