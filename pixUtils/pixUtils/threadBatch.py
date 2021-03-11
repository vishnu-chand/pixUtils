import queue
from queue import Queue
from functools import partial
from threading import Thread, Event
from collections import defaultdict


class ThreadBatch:
    def __init__(self, prcs, nConsumer, batchSize, qSize=200):
        self.jobs = defaultdict(list)
        self.batchSize = batchSize
        self.q = Queue(maxsize=qSize)
        [Thread(target=self.consumer, args=prcs).start() for name in range(nConsumer)]

    def getBatch(self):
        q = self.q
        data = q.get(block=True)
        batch = defaultdict(list)
        for k, v in data.items():
            if isinstance(v, partial):
                v = v()
            batch[k].append(v)
        for i in range(self.batchSize - 1):
            try:
                data = q.get(block=False)  # fetch consecutive data
                for k, v in data.items():
                    if isinstance(v, partial):
                        v = v()   # this can add a delay in q fetch
                    batch[k].append(v)
            except queue.Empty:
                break
        return batch

    def consumer(self, prcs):
        while True:
            batch = self.getBatch()
            threadDone = batch.pop('threadDone')
            prcs(**batch)
            for done in threadDone:
                done.set()

    def submit(self, jid, **data):
        done = Event()
        data['threadDone'] = done
        self.jobs[jid].append(done)
        self.q.put(data)

    def isDone(self, jid):
        for job in self.jobs[jid]:
            job.wait()


class ThreadBatchDummy:
    def __init__(self, prcs, nConsumer, batchSize, qSize=200):
        self.prcs = prcs

    def submit(self, jid, **data):
        batch = defaultdict(list)
        for k, v in data.items():
            if isinstance(v, partial):
                v = v()  # this can add a delay in q fetch
            batch[k].append(v)
        for prc in self.prcs:
            prc(**batch)

    def isDone(self, jid):
        return


