import queue
import threading
import traceback
from collections import defaultdict


class ThreadBatch:
    def __init__(self, nConsumer, maxBatchSize, qSize):
        self.jobs = defaultdict(list)
        self.maxBatchSize = maxBatchSize
        self.q = queue.Queue(maxsize=qSize)
        for name in range(nConsumer):
            threading.Thread(target=self.consumer, args=[self.batchPrcs]).start()

    def getBatch(self):
        data = self.q.get(block=True)
        yield self.__prePrcs(data)
        for i in range(self.maxBatchSize - 1):
            try:
                data = self.q.get(block=True, timeout=0.5)  # fetch consecutive data
                yield self.__prePrcs(data)  # this add a delay in q fetch
            except queue.Empty:
                break

    def __prePrcs(self, data):
        try:
            newData = self.prePrcs(**data)
            data.update(newData)
            ok = True
        except:
            ok = False
            self.onException(data['jid'])
        return ok, data

    def consumer(self, prcs):
        while True:
            batch = defaultdict(list)
            dones = []
            for ok, data in self.getBatch():
                dones.append(data['jobDone'])
                if ok:
                    for k, v in data.items():
                        batch[k].append(v)
            if batch:
                try:
                    prcs(**batch)
                except:
                    for jid in zip(batch['jid']):
                        self.onException(jid)
            for done in dones:
                done.set()

    def submit(self, jid, **data):
        done = threading.Event()
        data['jobDone'], data['jid'] = done, jid
        self.jobs[jid].append(done)
        self.q.put(data)

    def isDone(self, jid):
        for job in self.jobs[jid]:
            job.wait()

    def batchPrcs(self, **kw):
        raise NotImplemented

    def prePrcs(self, **kw):
        raise NotImplemented

    def onException(self, jid, **kw):
        raise NotImplemented


class DummyThreadBatch:
    def __init__(self, *a, **kw):
        pass

    def submit(self, *a, **data):
        newData = self.prePrcs(**data)
        data.update(newData)
        data = {k: [v] for k, v in data.items()}
        self.batchPrcs(**data)

    def isDone(self, *a, **kw):
        pass

    def batchPrcs(self, **kw):
        raise NotImplemented

    def prePrcs(self, **kw):
        raise NotImplemented

    def onException(self, jid, **kw):
        raise NotImplemented
