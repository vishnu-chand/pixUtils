import queue
from concurrent.futures import _base
from functools import partial
from threading import Thread
from pixUtils import *


# from pixUtils.torchCommon import *


class ThreadBatch:
    def __init__(self, prcs, nConsumer, batchSize, qSize=200):
        self.prcs = prcs
        self.jobs = defaultdict(list)
        self.batchSize = batchSize
        self.q = queue.Queue(maxsize=qSize)
        [Thread(target=self.consumer).start() for name in range(nConsumer)]

    def getBatch(self):
        def getData(data, batch):
            f = data['f']
            try:
                res = dict()
                for k, v in data.items():
                    img = v
                    if type(img) == str:
                        if '0_0' in img:
                            raise Exception(f"fail {img}")
                    if isinstance(v, partial):
                        v = v()  # this can add a delay in q consecutive fetch
                    res[k] = v
                for k, v in res.items():
                    batch[k].append(v)
            except Exception as exp:
                f.set_exception(exp)

        batch = defaultdict(list)
        getData(self.q.get(block=True), batch)
        for i in range(self.batchSize - 1):
            print("40 getBatch threadBatch2 self.q.qsize(): ", self.q.qsize())
            try:
                getData(self.q.get(block=False), batch)  # fetch consecutive data
            except queue.Empty:
                break
        return batch

    def consumer(self):
        while True:
            batch = self.getBatch()
            if batch:
                fs = batch.pop('f')
                try:
                    rs = self.prcs(**batch)
                    for f, r in zip(fs, rs):
                        f.set_result(r)
                except Exception as exp:
                    for f in fs:
                        f.set_exception(exp)

    def submit(self, jid, **data):
        data['f'] = _base.Future()
        self.jobs[jid].append(data['f'])
        self.q.put(data)

    def getResults(self, jid):
        for job in self.jobs.pop(jid):
            try:
                msg = job.result()
            except Exception as exp:
                msg = f"{exp}"
            yield msg


def main():
    def prcs(img):
        if type(img) == str:
            if '0_0' in img:
                raise Exception(f"fail {img}")
        time.sleep(.1)
        # img = cv2.imread('')
        # img += 1
        return img

    tb = ThreadBatch(prcs, 2, 2)
    tik = clk()
    for jid in range(4):
        for qid in range(100):
            tb.submit(jid, img=f"{jid}_{qid}")
    for jid in range(4):
        for result in tb.getResults(jid):
            print("64 main threadBatch2 result: ", result)
    print("83  threadBatch tik.tok() ", tik.tok("").last())


main()
