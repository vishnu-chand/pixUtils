import concurrent.futures
import logging
import queue
import time


class Exe(concurrent.futures.ThreadPoolExecutor):
    def __init__(self, max_workers=None, thread_name_prefix='', q=None):
        super().__init__(max_workers=max_workers, thread_name_prefix=thread_name_prefix)
        if q is None:
            q = queue.Queue()
        self._work_queue = q


executor2 = Exe(q=queue.Queue(maxsize=100))


def producer(name):
    res = []
    for i in range(5):
        time.sleep(.1)
        a = executor2.submit(consumer, f'{name}:{i}')
        res.append(a)
    for re in res:
        print(f"18: {re.result()} {executor2._work_queue.qsize()}")


def consumer(msg):
    time.sleep(.2)
    return msg


def main():
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)
    for i in range(4):
        executor.submit(producer, i)


if __name__ == "__main__":
    main()
