import threading
from MyQueue import myQueue
import time


class myThread (threading.Thread):
  def __init__(self, threadID, q, queueLock, func, params):
    threading.Thread.__init__(self)
    self.threadID = threadID
    self.q = q
    self.queueLock = queueLock
    self.func = func
    self.params = params

  def run(self):
    RunTrain(self.func, self.params, self.q, self.queueLock)


def RunTrain(func, params, q, queueLock):
  queueLock.acquire()
  while not q.isEmpty():
    label = q.dequeue()
    queueLock.release()
    print("Staring training for "+str(label)+"th label...")
    C = params["C"]
    func(label, params)
    print("Completed training for "+str(label)+"th label")
    time.sleep(1)
    queueLock.acquire()
  queueLock.release()
