import threading
import Queue
import time

class myThread (threading.Thread):
  def __init__(self, threadID, q):
    threading.Thread.__init__(self)
    self.threadID = threadID
    self.q = q

  def run(self, func, params):
    RunTrain(func, params, self.q)


def RunTrain(func, params, q):
  queueLock.acquire()
  while not workQueue.empty():
    label = q.get()
    queueLock.release()
    print("Staring training for "+str(label)+"th label...")
    C = params["C"]
    func(l, params)
    print("Completed training for "+str(label)+"th label")
    time.sleep(1)
    queueLock.acquire()
  queueLock.release()
