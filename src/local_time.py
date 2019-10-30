import time

class LocalTime:
  def __init__(self):
    self.localtime = time.asctime( time.localtime(time.time()) )