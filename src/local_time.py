import time

class LocalTime:
  def get():
    return time.asctime( time.localtime(time.time()) )