import logging
import math
logger = logging.getLogger()

class Clock():
    """
    A clock that return true every #epoch update
    """
    def __init__(self, epoch, start_status = True):
        self.clock = 0 + start_status*(math.ceil(epoch)-1)
        self.epoch = epoch
        logger.debug("Clock started with epoch={0} and clock={1}".format(self.epoch, self.clock))

    def update(self):
        self.clock += 1
        if self.clock >= self.epoch:
            self.clock = 0
            return True
            
            
def chuncker(iterable, n):
    """
    grouper([ABCDEFG], 3) --> [[ABC],[DEF],[G]]
    """
    ind = range(0,len(iterable), n)
    for i in range(len(ind)-1):
        yield iterable[ind[i]:ind[i+1]]
    if ind[-1] < len(iterable):
        yield iterable[ind[-1]:len(iterable)]
