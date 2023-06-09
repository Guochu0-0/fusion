import sys
import os

#sys.path.append("/home/ma-user/work/fusion")

from Code.model.rebuild import Rebuilder

if __name__ == '__main__':
    rebuilder = Rebuilder()
    rebuilder.train_over()