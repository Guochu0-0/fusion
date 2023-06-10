import sys
import os

from Code.model.fusiongan import FusionGan
#sys.path.append("/home/ma-user/work/fusion")

from Code.model.rebuild import Rebuilder

if __name__ == '__main__':
    #rebuilder = Rebuilder()
    #rebuilder.train_over()
    fusiongan = FusionGan()
    fusiongan.train_over()