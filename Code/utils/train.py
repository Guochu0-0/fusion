import sys
import os
#sys.path.append("/home/ma-user/work/fusion")

from Code.model.ae_pretrain import AE_pretrainer
from Code.model.fusiongan import FusionGan


from Code.model.rebuild import Rebuilder

if __name__ == '__main__':
    #rebuilder = Rebuilder()
    #rebuilder.train_over()
    fusiongan = FusionGan()
    fusiongan.train_over()
    #pretrainer = AE_pretrainer()
    #pretrainer.train_over()