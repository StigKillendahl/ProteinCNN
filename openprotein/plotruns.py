# This file is part of the ProteinCNN project.
#
# @author Stig Killendahl & Kevin Jon Jensen
#
# Based on the OpenProtein framework, please see the LICENSE file in the root directory.

import glob
import pickle
import matplotlib.pyplot as plt
import os
import numpy as np
from itertools import cycle


files = glob.glob('output/runs/*.p')

colors = cycle('bgrcmk')


for idx, f in enumerate(files):

    data = pickle.load(open(f, "rb"))
    c = next(colors)
    x = data['sample_num']
    y = data['drmsd_avg']
    y2 = data['train_loss_values']
    label = f.split('/')[-1] if os.name != 'nt' else f.split('\\')[-1]
    label = label.split('.')[0]
    plt.plot(x,y,label=label, c=c)
    plt.plot(x,y2,linestyle='dashed',c=c)
    plt.xlabel('Minibatches processed')
    plt.ylabel('dRMSD')


plt.legend()
plt.show()