import sys
sys.path.insert(0, '../../Utilities/')

import tensorflow as tf
import numpy as np
import time
from model import mdPINN
from data import xy_test, xy_l, xy_r, xy_b, xy_m, xy_f, layers, layers1
from utils import plotfig
import matplotlib.pyplot as plt

if __name__ == "__main__": 

    model = mdPINN(xy_l, xy_r, xy_b, xy_m, xy_f, layers, layers1)
    
    start_time = time.time()                
    model.train(50000)
    elapsed = time.time() - start_time                
    print('Training time: %.4f' % (elapsed)) 
    
    T_pred = model.predict(xy_test[:,0:2])  
    
    error_T = np.linalg.norm(xy_test[:,2:3] - T_pred,2)/np.linalg.norm(xy_test[:,2:3],2)
    print('Error T: %e' % (error_T))

    np.savetxt(r'../T.txt', T_pred, fmt='%.9e')
    
    fig = plotfig(xy_test[:,0:1], xy_test[:,1:2], xy_test[:,2:3], T_pred, xy_test[:,2:3].min(0), xy_test[:,2:3].max(0))
    plt.show()