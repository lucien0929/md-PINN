import sys
sys.path.insert(0, '../../Utilities/')

import tensorflow as tf
import numpy as np
import time
from model import mdPINN
from data import xy_u, xy_b, xy_p, xy_f, xy_test, layers, layers1, layers2, layers13, layers4, layers5

if __name__ == "__main__": 

    model = mdPINN(xy_u, xy_b, xy_p, xy_f, layers, layers1, layers2, layers13, layers4, layers5)
    
    start_time = time.time()                
    model.train(50000)
    elapsed = time.time() - start_time                
    print('Training time: %.4f' % (elapsed)) 
    
    u_pred, v_pred, p_pred = model.predict(xy_test[:,0:2])
       
    u = xy_test[:,2:3]
    v = xy_test[:,3:4]
    p = xy_test[:,4:5]
        
    error_u = np.linalg.norm(u - u_pred,2)/np.linalg.norm(u,2)
    print('Error u: %e' % (error_u)) 
    
    error_v = np.linalg.norm(v - v_pred,2)/np.linalg.norm(v,2)               
    print('Error v: %e' % (error_v)) 
    
    error_p = np.linalg.norm(p - p_pred,2)/np.linalg.norm(p,2)
    print('Error p: %e' % (error_p)) 
    
    np.savetxt(r'../u.txt', u_pred, fmt='%.9e')
    np.savetxt(r'../v.txt', v_pred, fmt='%.9e')
    np.savetxt(r'../p.txt', p_pred, fmt='%.9e')