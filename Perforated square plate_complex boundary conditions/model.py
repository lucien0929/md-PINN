import sys
sys.path.insert(0, '../../Utilities/')

import tensorflow as tf
import numpy as np
import time
from utils import nest_tanh, initialize_NN

class mdPINN:
    # Initialize the class
    def __init__(self, xy_l, xy_r, xy_b, xy_m, xy_f, layers, layers1):
                  
        
        self.x_f = xy_f[:,0:1]
        self.y_f = xy_f[:,1:2]

        self.x_l = xy_l[:,0:1]
        self.y_l = xy_l[:,1:2]

        self.x_r = xy_r[:,0:1]
        self.y_r = xy_r[:,1:2]

        self.x_b = xy_b[:,0:1]
        self.y_b = xy_b[:,1:2]

        
        self.x_m = xy_m[:,0:1]
        self.y_m = xy_m[:,1:2]
        
    
        self.layers = layers
        self.layers1 = layers1
        
        self.c = 1.2
        self.alpha = tf.Variable([0.5], dtype=tf.float32)       
        self.beta = tf.Variable([1], dtype=tf.float32) 

        # Initialize NNs
        self.weights, self.biases = initialize_NN(layers)
        self.weights1, self.biases1 = initialize_NN(layers1)
        # tf placeholders and graph
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=False,
                                                     log_device_placement=False))
        
        
        self.x_f_tf = tf.placeholder(tf.float32, shape=[None, self.x_f.shape[1]]) 
        self.y_f_tf = tf.placeholder(tf.float32, shape=[None, self.y_f.shape[1]])   
        
        self.x_l_tf = tf.placeholder(tf.float32, shape=[None, self.x_l.shape[1]]) 
        self.y_l_tf = tf.placeholder(tf.float32, shape=[None, self.y_l.shape[1]])   
        
        self.x_r_tf = tf.placeholder(tf.float32, shape=[None, self.x_r.shape[1]]) 
        self.y_r_tf = tf.placeholder(tf.float32, shape=[None, self.y_r.shape[1]])   
        
        self.x_b_tf = tf.placeholder(tf.float32, shape=[None, self.x_b.shape[1]]) 
        self.y_b_tf = tf.placeholder(tf.float32, shape=[None, self.y_b.shape[1]])  
        
        self.x_m_tf = tf.placeholder(tf.float32, shape=[None, self.x_m.shape[1]]) 
        self.y_m_tf = tf.placeholder(tf.float32, shape=[None, self.y_m.shape[1]])   

        self.T_pred_m = self.net_T(self.x_m_tf, self.y_m_tf) 
        self.T_pred_r = self.net_T(self.x_r_tf, self.y_r_tf) 
 
        self.f, self.T_xf, self.T_xf = self.net_f(self.x_f_tf, self.y_f_tf) 
        
        self.fb, self.T_xb, self.T_yb = self.net_f(self.x_b_tf, self.y_b_tf) 
        self.fm, self.T_xm, self.T_ym = self.net_f(self.x_m_tf, self.y_m_tf) 
        self.fr, self.T_xr, self.T_yr = self.net_f(self.x_r_tf, self.y_r_tf) 
        self.fl, self.T_xl, self.T_yl = self.net_f(self.x_l_tf, self.y_l_tf) 

        lambda_reg = 1e-9

        trainable_vars = tf.trainable_variables()
        l2_reg = tf.add_n([tf.nn.l2_loss(v) for v in trainable_vars if 'bias' not in v.name])   
              
        self.loss_1 = tf.reduce_mean(tf.square(self.T_pred_m))
        
        self.loss_2 = tf.reduce_mean(tf.square(self.T_yb))
        
        self.loss_3 = tf.reduce_mean(tf.square(-0.01*self.T_xl - 4*self.y_l**2))
        
        self.loss_4 = tf.reduce_mean(tf.square(-0.01*self.T_xr - 10*(self.T_pred_r - 25)))
        
        self.loss_5 = tf.reduce_mean(tf.square(self.f))
        
        self.loss = self.loss_1 + self.loss_2 + self.loss_3 + self.loss_4 + self.loss_5 + lambda_reg*l2_reg
        
        self.optimizer = tf.contrib.opt.ScipyOptimizerInterface(self.loss, 
                                                                method = 'L-BFGS-B', 
                                                                options = {'maxiter': 50000,
                                                                           'maxfun': 50000,
                                                                           'maxcor': 50,
                                                                           'maxls': 50,
                                                                           'ftol' : 1.0 * np.finfo(float).eps})
        
        self.optimizer_Adam = tf.train.AdamOptimizer()
        self.train_op_Adam = self.optimizer_Adam.minimize(self.loss)
        
        init = tf.global_variables_initializer()
        self.sess.run(init)
       
    def neural_net(self, X, weights, biases):
        num_layers = len(weights) + 1
        H = X
        for l in range(0,num_layers-2):
            W = weights[l]
            b = biases[l]
            H = tf.add(tf.matmul(H, W), b)
            H = nest_tanh(H, self.alpha, self.c)
        W = weights[-1]
        b = biases[-1]
        Y = tf.add(tf.matmul(H, W), b)
        return Y

    def net_T(self, x, y):          
        T = self.neural_net(tf.concat([x,y],1), self.weights, self.biases)
        T1 = self.neural_net(tf.concat([x,y],1), self.weights1, self.biases1)
        return T + self.beta*T1
    
    def net_f(self, x, y): 
        
        T =  self.net_T(x, y)
        
        T_x = tf.gradients(T, x)[0]
        T_y = tf.gradients(T, y)[0] 

        T_xx = tf.gradients(T_x, x)[0]
        T_yy = tf.gradients(T_y, y)[0] 

        f = 0.01*(T_xx + T_yy)
    
        return f, T_x, T_y
   
    def callback(self, loss, alpha, beta):
        print('Loss:', loss, 'alpha:', alpha, 'beta:', beta)

        
    def train(self,nIter):
        
        tf_dict = {self.x_f_tf: self.x_f, 
                   self.y_f_tf: self.y_f,   
                   self.x_l_tf: self.x_l, 
                   self.y_l_tf: self.y_l,   
                   self.x_r_tf: self.x_r, 
                   self.y_r_tf: self.y_r,   
                   self.x_b_tf: self.x_b, 
                   self.y_b_tf: self.y_b,   
                   self.x_m_tf: self.x_m, 
                   self.y_m_tf: self.y_m}
            
        start_time = time.time()
        for it in range(nIter):
            self.sess.run(self.train_op_Adam, tf_dict)           
            # Print
            if it % 10 == 0:
                elapsed = time.time() - start_time
                loss_value = self.sess.run(self.loss, tf_dict)               
                alpha = self.sess.run(self.alpha)
                beta = self.sess.run(self.beta)

            print('It: %d, Loss: %.3e, alpha: %.5f, beta: %.5f, Time: %.2f' % 
                    (it, loss_value, alpha, beta, elapsed))
            start_time = time.time()     
                                                                                
        self.optimizer.minimize(self.sess, 
                                feed_dict = tf_dict,         
                                fetches = [self.loss, self.alpha, self.beta], 
                                loss_callback = self.callback)                  
                                    
    def predict(self, X_star1):
         
        tf_dict1 = {self.x_m_tf: X_star1[:,0:1], self.y_m_tf: X_star1[:,1:2]}     
         
        T_star = self.sess.run(self.T_pred_m, tf_dict1) 
        
        return T_star
    
