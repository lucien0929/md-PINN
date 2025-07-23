import sys
sys.path.insert(0, '../../Utilities/')

import tensorflow as tf
import numpy as np
import time
from utils import nest_tanh, initialize_NN


class mdPINN:
    # Initialize the class
    def __init__(self, xy_u, xy_b, xy_p, xy_f, layers, layers1, layers2, layers3, layers4, layers5):
                  
        
        self.x_u = xy_u[:,0:1]
        self.y_u = xy_u[:,1:2]
        
        self.x_b = xy_b[:,0:1]
        self.y_b = xy_b[:,1:2]
        
        self.x_p = xy_p[:,0:1]
        self.y_p = xy_p[:,1:2]

        self.x_f = xy_f[:,0:1]
        self.y_f = xy_f[:,1:2]
  
        self.layers = layers
        self.layers1 = layers1

        self.c = 1.2
        self.alpha = tf.Variable([0.5], dtype=tf.float32)       
        self.beta = tf.Variable([1], dtype=tf.float32)

        # Initialize NNs
        self.weights, self.biases = initialize_NN(layers)
        self.weights1, self.biases1 = initialize_NN(layers1)
        self.weights2, self.biases2 = initialize_NN(layers2)
        self.weights3, self.biases3 = initialize_NN(layers3)
        self.weights4, self.biases4 = initialize_NN(layers4)
        self.weights5, self.biases5 = initialize_NN(layers5)

        # tf placeholders and graph
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=False,
                                                     log_device_placement=False))
        
        self.x_u_tf = tf.placeholder(tf.float32, shape=[None, self.x_u.shape[1]]) 
        self.y_u_tf = tf.placeholder(tf.float32, shape=[None, self.y_u.shape[1]]) 
        self.x_b_tf = tf.placeholder(tf.float32, shape=[None, self.x_b.shape[1]]) 
        self.y_b_tf = tf.placeholder(tf.float32, shape=[None, self.y_b.shape[1]]) 

        self.x_p_tf = tf.placeholder(tf.float32, shape=[None, self.x_p.shape[1]]) 
        self.y_p_tf = tf.placeholder(tf.float32, shape=[None, self.y_p.shape[1]])   
 
        self.x_f_tf = tf.placeholder(tf.float32, shape=[None, self.x_f.shape[1]]) 
        self.y_f_tf = tf.placeholder(tf.float32, shape=[None, self.y_f.shape[1]])
        
        
        self.uu_pred = self.net_u(self.x_u_tf, self.y_u_tf)
        self.vu_pred = self.net_v(self.x_u_tf, self.y_u_tf)
        
        self.ub_pred = self.net_u(self.x_b_tf, self.y_b_tf)
        self.vb_pred = self.net_v(self.x_b_tf, self.y_b_tf)
        
        self.pp_pred = self.net_p(self.x_p_tf, self.y_p_tf)    
        
        self.uf_pred = self.net_u(self.x_f_tf, self.y_f_tf)
        self.vf_pred = self.net_v(self.x_f_tf, self.y_f_tf)
        self.pf_pred = self.net_p(self.x_f_tf, self.y_f_tf)

        self.fuf, self.fvf, self.ff = self.pde(self.x_f_tf, self.y_f_tf) 
        
        lambda_reg = 1e-9

        trainable_vars = tf.trainable_variables()
        l2_reg = tf.add_n([tf.nn.l2_loss(v) for v in trainable_vars if 'bias' not in v.name])           
    
        self.loss_1 = tf.reduce_mean(tf.square(self.uu_pred - 1)) + \
                      tf.reduce_mean(tf.square(self.vu_pred)) 
        
        self.loss_2 = tf.reduce_mean(tf.square(self.ub_pred)) + \
                      tf.reduce_mean(tf.square(self.vb_pred)) 
        
        self.loss_3 = tf.reduce_mean(tf.square(self.pp_pred)) 
        
        self.loss_4 = tf.reduce_mean(tf.square(self.fuf)) + \
                      tf.reduce_mean(tf.square(self.fvf)) + \
                      tf.reduce_mean(tf.square(self.ff)) 
                      
        self.loss =  self.loss_1  + self.loss_2 + self.loss_3 + self.loss_4 + lambda_reg*l2_reg
        
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

    def net_u(self, x, y):  
        u = self.neural_net(tf.concat([x,y],1), self.weights, self.biases)
        u1 = self.neural_net(tf.concat([x,y],1), self.weights1, self.biases1)
        return u + self.beta*u1
    
    def net_v(self, x, y):  
        v = self.neural_net(tf.concat([x,y],1), self.weights2, self.biases2)
        v1 = self.neural_net(tf.concat([x,y],1), self.weights3, self.biases3)
        return v + self.beta*v1
    
    def net_p(self, x, y):  
        p = self.neural_net(tf.concat([x,y],1), self.weights4, self.biases4)
        p1 = self.neural_net(tf.concat([x,y],1), self.weights5, self.biases5)
        return p + self.beta*p1
   
    def pde(self, x, y):
        
        u = self.net_u(x,y)
        v = self.net_v(x,y)
        p = self.net_p(x,y)

        u_x = tf.gradients(u, x)[0]
        u_y = tf.gradients(u, y)[0] 
        v_x = tf.gradients(v, x)[0]
        v_y = tf.gradients(v, y)[0]
        p_x = tf.gradients(p, x)[0]
        p_y = tf.gradients(p, y)[0]
        
        u_xx = tf.gradients(u_x, x)[0]
        u_yy = tf.gradients(u_y, y)[0]
        v_xx = tf.gradients(v_x, x)[0]
        v_yy = tf.gradients(v_y, y)[0]
        
        fu = (u*u_x + v*u_y) + p_x - 0.01*(u_xx + u_yy) 
        fv = (u*v_x + v*v_y) + p_y - 0.01*(v_xx + v_yy)
        f = u_x + v_y
            
        return fu, fv, f
   
    def callback(self, loss, alpha, beta):
        print('Loss:', loss, 'alpha:', alpha, 'beta:', beta)

        
    def train(self,nIter):
        
        tf_dict = {self.x_u_tf: self.x_u, 
                   self.y_u_tf: self.y_u,
                   self.x_b_tf: self.x_b, 
                   self.y_b_tf: self.y_b,
                   self.x_p_tf: self.x_p, 
                   self.y_p_tf: self.y_p,
                   self.x_f_tf: self.x_f,
                   self.y_f_tf: self.y_f}
            
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
                                    
    def predict(self, X_star):
              
        u_star = self.sess.run(self.uf_pred, {self.x_f_tf: X_star[:,0:1], self.y_f_tf: X_star[:,1:2]})  
        v_star = self.sess.run(self.vf_pred, {self.x_f_tf: X_star[:,0:1], self.y_f_tf: X_star[:,1:2]})  
        p_star = self.sess.run(self.pf_pred, {self.x_f_tf: X_star[:,0:1], self.y_f_tf: X_star[:,1:2]}) 
     
        return u_star, v_star, p_star
    
