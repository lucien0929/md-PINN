import sys
sys.path.insert(0, '../../Utilities/')

import tensorflow as tf
import numpy as np
import time
from utils import nest_tanh, initialize_NN


class mdPINN:
    # Initialize the class
    def __init__(self, xy_0, xy_b, xy_f, layers, layers1):
                  
        self.x_f = xy_f[:,0:1]
        self.t_f = xy_f[:,1:2]
        self.u_f = xy_f[:,2:3]

        self.x_0 = xy_0[:,0:1]
        self.t_0 = xy_0[:,1:2]
        self.u_0 = xy_0[:,2:3]

        self.x_b = xy_b[:,0:1]
        self.t_b = xy_b[:,1:2]
        self.u_b = xy_b[:,2:3]

  
        self.layers = layers
        self.layers1 = layers1

        self.c = 1.2
        self.alpha = tf.Variable([0.5], dtype=tf.float32)       
        self.beta = tf.Variable([1], dtype=tf.float32)         
        self.gamma = tf.Variable([0.1], dtype=tf.float32)

        # Initialize NNs
        self.weights, self.biases = initialize_NN(layers)
        self.weights1, self.biases1 = initialize_NN(layers1)

        # tf placeholders and graph
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=False,
                                                     log_device_placement=False))
        
        self.x_f_tf = tf.placeholder(tf.float32, shape=[None, self.x_f.shape[1]]) 
        self.t_f_tf = tf.placeholder(tf.float32, shape=[None, self.t_f.shape[1]])   
       
        self.x_0_tf = tf.placeholder(tf.float32, shape=[None, self.x_0.shape[1]]) 
        self.t_0_tf = tf.placeholder(tf.float32, shape=[None, self.t_0.shape[1]])   
       
        self.x_b_tf = tf.placeholder(tf.float32, shape=[None, self.x_b.shape[1]]) 
        self.t_b_tf = tf.placeholder(tf.float32, shape=[None, self.t_b.shape[1]])   


        self.T_pred_0 = self.net_T(self.x_0_tf, self.t_0_tf) 
        self.T_pred_b = self.net_T(self.x_b_tf, self.t_b_tf) 
        self.T_pred_f = self.net_T(self.x_f_tf, self.t_f_tf) 


        self.f = self.net_f(self.x_f_tf, self.t_f_tf) 
        self.fb = self.net_f(self.x_b_tf, self.t_b_tf) 
       
        lambda_reg = 1e-9

        trainable_vars = tf.trainable_variables()
        l2_reg = tf.add_n([tf.nn.l2_loss(v) for v in trainable_vars if 'bias' not in v.name])    
              
        self.loss_1 = tf.reduce_mean(tf.square(self.T_pred_0 - self.u_0))#q=5
       
        self.loss_2 = tf.reduce_mean(tf.square(self.T_pred_b - self.u_b))#q=5         
       
        self.loss_3 = tf.reduce_mean(tf.square(self.T_pred_f - self.u_f))#q=5    
       
        self.loss_4 = tf.reduce_mean(tf.square(self.f))
       
        self.loss = self.loss_1 + self.loss_2 + self.loss_3 + self.loss_4 + lambda_reg*l2_reg
        
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
        u = self.neural_net(tf.concat([x,y],1), self.weights, self.biases)
        u1 = self.neural_net(tf.concat([x,y],1), self.weights1, self.biases1)
        return u + self.beta*u1
 
    def net_f(self, x, t): 
        
        u =  self.net_T(x, t)
        
        u_x = tf.gradients(u, x)[0]
        u_t = tf.gradients(u, t)[0] 

        u_xx = tf.gradients(u_x, x)[0]
        u_xxx = tf.gradients(u_xx, x)[0]
        u_xxxx = tf.gradients(u_xxx, x)[0]

        f = u_t + u*u_x + u_xx + self.gamma*u_xxxx 
    
        return f
   
    def callback(self, loss, alpha, beta, gamma):
        print('Loss:', loss, 'alpha:', alpha, 'beta:', beta, 'gamma:', gamma)

        
    def train(self,nIter):
        
        tf_dict = {self.x_f_tf: self.x_f, 
                   self.t_f_tf: self.t_f,   
                   self.x_0_tf: self.x_0, 
                   self.t_0_tf: self.t_0,   
                   self.x_b_tf: self.x_b, 
                   self.t_b_tf: self.t_b}
            
        start_time = time.time()
        for it in range(nIter):
            self.sess.run(self.train_op_Adam, tf_dict)           
            # Print
            if it % 10 == 0:
                elapsed = time.time() - start_time
                loss_value = self.sess.run(self.loss, tf_dict)
                alpha = self.sess.run(self.alpha)
                beta = self.sess.run(self.beta)
                gamma = self.sess.run(self.gamma)

            print('It: %d, Loss: %.3e, alpha: %.5f, beta: %.5f, gamma: %.5f, Time: %.2f' % 
                    (it, loss_value, alpha, beta, gamma, elapsed))
            start_time = time.time()     
                                                                                
        self.optimizer.minimize(self.sess, 
                                feed_dict = tf_dict,         
                                fetches = [self.loss, self.alpha, self.beta, self.gamma], 
                                loss_callback = self.callback)              
                                    
    def predict(self, X_star1):
         
        tf_dict1 = {self.x_0_tf: X_star1[:,0:1], self.t_0_tf: X_star1[:,1:2]}     
         
        T_star = self.sess.run(self.T_pred_0, tf_dict1)  
       
        return T_star
    
