import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

def nest_tanh(x, alpha, c):
    y = (tf.tanh(alpha*x)) + c
    y1 = tf.tanh(alpha*x*y)
    y2 = tf.tanh(alpha*x*y1)
    return(y2)

def plotfig(x, y, u1, u2, color_l, color_r):
    fig = plt.figure(figsize=(10,4))
    ax = fig.add_subplot(121) 
    plt.scatter(x, y, s = 5, c =u1, cmap = 'jet', vmin = color_l, vmax = color_r)
    plt.axis('equal')
    plt.colorbar()
    plt.title('reference')
    ax = fig.add_subplot(122)
    plt.scatter(x, y, s = 5, c =u2, cmap = 'jet', vmin = color_l, vmax = color_r)
    plt.axis('equal')
    plt.colorbar()
    plt.title('predicted')
    return fig

def initialize_NN(layers):        
    weights = []
    biases = []
    num_layers = len(layers) 
    for l in range(0,num_layers-1):
        W = xavier_init(size=[layers[l], layers[l+1]])
        b = tf.Variable(tf.zeros([1,layers[l+1]], dtype=tf.float32), dtype=tf.float32)
        weights.append(W)
        biases.append(b)        
    return weights, biases
        
def xavier_init(size):
    in_dim = size[0]
    out_dim = size[1]        
    xavier_stddev = np.sqrt(2/(in_dim + out_dim))
    return tf.Variable(tf.truncated_normal([in_dim, out_dim], stddev=xavier_stddev), dtype=tf.float32)
    
    
