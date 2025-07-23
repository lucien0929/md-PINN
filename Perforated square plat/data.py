import numpy as np

xy_test = np.loadtxt('../reference.txt')
xy_l = np.loadtxt('../left_boundary.txt')#.reshape(12657,1)
xy_r = np.loadtxt('../right_boundary.txt')#.reshape(12657,1)
xy_b = np.loadtxt('../adiabatic_boundary.txt')#.reshape(12657,1)
xy_m = np.loadtxt('../middle_boundary.txt')[:,0:2]

idx_f = np.random.choice(xy_test.shape[0], 2000, replace=False)
xy_f = xy_test[idx_f]

layers = [2, 20, 20, 20, 20, 1]           
layers1 = [2, 50, 1]

