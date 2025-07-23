import numpy as np

xy_test = np.loadtxt('../reference.txt')
xy_u = np.loadtxt('../upper_boundary.txt')
xy_b =  np.loadtxt('../no_slip_boundary.txt')

idx_f = np.random.choice(xy_test.shape[0], 1000, replace=False)
xy_f = xy_test[idx_f]

xy_p = np.hstack((np.full((20,1), -0.5), np.full((20,1), -0.5)))

layers = [2, 20, 20, 20, 20, 1]           
layers1 = [2, 50, 1]
layers2 = [2, 20, 20, 20, 20, 1]           
layers3 = [2, 50, 1]
layers4 = [2, 20, 20, 20, 20, 1]           
layers5 = [2, 50, 1]

