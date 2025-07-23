import numpy as np

xy_test = np.loadtxt('../reference.txt')
xy_0 = xy_test[xy_test[:, 1] == 0]
xy_b = np.vstack((xy_test[xy_test[:, 0] == 10], xy_test[xy_test[:, 0] == -10]))

idx_f = np.random.choice(xy_test.shape[0], 1000, replace=False)
xy_f = xy_test[idx_f]


layers = [2, 20, 20, 20, 20, 20, 1]           
layers1 = [2, 50, 1]

