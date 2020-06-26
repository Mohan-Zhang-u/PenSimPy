from skopt import gp_minimize
import numpy as np

noise_level = 0.1

from random import random, seed
seed(777)
def obj_fun(x, noise_level=noise_level):
    return np.sin(5 * x[0]) * (1 - np.tanh(x[0] ** 2)) + np.random.randn() * noise_level


res = gp_minimize(obj_fun,  # the function to minimize
                  [(-2.0, 2.0)],  # the bounds on each dimension of x
                  x0=[0.2],  # the starting point
                  acq_func="LCB",  # the acquisition function (optional)
                  n_calls=15,  # the number of evaluations of f including at x0
                  n_random_starts=0,  # the number of random initialization points
                  random_state=777)

print(res)
