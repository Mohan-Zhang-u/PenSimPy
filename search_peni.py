from pensimpy.helper.get_recipe_trend import get_recipe_trend
from pensimpy.env_setup.peni_env_setup import PenSimEnv
import numpy as np
from random import random, seed
from skopt import gp_minimize
from skopt.space import Real, Integer
import math
from gpflowopt.domain import ContinuousParameter
from gpflowopt.design import LatinHyperCube, FactorialDesign, RandomDesign
from gpflowopt.bo import BayesianOptimizer
from gpflowopt.acquisition import ExpectedImprovement
import gpflow
from gpflowopt.optim import SciPyOptimizer
import GPy
import GPyOpt
from math import log
import torch
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_model
from gpytorch.mlls import ExactMarginalLogLikelihood
import matplotlib.pyplot as plt
from botorch.acquisition import UpperConfidenceBound
from botorch.optim import optimize_acqf


class RecipeBuilder:
    """builds the recipe that can run with the env, given setpoints"""

    def __init__(self, random_int):
        self.Fs_trend = None
        self.Foil_trend = None
        self.Fg_trend = None
        self.pres_trend = None
        self.discharge_trend = None
        self.water_trend = None
        self.random_int = random_int

    def init_recipe(self, Fs_sp, Foil_sp, Fg_sp, pres_sp, discharge_sp, water_sp):
        Fs = [15, 60, 80, 100, 120, 140, 160, 180, 200, 220, 240, 260, 280, 300, 320, 340, 360, 380, 400, 800, 1750]
        self.Fs_trend = get_recipe_trend(Fs, Fs_sp)
        Foil = [20, 80, 280, 300, 320, 340, 360, 380, 400, 1750]
        self.Foil_trend = get_recipe_trend(Foil, Foil_sp)
        Fg = [40, 100, 200, 450, 1000, 1250, 1750]
        self.Fg_trend = get_recipe_trend(Fg, Fg_sp)
        pres = [62, 125, 150, 200, 500, 750, 1000, 1750]
        self.pres_trend = get_recipe_trend(pres, pres_sp)
        discharge = [500, 510, 650, 660, 750, 760, 850, 860, 950, 960, 1050,
                     1060, 1150, 1160, 1250, 1260, 1350, 1360, 1750]
        self.discharge_trend = get_recipe_trend(discharge, discharge_sp)
        water = [250, 375, 750, 800, 850, 1000, 1250, 1350, 1750]
        self.water_trend = get_recipe_trend(water, water_sp)

        # exception
        PAA = [25, 200, 1000, 1500, 1750]
        PAA_sp = [5, 0, 10, 4, 0]
        self.PAA_trend = get_recipe_trend(PAA, PAA_sp)

    def recipe_at_t(self, t):
        t -= 1
        return self.Fs_trend[t], self.Foil_trend[t], self.Fg_trend[t], self.pres_trend[t], self.discharge_trend[t], \
               self.water_trend[t], self.PAA_trend[t]

    def split(self, x):
        Fs_len, Foil_len, Fg_len, pres_len, discharge_len, water_len = 21, 10, 7, 8, 20, 9
        recipe_Fs_sp = x[:Fs_len]
        recipe_Foil_sp = x[Fs_len:
                           Fs_len + Foil_len]
        recipe_Fg_sp = x[Fs_len + Foil_len:
                         Fs_len + Foil_len + Fg_len]
        recipe_pres_sp = x[Fs_len + Foil_len + Fg_len:
                           Fs_len + Foil_len + Fg_len + pres_len]
        recipe_discharge_sp = x[Fs_len + Foil_len + Fg_len + pres_len:
                                Fs_len + Foil_len + Fg_len + pres_len + discharge_len]
        recipe_water_sp = x[Fs_len + Foil_len + Fg_len + pres_len + discharge_len:
                            Fs_len + Foil_len + Fg_len + pres_len + discharge_len + water_len]

        return recipe_Fs_sp, recipe_Foil_sp, recipe_Fg_sp, recipe_pres_sp, recipe_discharge_sp, recipe_water_sp

    def get_batch_yield(self, sp_points):
        Fs_sp, Foil_sp, Fg_sp, pres_sp, discharge_sp, water_sp = self.split(sp_points)

        env = PenSimEnv(random_seed_ref=self.random_int)
        # env = PenSimEnv(random_seed_ref=np.random.randint(1000))
        done = False
        observation, batch_data = env.reset()
        self.init_recipe(Fs_sp, Foil_sp, Fg_sp, pres_sp, discharge_sp, water_sp)
        time_stamp, batch_yield, yield_pre = 0, 0, 0
        while not done:
            time_stamp += 1
            Fs, Foil, Fg, Fpres, Fdischarge, Fw, Fpaa = self.recipe_at_t(time_stamp)
            observation, batch_data, reward, done = env.step(time_stamp,
                                                             batch_data,
                                                             Fs, Foil, Fg, Fpres, Fdischarge, Fw, Fpaa)
            batch_yield += reward
        return -batch_yield

    def get_batch_yield_gpflow(self, X):
        x = [8, 15, 30, 75, 150, 30, 37, 43, 47, 51, 57, 61, 65, 72, 76, 80, 84, 90, 116, 90, 80,
             22, 30, 35, 34, 33, 32, 31, 30, 29, 23,
             30, 42, 55, 60, 75, 65, 60,
             0.6, 0.7, 0.8, 0.9, 1.1, 1, 0.9, 0.9,
             0, 4000, 0, 4000, 0, 4000, 0, 4000, 0, 4000, 0, 4000, 0, 4000, 0, 4000, 0, 4000, 0, 0,
             0, 500, 100, 0, 400, 150, 250, 0, 100]
        yields = []
        for i in range(len(X)):
            water_sp = X[i].tolist()
            Fs_sp, Foil_sp, Fg_sp, pres_sp, discharge_sp, _ = self.split(x)

            # Fs_sp, Foil_sp, Fg_sp, pres_sp, discharge_sp, water_sp = self.split(X[i].tolist())

            env = PenSimEnv(random_seed_ref=self.random_int)
            done = False
            observation, batch_data = env.reset()
            self.init_recipe(Fs_sp, Foil_sp, Fg_sp, pres_sp, discharge_sp, water_sp)
            time_stamp, batch_yield, yield_pre = 0, 0, 0
            while not done:
                time_stamp += 1
                Fs, Foil, Fg, Fpres, Fdischarge, Fw, Fpaa = self.recipe_at_t(time_stamp)
                observation, batch_data, reward, done = env.step(time_stamp,
                                                                 batch_data,
                                                                 Fs, Foil, Fg, Fpres, Fdischarge, Fw, Fpaa)
                batch_yield += reward
            yields.append(-batch_yield)

        yields = np.array(yields)
        return yields[:, None]

    def get_bound(self, x, x_opt, manup_scale):
        lower_bound = x - x * manup_scale
        upper_bound = x + x * manup_scale

        lower_bound_new = x_opt - x_opt * manup_scale
        upper_bound_new = x_opt + x_opt * manup_scale

        lower_bound_new = lower_bound_new if lower_bound_new > lower_bound else lower_bound
        upper_bound_new = upper_bound_new if upper_bound_new < upper_bound else upper_bound

        return lower_bound_new, upper_bound_new

    def gpyopt(self):
        # x = [8, 15, 30, 75, 150, 30, 37, 43, 47, 51, 57, 61, 65, 72, 76, 80, 84, 90, 116, 90, 80,
        #      22, 30, 35, 34, 33, 32, 31, 30, 29, 23,
        #      30, 42, 55, 60, 75, 65, 60,
        #      0.6, 0.7, 0.8, 0.9, 1.1, 1, 0.9, 0.9,
        #      0, 4000, 0, 4000, 0, 4000, 0, 4000, 0, 4000, 0, 4000, 0, 4000, 0, 4000, 0, 4000, 0, 0,
        #      0, 500, 100, 0, 400, 150, 250, 0, 100]

        x = [0, 500, 100, 0, 400, 150, 250, 0, 100]

        lower = [ele * 0.9 if ele != 0 else ele for ele in x]
        upper = [ele * 1.1 if ele != 0 else 1 for ele in x]
        bounds = []
        for i in range(len(x)):
            tmp = {'name': f'x{i}', 'type': 'continuous', 'domain': (lower[i], upper[i])}
            bounds.append(tmp)

        constraints = [
            {
                'name': 'constrain_1',
                'constraint': '(x[:,0] + x[:,1]) - 23'
            },
        ]

        seed(274)
        myBopt = GPyOpt.methods.BayesianOptimization(f=self.get_batch_yield_gpflow,
                                                     initial_design_numdata=4,
                                                     domain=bounds,
                                                     verbosity=True)
        myBopt.run_optimization(max_iter=100, eps=-1)
        print(f"=== x_opt: {myBopt.x_opt}")
        print(f"=== fx_opt: {myBopt.fx_opt}")
        # print(f"=== before: {self.get_batch_yield(x)}")

        plt.plot(myBopt.Y.T[0])
        plt.show()

    def botorch(self):
        x = [8, 15, 30, 75, 150, 30, 37, 43, 47, 51, 57, 61, 65, 72, 76, 80, 84, 90, 116, 90, 80,
             22, 30, 35, 34, 33, 32, 31, 30, 29, 23,
             30, 42, 55, 60, 75, 65, 60,
             0.6, 0.7, 0.8, 0.9, 1.1, 1, 0.9, 0.9,
             0, 4000, 0, 4000, 0, 4000, 0, 4000, 0, 4000, 0, 4000, 0, 4000, 0, 4000, 0, 4000, 0, 0,
             0, 500, 100, 0, 400, 150, 250, 0, 100]

        X = []
        for _ in range(10):
            tmp = [ele * (1 + np.random.randint(-10, 10) / 100) for ele in x]
            X.append(tmp)

        train_X = torch.FloatTensor(X)
        print(f"=== train_X: {train_X.shape}")

        Y = self.get_batch_yield_gpflow(train_X)
        train_Y = torch.FloatTensor(Y)

        print(f"=== train_Y: {train_Y.shape}")
        gp = SingleTaskGP(train_X, train_Y)
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        fit_gpytorch_model(mll)

        UCB = UpperConfidenceBound(gp, beta=0.1)

        lower = [ele * 0.9 if ele != 0 else ele for ele in x]
        upper = [ele * 1.1 if ele != 0 else 1 for ele in x]

        bounds = torch.stack([torch.FloatTensor(lower), torch.FloatTensor(upper)])
        candidate, acq_value = optimize_acqf(
            UCB, bounds=bounds, q=1, num_restarts=5, raw_samples=20,
        )

        print(f"=== candidate: {candidate}")
        print(f"=== acq_value: {acq_value}")
        print(f"=== actual yield: {self.get_batch_yield_gpflow(candidate)}")

    def gpflow_opt(self):
        # default
        x = [8, 15, 30, 75, 150, 30, 37, 43, 47, 51, 57, 61, 65, 72, 76, 80, 84, 90, 116, 90, 80,
             22, 30, 35, 34, 33, 32, 31, 30, 29, 23,
             30, 42, 55, 60, 75, 65, 60,
             0.6, 0.7, 0.8, 0.9, 1.1, 1, 0.9, 0.9,
             0, 4000, 0, 4000, 0, 4000, 0, 4000, 0, 4000, 0, 4000, 0, 4000, 0, 4000, 0, 4000, 0, 0,
             0, 500, 100, 0, 400, 150, 250, 0, 100]
        lower = [ele * 0.9 if ele != 0 else ele for ele in x]
        upper = [ele * 1.1 if ele != 0 else 1 for ele in x]
        domain = np.sum([ContinuousParameter('x{0}'.format(i), l, u) for i, l, u in zip(range(len(x)), lower, upper)])
        design = RandomDesign(10, domain)
        X = design.generate()
        print(f"==== X: {X}")
        print(f"==== X: {len(X)}")

        domain1 = np.sum([ContinuousParameter('y{0}'.format(i), l, u) for i, l, u in zip(range(len(x)), x, x)])
        x0 = RandomDesign(1, domain1)

        Y = self.get_batch_yield_gpflow(X)

        # initializing a standard BO model, Gaussian Process Regression with
        # Matern52 ARD Kernel
        model = gpflow.gpr.GPR(X, Y, gpflow.kernels.Matern52(domain.size, ARD=True))
        alpha = ExpectedImprovement(model)

        # Now we must specify an optimization algorithm to optimize the acquisition
        # function, each iteration.
        acqopt = SciPyOptimizer(domain)

        # Now create the Bayesian Optimizer
        optimizer = BayesianOptimizer(domain, alpha, optimizer=acqopt, initial=x0, verbose=True)
        # with optimizer.silent():
        r = optimizer.optimize(self.get_batch_yield_gpflow, n_iter=10)
        print(r)

    def benchmark(self, total_calls, n_calls, n_random_starts, manup_scale):
        # default
        x = [8, 15, 30, 75, 150, 30, 37, 43, 47, 51, 57, 61, 65, 72, 76, 80, 84, 90, 116, 90, 80,
             22, 30, 35, 34, 33, 32, 31, 30, 29, 23,
             30, 42, 55, 60, 75, 65, 60,
             0.6, 0.7, 0.8, 0.9, 1.1, 1, 0.9, 0.9,
             0, 4000, 0, 4000, 0, 4000, 0, 4000, 0, 4000, 0, 4000, 0, 4000, 0, 4000, 0, 4000, 0, 0,
             0, 500, 100, 0, 400, 150, 250, 0, 100]

        x0 = [8, 16, 27, 70, 145, 32, 34, 38, 45, 53, 57, 55, 61, 72, 81, 86, 76, 84, 119, 99,
              87, 21, 29, 36, 30, 35, 34, 34, 31, 27, 25,
              31, 46, 49, 66, 81, 67, 66,
              0.6006477795120947, 0.662649715237619, 0.8400327687015882, 0.9184041217051334, 1.1103892968816163, 1.0352994970737246, 0.9182184239159428, 0.9006338441224637,
              0, 4155, 1, 3948, 1, 4368, 0, 3816, 1, 4036, 1, 4253, 0, 3600, 1, 3704, 1, 4307, 1, 1,
              1, 515, 91, 0, 392, 163, 235, 0, 102]

        num_iter = 0

        space = []
        yields_summary = []
        recipe_summary = []

        while total_calls > 0:
            print(f"=== === running iter @ {num_iter} with seed {self.random_int}")

            if num_iter == 0:
                recipe_Fs_sp, recipe_Foil_sp, recipe_Fg_sp, \
                recipe_pres_sp, recipe_discharge_sp, recipe_water_sp = self.split(x)

                recipe_Fs_sp_opt, recipe_Foil_sp_opt, recipe_Fg_sp_opt, \
                recipe_pres_sp_opt, recipe_discharge_sp_opt, recipe_water_sp_opt = self.split(x0)

                for Fs, Fs_opt in zip(recipe_Fs_sp, recipe_Fs_sp_opt):
                    lower_bound, upper_bound = self.get_bound(Fs, Fs_opt, manup_scale)
                    space.append(Integer(int(lower_bound), int(upper_bound)))

                for Foil, Foil_opt in zip(recipe_Foil_sp, recipe_Foil_sp_opt):
                    lower_bound, upper_bound = self.get_bound(Foil, Foil_opt, manup_scale)
                    space.append(Integer(int(lower_bound), int(upper_bound)))

                for Fg, Fg_opt in zip(recipe_Fg_sp, recipe_Fg_sp_opt):
                    lower_bound, upper_bound = self.get_bound(Fg, Fg_opt, manup_scale)
                    space.append(Integer(int(lower_bound), int(upper_bound)))

                for pres, pres_opt in zip(recipe_pres_sp, recipe_pres_sp_opt):
                    lower_bound, upper_bound = self.get_bound(pres, pres_opt, manup_scale)
                    space.append(Real(lower_bound, upper_bound))

                for discharge, discharge_opt in zip(recipe_discharge_sp, recipe_discharge_sp_opt):
                    if discharge != 0:
                        lower_bound, upper_bound = self.get_bound(discharge, discharge_opt, manup_scale)
                        space.append(Integer(int(lower_bound), int(upper_bound)))
                    else:
                        space.append(Integer(0, 1))

                for water, water_opt in zip(recipe_water_sp, recipe_water_sp_opt):
                    if water != 0:
                        lower_bound, upper_bound = self.get_bound(water, water_opt, manup_scale)
                        space.append(Integer(int(lower_bound), int(upper_bound)))
                    else:
                        space.append(Integer(0, 1))

            # if num_iter == 0:
            #     res_gp = gp_minimize(self.get_batch_yield,
            #                          space,
            #                          n_calls=n_calls,
            #                          n_random_starts=n_random_starts,
            #                          random_state=np.random.randint(1000),
            #                          n_jobs=-1,
            #                          verbose=True)
            # else:
            #     res_gp = gp_minimize(self.get_batch_yield,
            #                          space,
            #                          x0=x0,
            #                          n_calls=n_calls,
            #                          n_random_starts=n_random_starts,
            #                          random_state=np.random.randint(1000),
            #                          n_jobs=-1,
            #                          verbose=True)

            res_gp = gp_minimize(self.get_batch_yield,
                                 space,
                                 x0=x0,
                                 n_calls=n_calls,
                                 n_random_starts=n_random_starts,
                                 random_state=np.random.randint(1000),
                                 n_jobs=-1)

            num_iter += 1
            total_calls -= n_calls
            x0 = res_gp.x
            yields_summary.extend(res_gp.func_vals.tolist())
            recipe_summary.extend(res_gp.x_iters)

        return yields_summary, recipe_summary


for seed in range(1, 101):
    recipe_builder = RecipeBuilder(random_int=seed)
    yields, recipes = recipe_builder.benchmark(total_calls=100, n_calls=100, n_random_starts=4, manup_scale=0.1)
    print(len(yields))
    print(len(recipes))
    print(yields)

    # import matplotlib.pyplot as plt
    #
    # new_yields = [-ele for ele in yields]
    #
    # import numpy as np
    #
    # print(np.mean(new_yields))
    # print(np.std(new_yields))
    # print(np.min(new_yields))
    # print(np.max(new_yields))
    #
    # plt.plot(new_yields)
    # plt.show()

    import pickle

    with open(f'100_4_seed_{seed}_yield', 'wb') as fp:
        pickle.dump(yields, fp)

    with open(f'100_4_seed_{seed}_recipe', 'wb') as fp:
        pickle.dump(recipes, fp)

# import pickle
#
# with open('1000_100_1_seed_1_yield', 'rb') as fp:
#     yields = pickle.load(fp)
#
# with open('1000_100_1_seed_1_recipe', 'rb') as fp:
#     recipe = pickle.load(fp)
#
# new_yields = [-ele for ele in yields]
# # print(max(new_yields))
# # print(new_yields.index((max(new_yields))))
# # print(recipe[new_yields.index((max(new_yields)))])
# #
# # top_1_recipe = [8, 16, 27, 70, 145, 32, 34, 38, 45, 53, 57, 55, 61, 72, 81, 86, 76, 84, 119, 99, 87, 21, 29, 36, 30, 35,
# #                 34, 34, 31, 27, 25, 31, 46, 49, 66, 81, 67, 66, 0.6006477795120947, 0.662649715237619,
# #                 0.8400327687015882, 0.9184041217051334, 1.1103892968816163, 1.0352994970737246, 0.9182184239159428,
# #                 0.9006338441224637, 0, 4155, 1, 3948, 1, 4368, 0, 3816, 1, 4036, 1, 4253, 0, 3600, 1, 3704, 1, 4307, 1,
# #                 1, 1, 515, 91, 0, 392, 163, 235, 0, 102]
#
# import matplotlib.pyplot as plt
# plt.plot(new_yields)
# plt.show()
# import numpy as np
# print(np.mean(new_yields))
# print(np.std(new_yields))
# print(np.min(new_yields))
# print(np.max(new_yields))

# recipe_builder.botorch()
# recipe_builder.gpyopt()
