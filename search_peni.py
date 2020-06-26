from pensimpy.helper.get_recipe_trend import get_recipe_trend
from pensimpy.env_setup.peni_env_setup import PenSimEnv
import numpy as np
from random import random, seed
from skopt import gp_minimize
from skopt.space import Real, Integer
import math


class RecipeBuilder:
    """builds the recipe that can run with the env, given setpoints"""

    def __init__(self):
        self.Fs_trend = None
        self.Foil_trend = None
        self.Fg_trend = None
        self.pres_trend = None
        self.discharge_trend = None
        self.water_trend = None

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

    def run(self, t):
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

        env = PenSimEnv(random_seed_ref=123)
        done = False
        batch_data = env.reset()
        self.init_recipe(Fs_sp, Foil_sp, Fg_sp, pres_sp, discharge_sp, water_sp)
        time_stamp, batch_yield, yield_pre = 0, 0, 0
        while not done:
            time_stamp += 1
            Fs, Foil, Fg, Fpres, Fdischarge, Fw, Fpaa = self.run(time_stamp)
            observation, batch_data, reward, done = env.step(time_stamp,
                                                             batch_data,
                                                             Fs, Foil, Fg, Fpres, Fdischarge, Fw, Fpaa)
            batch_yield += reward
        return -batch_yield

    def benchmark(self, total_calls, n_calls):
        x = [8, 15, 30, 75, 150, 30, 37, 43, 47, 51, 57, 61, 65, 72, 76, 80, 84, 90, 116, 90, 80,
             22, 30, 35, 34, 33, 32, 31, 30, 29, 23,
             30, 42, 55, 60, 75, 65, 60,
             0.6, 0.7, 0.8, 0.9, 1.1, 1, 0.9, 0.9,
             0, 4000, 0, 4000, 0, 4000, 0, 4000, 0, 4000, 0, 4000, 0, 4000, 0, 4000, 0, 4000, 0, 0,
             0, 500, 100, 0, 400, 150, 250, 0, 100]
        num_iter = 0

        space = []

        while total_calls > 0:
            print(f"=== running iter {num_iter}")
            print(f"=== x: {x}")
            manup_scale = 0.08 * math.exp(-0.27 * num_iter) + 0.02
            print(f"=== manup_scale: {manup_scale}")
            total_calls -= n_calls

            recipe_Fs_sp, recipe_Foil_sp, recipe_Fg_sp, \
            recipe_pres_sp, recipe_discharge_sp, recipe_water_sp = self.split(x)

            if num_iter == 0:
                for Fs in recipe_Fs_sp:
                    space.append(Integer(int(Fs - Fs * manup_scale), int(Fs + Fs * manup_scale)))

                for Foil in recipe_Foil_sp:
                    space.append(Integer(int(Foil - Foil * manup_scale), int(Foil + Foil * manup_scale)))

                for Fg in recipe_Fg_sp:
                    space.append(Integer(int(Fg - Fg * manup_scale), int(Fg + Fg * manup_scale)))

                for pres in recipe_pres_sp:
                    space.append(Real(pres - pres * manup_scale, pres + pres * manup_scale))

                for discharge in recipe_discharge_sp:
                    if discharge != 0:
                        space.append(
                            Integer(int(discharge - discharge * manup_scale), int(discharge + discharge * manup_scale)))
                    else:
                        space.append(Integer(0, 1))

                for water in recipe_water_sp:
                    if water != 0:
                        space.append(Integer(int(water - water * manup_scale), int(water + water * manup_scale)))
                    else:
                        space.append(Integer(0, 1))

            num_iter += 1
            print(f"=== space degree: {len(space)}")

            res_gp = gp_minimize(self.get_batch_yield,
                                 space,
                                 x0=x,
                                 n_calls=n_calls,
                                 n_random_starts=1,
                                 random_state=123,
                                 n_jobs=-1)
            min_val = 0
            min_idx = -1
            for idx, val in enumerate(res_gp.func_vals):
                if val < min_val:
                    min_val = val
                    min_idx = idx

            print(res_gp.func_vals)
            print(f"=== min val is {min_val} & at NO. {min_idx}")
            print(f"=== corresponding x is: {res_gp.x}")
            print(f"=== mean is {np.mean(res_gp.func_vals)}")
            x = res_gp.x


recipe_builder = RecipeBuilder()
recipe_builder.benchmark(total_calls=100, n_calls=20)
