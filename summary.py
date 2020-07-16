import time
from pensimpy.pensim_classes.Recipe import Recipe
from pensimpy.env_setup.peni_env_setup import PenSimEnv
from pensimpy.helper.show_params import show_params
import numpy as np

# ===============paper data===========================
data_paper = [3062600.0, 3723600.0, 3991800.0, 3361700.0, 3785200.0, 3842700.0, 3413300.0, 3572900.0, 3751400.0,
              3307600.0, 2830800.0, 3277100.0, 3548800.0, 2981600.0, 3099500.0, 3415400.0, 3812000.0, 3588600.0,
              3825600.0, 3566200.0, 3186200.0, 3608100.0, 3837100.0, 3488100.0, 4044300.0, 3320800.0, 3470600.0,
              3059300.0, 3291400.0, 3464200.0]

data_paper = [ele / 1e3 for ele in data_paper]

print(f"=== data_paper: {data_paper}")

# ==================pensimpy==============================
pensimpy_data = []
for _ in range(30):
    t = time.time()
    # Random_seed_ref from 0 to 1000
    env = PenSimEnv(random_seed_ref=np.random.randint(1000))
    done = False
    observation, batch_data = env.reset()

    # default
    x = [8, 15, 30, 75, 150, 30, 37, 43, 47, 51, 57, 61, 65, 72, 76, 80, 84, 90, 116, 90, 80,
         22, 30, 35, 34, 33, 32, 31, 30, 29, 23,
         30, 42, 55, 60, 75, 65, 60,
         0.6, 0.7, 0.8, 0.9, 1.1, 1, 0.9, 0.9,
         0, 4000, 0, 4000, 0, 4000, 0, 4000, 0, 4000, 0, 4000, 0, 4000, 0, 4000, 0, 4000, 0, 0,
         0, 500, 100, 0, 400, 150, 250, 0, 100]
    # factor = 50
    # x = [ele * (1 + np.random.randint(-factor, factor) / 100) for ele in x]

    recipe = Recipe(x)

    time_stamp, batch_yield, yield_pre = 0, 0, 0
    while not done:
        # time is from 1 to 1150
        time_stamp += 1

        # Get action from recipe agent based on time
        Fs, Foil, Fg, pressure, Fremoved, Fw, Fpaa = recipe.run(time_stamp)

        # Run and get the reward
        # observation is a class which contains all the variables, e.g. observation.Fs.y[k], observation.Fs.t[k]
        # are the Fs value and corresponding time at k
        observation, batch_data, reward, done = env.step(time_stamp,
                                                         batch_data,
                                                         Fs, Foil, Fg, pressure, Fremoved, Fw, Fpaa)
        batch_yield += reward

    print(f"=== cost: {int(time.time() - t)} s")
    print(f"=== batch_yield: {batch_yield}")
    pensimpy_data.append(batch_yield)

print(f"=== pensimpy_data: {pensimpy_data}")

# pensimpy_data = [3263.1018173952652, 3487.995601204536, 2964.3501455702744, 3805.0520151736246, 3073.753601005814,
#                  3750.8104335453586, 3851.5508068977224, 3129.6648078192247, 3416.322321456596, 3614.942120486081,
#                  3699.6835146913168, 3490.5258784310327, 2972.979055299062, 3387.492201578702, 3344.873472768147,
#                  3203.894033269676, 3076.506170834993, 3295.518612141186, 3222.6472928511616, 3517.3963878367863,
#                  3849.9136617366494, 3689.2853899502484, 3014.2274724165327, 3633.4086530181326, 3443.537293804536,
#                  3496.3453349651495, 3472.0668779865405, 3591.045856909633, 3874.7761781227223, 3334.4317077177293,
#                  3536.078070053139, 3521.031749833082, 3138.084342552033, 3553.902969921243, 3623.7028199916963,
#                  3629.2768656708104, 3073.9251504757567, 3710.8553932290156, 3574.2353965188563, 3370.3574751145443,
#                  3500.808347312636, 2987.0095094099406, 3462.150374611891, 3219.3358919703323, 3420.4199847678374,
#                  3296.8046776338547, 3340.5397735844685, 3364.1565296818703, 3259.5098533676055, 3629.2326918539497,
#                  3073.9251504757567, 3710.8553932290156, 3574.2353965188563, 3370.3574751145443, 3500.808347312636,
#                  2987.0095094099406, 3462.150374611891, 3219.3358919703323, 3420.4199847678374, 3296.8046776338547,
#                  3340.5397735844685, 3364.1565296818703, 3259.5098533676055, 3629.2326918539497, 3073.9251504757567,
#                  3710.8553932290156, 3574.2353965188563, 3370.3574751145443, 3500.808347312636, 2987.0095094099406,
#                  3462.150374611891, 3219.3358919703323, 3420.4199847678374, 3296.8046776338547, 3340.5397735844685,
#                  3364.1565296818703, 3259.5098533676055, 3629.2326918539497, 3073.9251504757567, 3710.8553932290156,
#                  3574.2353965188563, 3370.3574751145443, 3500.808347312636, 2987.0095094099406, 3462.150374611891,
#                  3219.3358919703323, 3420.4199847678374, 3296.8046776338547, 3340.5397735844685, 3364.1565296818703,
#                  3259.5098533676055, 3629.2326918539497, 3073.9251504757567, 3710.8553932290156, 3574.2353965188563,
#                  3370.3574751145443, 3500.808347312636, 2987.0095094099406, 3462.150374611891, 3219.3358919703323]

import seaborn as sns
import matplotlib.pyplot as plt

sns.distplot(data_paper, hist=False, kde=True,
             kde_kws={'linewidth': 3, 'shade': True},
             label='Batch Records')

sns.distplot(pensimpy_data, hist=False, kde=True,
             kde_kws={'linewidth': 3, 'shade': True},
             label='Quartic Implementation')

plt.legend(prop={'size': 8})
plt.yticks([])
plt.title('Penicillin Yield Comparison')
plt.xlabel('Penicillin Yield (kg)')
plt.ylabel('Density')
plt.show()
