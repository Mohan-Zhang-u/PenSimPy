import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

base_line = [3998.270799492237, 3450.9893438147974, 3522.6999789430474, 4423.041361284175, 3879.926647339472,
             3871.4665640840867, 3589.5718541750193, 3870.432673060489, 4007.0675578589603, 3789.319473360788,
             3757.0932063539676, 3585.265953860275, 3617.8184072727695, 3212.373058629767, 3379.908142292542,
             3413.1884778519725, 3767.916953994205, 3410.985681408933, 3425.951819955421, 3803.7560073314726,
             3521.862759073743, 3592.4959683418065, 2968.8767473585963, 3610.468235779067, 3995.631547714517,
             3657.9009660294187, 3384.1667849345986, 3479.2703710610913, 3879.773666315146, 3807.454430519059,
             3397.6406445312746, 4268.848779351172, 3494.0160639698097, 3610.416953919331, 3505.931843670571,
             3447.85571915065, 4033.0975531557347, 4162.850052354707, 3555.002399147275, 4186.5513507124715,
             3647.4975712617475, 3980.2685231333426, 3663.3880554382267, 4004.211024661507, 3516.9446555001955,
             3679.818610299045, 3357.806494853542, 4488.546778324133, 3464.8167962386246, 3680.7570299814556,
             3836.878107120413, 3718.222750730553, 3622.2270218215835, 3498.315041480005, 3867.9691655150723,
             3712.4499442058027, 3268.4322755752264, 3438.2403376359143, 4018.2933545327387, 3906.2653072857534,
             3740.8638361818485, 4031.245614514647, 3985.917347398899, 3458.7286965667727, 3386.2994094333417,
             4165.07831284846, 3452.22918720157, 3473.2654209188395, 3935.498714897459, 3960.069697172046,
             3550.7876091158705, 3673.013754933395, 3595.8834337110675, 4264.097111732126, 3929.030791605194,
             3216.0613640587762, 3737.978326188811, 3753.5888900481737, 3755.080730734, 4358.725143768074,
             3554.224666676063, 3765.888475802928, 3958.5819910447867, 3448.905807467216, 3468.3083735754353,
             3881.0264734169327, 4100.258944802328, 4453.089513575393, 3892.2378119524715, 4274.94474600546,
             3299.3297142361525, 3975.9298984833013, 3562.582256584628, 3697.9572555084796, 3616.030871857157,
             3756.8192210428224, 3690.797517255054, 3851.8232501508605, 4290.582919906925, 3556.9984522072536]

imps = []

for seed in range(1, 83):
    with open(f'100_4_seed_{seed}_yield', 'rb') as fp:
        yields = pickle.load(fp)
        yields = [-y for y in yields]
        # print(f"=== mean: {np.mean(yields)}")
        # print(f"=== std: {np.std(yields)}")
        # print(f"=== min: {np.min(yields)}")
        # print(f"=== max: {np.max(yields)}")
        imp = (np.mean(yields) - base_line[seed - 1]) / base_line[seed - 1]
        imps.append(imp)

# plt.plot(yields)
# plt.show()
# exit()

print(imps)
term1 = np.mean(np.mean(imps) + np.median(imps))
term2 = np.mean(np.std(imps) + (np.percentile(imps, 75) - np.percentile(imps, 25)) / 2)



# sns.distplot(imps, hist=True, label=f'{np.round(term1 * 100, 2)}% +- {np.round(term2, 2)}')

# sns.distplot(imps, hist=False, kde=True,
#              kde_kws={'linewidth': 3, 'shade': True},
#              label=f'{np.round(term1 * 100, 2)}% +- {np.round(term2, 2)}')

import pandas as pd
pd.Series(imps).hist(bins=83)

plt.legend(prop={'size': 8})
plt.yticks([])
plt.title('Penicillin Yield Improvements')
plt.xlabel('Improvement Percentage')
plt.ylabel('Density')
plt.show()

# with open('1000_opt_1000_baseline.pkl', 'rb') as fp:
#     yields = pickle.load(fp)
#
# print(yields[0]['yields'])
# imps = []
# for seed in range(1, 21):
#     with open('1000_opt_1000_baseline.pkl', 'rb') as fp:
#         yields = pickle.load(fp)
#         yields = yields[seed]['yields']
#         imp = (np.mean(yields) - 3997) / 3997
#         imps.append(imp)
#
# print(imps)

# new_yields = [-ele for ele in new_yields]
# print(len(new_yields))
#
# print(f"=== mean: {np.mean(new_yields)}")
# print(f"=== std: {np.std(new_yields)}")
# print(f"=== min: {np.min(new_yields)}")
# print(f"=== max: {np.max(new_yields)}")
#
# plt.plot(new_yields)
# plt.show()
