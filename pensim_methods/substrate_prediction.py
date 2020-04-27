from scipy.signal import savgol_filter
import numpy as np
from scipy.io import loadmat


def substrate_prediction(k, x):
    j = k - 1
    Raman_Spec_sg = savgol_filter(x.Raman_Spec.Intensity[:, j - 1].tolist(), 5, 2)
    Raman_Spec_sg_d = np.diff(Raman_Spec_sg)
    PAA_peaks_Spec = []
    PAA_peaks_Spec.extend(Raman_Spec_sg_d[349:500])
    PAA_peaks_Spec.extend(Raman_Spec_sg_d[799:860])
    No_LV = 4
    b = loadmat('./Matlab_model/PAA_PLS_model.mat')['b']
    x.PAA_pred.y[j - 1] = sum([x * y for x, y in zip(PAA_peaks_Spec, b[No_LV - 1, :].tolist())])
    if j > 20:
        x.PAA_pred.y[j - 1] = (x.PAA_pred.y[j - 2] + x.PAA_pred.y[j - 3] + x.PAA_pred.y[j - 1]) / 3
    return x
