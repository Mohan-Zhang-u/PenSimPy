import numpy as np
import math
from helper.smooth_py import smooth_py


def raman_sim(k, x, h, T, raman_spectra):
    # Building history of Raman Spectra
    Wavenumber_max = 2200
    Intensity_shift1 = np.ones((Wavenumber_max, 1), dtype=int)

    Intensity_shift1[:, 0] = [math.exp((j + 1) / 1100) - 0.5 for j in range(Wavenumber_max)]

    a = -0.000178143846614472
    b = 1.05644816081515
    c = -0.00681439987249108
    d = -0.02
    Product_S = x.P.y[k - 1] / 40
    Biomass_S = x.X.y[k - 1] / 40
    Viscosity_S = x.Viscosity.y[k - 1] / 100
    Time_S = k / (T / h)
    Intensity_increase1 = a * Biomass_S + b * Product_S + c * Viscosity_S + d * Time_S
    scaling_factor = 370000
    Gluc_increase = 1714.2857142857142
    PAA_increase = 1700
    Prod_increase = 100000

    # Loading in the reference Raman Spectral file
    New_Spectra = Intensity_increase1 * scaling_factor * Intensity_shift1 + np.array([raman_spectra]).T
    x.Raman_Spec.Intensity[k - 1, :] = np.squeeze(New_Spectra).tolist()

    random_noise = [50] * Wavenumber_max
    random_number = np.random.randint(-1, 2, size=(Wavenumber_max, 1))
    random_noise = np.multiply(random_noise, random_number.T)[0]

    random_noise_summed = np.cumsum(random_noise)
    random_noise_summed_smooth = smooth_py(random_noise_summed, 25)

    New_Spectra_noise = New_Spectra + 10 * np.array([random_noise_summed_smooth]).T

    x.Raman_Spec.Intensity[k - 1, :] = np.squeeze(New_Spectra_noise).tolist()

    # Aim 3. Creating the bell curve response for Glucose
    Glucose_raw_peaks_G_peaka = np.zeros((Wavenumber_max, 1), dtype=float)
    Glucose_raw_peaks_G_peakb = np.zeros((Wavenumber_max, 1), dtype=float)
    Glucose_raw_peaks_G_peakc = np.zeros((Wavenumber_max, 1), dtype=float)
    PAA_raw_peaks_G_peaka = np.zeros((Wavenumber_max, 1), dtype=float)
    PAA_raw_peaks_G_peakb = np.zeros((Wavenumber_max, 1), dtype=float)
    Product_raw_peaka = np.zeros((Wavenumber_max, 1), dtype=float)
    Product_raw_peakb = np.zeros((Wavenumber_max, 1), dtype=float)

    # Glucose peaks
    # Peak A
    Glucose_raw_peaks_G_peaka[78: 359, 0] = [0.011398350868612364 * math.exp(-0.0004081632653061224 * j * j) for j in range(-140, 141)]

    # Peak B
    Glucose_raw_peaks_G_peakb[598: 679, 0] = [0.009277727451196111 * math.exp(-0.005 * j * j) for j in range(-40, 41)]

    # Peak C
    Glucose_raw_peaks_G_peakc[852: 1253, 0] = [0.007978845608028654 * math.exp(-0.0002 * j * j) for j in range(-200, 201)]

    # PAA  peaks
    # Peak A
    PAA_raw_peaks_G_peaka[298: 539, 0] = [0.01329807601338109 * math.exp(-0.0005555555555555556 * j * j) for j in range(-120, 121)]

    # Peak B
    PAA_raw_peaks_G_peakb[808: 869, 0] = [0.01237030326826148 * math.exp(-0.008888888888888889 * j * j) for j in range(-30, 31)]

    # Adding in  Peak aPen G Peak
    Product_raw_peaka[679: 920, 0] = [0.02659615202676218 * math.exp(-0.0022222222222222222 * j * j) for j in range(-120, 121)]

    # Adding in  Peak b for Pen G Peak
    Product_raw_peakb[299: 2100, 0] = [0.02659615202676218 * math.exp(-0.0022222222222222222 * j * j) for j in range(-900, 901)]

    total_peaks_G = Glucose_raw_peaks_G_peaka + Glucose_raw_peaks_G_peakb + Glucose_raw_peaks_G_peakc
    total_peaks_PAA = PAA_raw_peaks_G_peaka + PAA_raw_peaks_G_peakb
    total_peaks_P = Product_raw_peakb + Product_raw_peaka
    K_G = 0.005

    Substrate_raman = x.S.y[k - 1]
    PAA_raman = x.PAA.y[k - 1]

    term1 = total_peaks_G * Gluc_increase * Substrate_raman / (K_G + Substrate_raman)
    term2 = total_peaks_PAA * PAA_increase * PAA_raman
    term3 = total_peaks_P * Prod_increase * x.P.y[k - 1]

    x.Raman_Spec.Intensity[k - 1, :] = np.squeeze(New_Spectra_noise + term1 + term2 + term3).tolist()

    return x
