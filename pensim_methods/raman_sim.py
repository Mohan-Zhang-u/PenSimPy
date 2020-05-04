import numpy as np
import math
from helper.smooth_py import smooth_py


def raman_sim(k, x, h, T, raman_spectra):
    # Building history of Raman Spectra
    Wavenumber_max = 2200
    Intensity_shift1 = np.ones((Wavenumber_max, 1), dtype=int)

    Intensity_shift1[:, 0] = [math.exp((j + 1) / 1100) - 0.5 for j in range(Wavenumber_max)]

    # # todo check here, get rid of a list
    # New_Spectra = np.ones((Wavenumber_max, 1), dtype=int)

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
    peaka = 219
    peaka_width = 70
    peaka_lenght = peaka_width * 2
    peaka_std_dev = peaka_width / 2
    mean = 0
    for xx in range(-peaka_lenght, peaka_lenght + 1):
        Glucose_raw_peaks_G_peaka[xx + peaka - 1] = (peaka_std_dev * (2 * math.pi) ** .5) ** -1 * math.exp(
            -0.5 * ((xx - mean) / peaka_std_dev) ** 2)

    # Peak B
    peakb = 639
    peakb_width = 20
    peakb_lenght = peakb_width * 2
    peakb_std_dev = peakb_width / 2
    mean = 0
    for xx in range(-peakb_lenght, peakb_lenght + 1):
        Glucose_raw_peaks_G_peakb[xx + peakb - 1] = (peakb_std_dev * (2 * math.pi) ** .5) ** -1 * math.exp(
            -0.5 * ((xx - mean) / peakb_std_dev) ** 2) / 4.3

    # Peak C
    peakc = 1053
    peakc_width = 100
    peakc_lenght = peakc_width * 2
    peakc_std_dev = peakc_width / 2
    mean = 0
    for xx in range(-peakc_lenght, peakc_lenght + 1):
        Glucose_raw_peaks_G_peakc[xx + peakc - 1] = (peakc_std_dev * (2 * math.pi) ** .5) ** -1 * math.exp(
            -0.5 * ((xx - mean) / peakc_std_dev) ** 2)

    # PAA  peaks
    # Peak A
    peaka = 419
    peaka_width = 60
    peaka_lenght = peaka_width * 2
    peaka_std_dev = peaka_width / 2
    mean = 0
    for xx in range(-peaka_lenght, peaka_lenght + 1):
        PAA_raw_peaks_G_peaka[xx + peaka - 1] = (peaka_std_dev * (2 * math.pi) ** .5) ** -1 * math.exp(
            -0.5 * ((xx - mean) / peaka_std_dev) ** 2)

    # Peak B
    peakb = 839
    peakb_width = 15
    peakb_lenght = peakb_width * 2
    peakb_std_dev = peakb_width / 2
    mean = 0
    for xx in range(-peakb_lenght, peakb_lenght + 1):
        PAA_raw_peaks_G_peakb[xx + peakb - 1] = ((peakb_std_dev * (2 * math.pi) ** .5) ** -1 * math.exp(
            -0.5 * ((xx - mean) / peakb_std_dev) ** 2) / 4.3)

    # Adding in  Peak aPen G Peak
    peakPa = 800
    peakPa_width = 30
    peakPa_lenght = peakPa_width * 4
    peakPa_std_dev = peakPa_width / 2
    mean = 0
    for xx in range(-peakPa_lenght, peakPa_lenght + 1):
        Product_raw_peaka[xx + peakPa - 1] = (peakPa_std_dev * (2 * math.pi) ** .5) ** -1 * math.exp(
            -0.5 * ((xx - mean) / peakPa_std_dev) ** 2)

    # Adding in  Peak b for Pen G Peak
    peakPb = 1200
    peakPb_width = 30
    peakPb_lenght = peakPb_width * 30
    peakPb_std_dev = peakPb_width / 2
    mean = 0
    for xx in range(-peakPb_lenght, peakPb_lenght + 1):
        Product_raw_peakb[xx + peakPb - 1] = (peakPb_std_dev * (2 * math.pi) ** .5) ** -1 * math.exp(
            -0.5 * ((xx - mean) / peakPb_std_dev) ** 2)

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
