import numpy as np
import math
from helper.smooth_py import smooth_py


def raman_sim(k, x, h, T, reference_Spectra_2200):
    # Building history of Raman Spectra
    Wavenumber_max = 2200
    Intensity_shift1 = np.ones((Wavenumber_max, 1), dtype=int)

    for j in range(1, Wavenumber_max + 1):
        b = j / (Wavenumber_max * 0.5)
        Intensity_shift1[j - 1, 0] = math.exp(b) - 0.5

    # todo check here, get rid of a list
    New_Spectra = np.ones((Wavenumber_max, 1), dtype=int)

    a = -0.00178143846614472 * 0.1
    b = 1.05644816081515
    c = -0.0681439987249108 * 0.1
    d = -0.02
    Product_S = x.P.y[k - 1] / 40
    Biomass_S = x.X.y[k - 1] / 40
    Viscosity_S = x.Viscosity.y[k - 1] / 100
    Time_S = k / (T / h)
    Intensity_increase1 = a * Biomass_S + b * Product_S + c * Viscosity_S + d * Time_S
    scaling_factor = 370000
    Gluc_increase = 800000 * 3 / 1400

    PAA_increase = 1700000 / 1000
    Prod_increase = 100000

    # Loading in the reference Raman Spectral file
    x.Raman_Spec.Wavelength = reference_Spectra_2200[0:Wavenumber_max, 0].astype('int').tolist()
    reference_spectra = reference_Spectra_2200[0:Wavenumber_max, 1].astype('float').tolist()
    New_Spectra = Intensity_increase1 * scaling_factor * Intensity_shift1 + np.array([reference_spectra]).T
    x.Raman_Spec.Intensity[:, k - 1] = np.squeeze(New_Spectra).tolist()

    random_noise = np.ones((Wavenumber_max + 1, 1), dtype=int)
    random_noise_summed = np.ones((Wavenumber_max, 1), dtype=int)
    New_Spectra_noise = np.ones((Wavenumber_max, 1), dtype=int)

    random_number = np.random.randint(1, 4, size=(Wavenumber_max, 1))

    for i in range(Wavenumber_max):
        noise_factor = 50
        if random_number[i][0] == 1:
            random_noise[i] = 0
        elif random_number[i][0] == 2:
            random_noise[i] = noise_factor
        else:
            random_noise[i] = -noise_factor

    for i in range(Wavenumber_max):
        random_noise_summed[i] = sum(np.squeeze(random_noise).tolist()[0:i])

    random_noise_summed_smooth = smooth_py(np.squeeze(random_noise_summed).tolist(), 25)

    New_Spectra_noise = New_Spectra + 10 * np.array([random_noise_summed_smooth]).T

    x.Raman_Spec.Intensity[:, k - 1] = np.squeeze(New_Spectra_noise).tolist()

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
        Glucose_raw_peaks_G_peaka[xx + peaka - 1] = (peaka_std_dev * np.sqrt(2 * math.pi)) ** -1 * math.exp(
            -0.5 * ((xx - mean) / peaka_std_dev) ** 2)

    # Peak B
    peakb = 639
    peakb_width = 20
    peakb_lenght = peakb_width * 2
    peakb_std_dev = peakb_width / 2
    mean = 0
    for xx in range(-peakb_lenght, peakb_lenght + 1):
        Glucose_raw_peaks_G_peakb[xx + peakb - 1] = (peakb_std_dev * np.sqrt(2 * math.pi)) ** -1 * math.exp(
            -0.5 * ((xx - mean) / peakb_std_dev) ** 2) / 4.3

    # Peak C
    peakc = 1053
    peakc_width = 100
    peakc_lenght = peakc_width * 2
    peakc_std_dev = peakc_width / 2
    mean = 0
    for xx in range(-peakc_lenght, peakc_lenght + 1):
        Glucose_raw_peaks_G_peakc[xx + peakc - 1] = (peakc_std_dev * np.sqrt(2 * math.pi)) ** -1 * math.exp(
            -0.5 * ((xx - mean) / peakc_std_dev) ** 2)

    # PAA  peaks
    # Peak A
    peaka = 419
    peaka_width = 60
    peaka_lenght = peaka_width * 2
    peaka_std_dev = peaka_width / 2
    mean = 0
    for xx in range(-peaka_lenght, peaka_lenght + 1):
        PAA_raw_peaks_G_peaka[xx + peaka - 1] = (peaka_std_dev * np.sqrt(2 * math.pi)) ** -1 * math.exp(
            -0.5 * ((xx - mean) / peaka_std_dev) ** 2)

    # Peak B
    peakb = 839
    peakb_width = 15
    peakb_lenght = peakb_width * 2
    peakb_std_dev = peakb_width / 2
    mean = 0
    for xx in range(-peakb_lenght, peakb_lenght + 1):
        PAA_raw_peaks_G_peakb[xx + peakb - 1] = ((peakb_std_dev * np.sqrt(2 * math.pi)) ** -1 * math.exp(
            -0.5 * ((xx - mean) / peakb_std_dev) ** 2) / 4.3)

    # Adding in  Peak aPen G Peak
    peakPa = 800
    peakPa_width = 30
    peakPa_lenght = peakPa_width * 4
    peakPa_std_dev = peakPa_width / 2
    mean = 0
    for xx in range(-peakPa_lenght, peakPa_lenght + 1):
        Product_raw_peaka[xx + peakPa - 1] = (peakPa_std_dev * np.sqrt(2 * math.pi)) ** -1 * math.exp(
            -0.5 * ((xx - mean) / peakPa_std_dev) ** 2)

    # Adding in  Peak b for Pen G Peak
    peakPb = 1200
    peakPb_width = 30
    peakPb_lenght = peakPb_width * 30
    peakPb_std_dev = peakPb_width / 2
    mean = 0
    for xx in range(-peakPb_lenght, peakPb_lenght + 1):
        Product_raw_peakb[xx + peakPb - 1] = (peakPb_std_dev * np.sqrt(2 * math.pi)) ** -1 * math.exp(
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

    x.Raman_Spec.Intensity[:, k - 1] = np.squeeze(New_Spectra_noise + term1 + term2 + term3).tolist()

    return x
