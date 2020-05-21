import numpy as np

H = 0.2  # [h]

reference_Spectra_2200 = np.genfromtxt('./spectra_data/reference_Specra.txt', dtype='str')
raman_wavenumber = reference_Spectra_2200[0:2200, 0].astype('int').tolist()
raman_spectra = reference_Spectra_2200[0:2200, 1].astype('float').tolist()

model_data = np.loadtxt("./Matlab_model/PAA_PLS_model.txt", comments="#", delimiter=",", unpack=False)
