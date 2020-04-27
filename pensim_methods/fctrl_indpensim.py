import math
from helper.PIDSimple3 import PIDSimple3
import numpy as np
from scipy.interpolate import interp1d
from pensim_classes.U import U


def fctrl_indpensim(x, xd, k, h, T, ctrl_flags, Recipe_Fs_sp):
    """
    Control strategies: Sequential batch control and PID control
    :param x:
    :param xd:
    :param k:
    :param h:
    :param T:
    :param ctrl_flags:
    :return:
    """
    # pH controller
    u = U()
    pH_sensor_error = 0
    if ctrl_flags.Faults == 8:
        pH_sensor_error = 0.1
        Ramp_function = [[0, 0],
                         [200, 0],
                         [800, pH_sensor_error],
                         [1750, pH_sensor_error]]
        Ramp_function = np.array(Ramp_function)
        tInterp = [x for x in range(1, 1751)]
        f = interp1d(Ramp_function[:, 0], Ramp_function[:, 1], kind='linear', fill_value='extrapolate')
        Ramp_function_interp = f(tInterp)
        pH_sensor_error = Ramp_function_interp[k - 1]
        u.Fault_ref = 1

    # builds the error history. Samples 1 and 2 are calculated separately
    # because there is only instance of the error available
    pH_sp = ctrl_flags.pH_sp
    if k == 1 or k == 2:
        ph_err = pH_sp - (-math.log(x.pH.y[0]) / math.log(10)) + pH_sensor_error
        ph_err1 = pH_sp - (-math.log(x.pH.y[0]) / math.log(10)) + pH_sensor_error
    else:
        ph_err = pH_sp - (-math.log(x.pH.y[k - 2]) / math.log(10)) + pH_sensor_error
        ph_err1 = - (-math.log(x.pH.y[k - 3]) / math.log(10)) + pH_sensor_error

    # builds the pH history of the current and previous two samples
    if k == 1 or k == 2:
        ph = -math.log(x.pH.y[0]) / math.log(10)
        ph1 = -math.log(x.pH.y[0]) / math.log(10)
        ph2 = -math.log(x.pH.y[0]) / math.log(10)
    elif k == 3:
        ph = -math.log(x.pH.y[1]) / math.log(10)
        ph1 = -math.log(x.pH.y[0]) / math.log(10)
        ph2 = -math.log(x.pH.y[0]) / math.log(10)
    else:
        ph = -math.log(x.pH.y[k - 2]) / math.log(10)
        ph1 = -math.log(x.pH.y[k - 3]) / math.log(10)
        ph2 = -math.log(x.pH.y[k - 4]) / math.log(10)

    # pH has decreased under 0.05 from set-point, add some base solution
    if ph_err >= -0.05:
        ph_on_off = 1
        if k == 1:
            Fb = PIDSimple3(x.Fb.y[0], ph_err, ph_err1, ph, ph1, ph2, 0, 225, 8e-2, 4.0000e-05, 8, h)
        else:
            Fb = PIDSimple3(x.Fb.y[k - 2], ph_err, ph_err1, ph, ph1, ph2, 0, 225, 8e-2, 4.0000e-05, 8, h)
        Fa = 0
    elif ph_err <= -0.05:
        ph_on_off = 1
        if k == 1:
            Fa = PIDSimple3(x.Fa.y[0], ph_err, ph_err1, ph, ph1, ph2, 0, 225, 8e-2, 12.5, 0.125, h)
            Fb = 0
        else:
            Fa = PIDSimple3(x.Fa.y[k - 2], ph_err, ph_err1, ph, ph1, ph2, 0, 225, 8e-2, 12.5, 0.125, h)
            Fb = x.Fb.y[k - 2] * 0.5
    else:
        ph_on_off = 0
        Fb = 0
        Fa = 0

    # Temperature controller
    T_sensor_error = 0
    if ctrl_flags.Faults == 7:
        T_sensor_error = 0.4
        Ramp_function = [[0, 0],
                         [200, 0],
                         [800, T_sensor_error],
                         [1750, T_sensor_error]]
        Ramp_function = np.array(Ramp_function)
        tInterp = [x for x in range(1, 1751)]
        f = interp1d(Ramp_function[:, 0], Ramp_function[:, 1], kind='linear', fill_value='extrapolate')
        Ramp_function_interp = f(tInterp)
        T_sensor_error = Ramp_function_interp[k - 1]
        u.Fault_ref = 1

    # builds the error history.  Samples 1 and 2 are calculated separately
    # because there is only instance of the error available
    T_sp = ctrl_flags.T_sp
    if k == 1 or k == 2:
        temp_err = T_sp - x.T.y[0] + T_sensor_error
        temp_err1 = T_sp - x.T.y[0] + T_sensor_error
    else:
        temp_err = T_sp - x.T.y[k - 2] + T_sensor_error
        temp_err1 = T_sp - x.T.y[k - 3] + T_sensor_error

    # builds the temperature history of current and previous two samples.
    if k == 1 or k == 2:
        temp = x.T.y[0]
        temp1 = x.T.y[0]
        temp2 = x.T.y[0]
    elif k == 3:
        temp = x.T.y[1]
        temp1 = x.T.y[0]
        temp2 = x.T.y[0]
    else:
        temp = x.T.y[k - 2]
        temp1 = x.T.y[k - 3]
        temp2 = x.T.y[k - 4]

    # Threshold for heating. Heating is activated only if the temperature drop
    # is more than 1 degree celsius
    if temp_err <= 0.05:
        temp_on_off = 0
        if k == 1:
            Fc = PIDSimple3(x.Fc.y[0], temp_err, temp_err1, temp, temp1, temp2, 0, 1.5e3, -300, 1.6, 0.005, h)
            Fh = 0
        else:
            Fc = PIDSimple3(x.Fc.y[k - 2], temp_err, temp_err1, temp, temp1, temp2, 0, 1.5e3, -300, 1.6, 0.005, h)
            Fh = x.Fh.y[k - 2] * 0.1
    else:
        temp_on_off = 1
        if k == 1:
            Fh = PIDSimple3(x.Fc.y[0], temp_err, temp_err1, temp, temp1, temp2, 0, 1.5e3, 50, 0.050, 1, h)
            Fc = 0
        else:
            Fh = PIDSimple3(x.Fc.y[k - 2], temp_err, temp_err1, temp, temp1, temp2, 0, 1.5e3, 50, 0.050, 1, h)
            Fc = x.Fc.y[k - 2] * 0.3
    Fc = 1e-4 if Fc < 1e-4 else Fc
    Fh = 1e-4 if Fh < 1e-4 else Fh

    # Sequential Batch control strategy
    # If Sequential Batch Control (SBC) = 1, operator controlled
    if ctrl_flags.SBC == 1:
        Foil = xd.Foil.y[k - 1]
        F_discharge = xd.F_discharge_cal.y[k - 1]
        pressure = xd.pressure.y[k - 1]
        Fpaa = xd.Fpaa.y[k - 1]
        Fw = xd.Fw.y[k - 1]
        viscosity = xd.viscosity.y[k - 1]
        Fg = xd.Fg.y[k - 1]
        Fs = xd.Fs.y[k - 1]

    # SBC - Fs
    if ctrl_flags.SBC == 0:
        viscosity = 4
        Recipe_Fs = [15, 60, 80, 100, 120, 140, 160, 180, 200, 220, 240,
                     260, 280, 300, 320, 340, 360, 380, 400, 800, 1750]
        # Recipe_Fs_sp = [8, 15, 30, 75, 150, 30, 37, 43, 47, 51, 57, 61, 65, 72, 76, 80, 84, 90, 116, 90, 80]

        for SQ in range(1, len(Recipe_Fs) + 1):
            if k <= Recipe_Fs[SQ - 1]:
                Fs = Recipe_Fs_sp[SQ - 1]
                break
            else:
                Fs = Recipe_Fs_sp[-1]
        if ctrl_flags.PRBS == 1:
            if k > 500 and np.remainder(k, 100) == 0:
                random_number = np.random.randint(1, 4)
                noise_factor = 15
                if random_number == 1:
                    random_noise = 0
                elif random_number == 2:
                    random_noise = noise_factor
                else:
                    random_noise = -noise_factor
                x.PRBS_noise_addition[k - 1] = random_noise
            Fs = x.Fs.y[k - 2] if k > 475 else Fs
            if k > 500 and np.remainder(k, 100) == 0:
                Fs = x.Fs.y[k - 2] + x.PRBS_noise_addition[-1]
        else:
            x.PRBS_noise_addition[k - 1] = 0

        # SBC - Foil
        Recipe_Foil = [20, 80, 280, 300, 320, 340, 360, 380, 400, 1750]
        Recipe_Foil_sp = [22, 30, 35, 34, 33, 32, 31, 30, 29, 23]
        for SQ in range(1, len(Recipe_Foil) + 1):
            if k <= Recipe_Foil[SQ - 1]:
                Foil = Recipe_Foil_sp[SQ - 1]
                break
            else:
                Foil = Recipe_Foil_sp[-1]

        # SBC - Fg
        Recipe_Fg = [40, 100, 200, 450, 1000, 1250, 1750]
        Recipe_Fg_sp = [30, 42, 55, 60, 75, 65, 60]
        for SQ in range(1, len(Recipe_Fg) + 1):
            if k <= Recipe_Fg[SQ - 1]:
                Fg = Recipe_Fg_sp[SQ - 1]
                break
            else:
                Fg = Recipe_Fg_sp[-1]

        # SBC - pressure
        Recipe_pres = [62.5, 125, 150, 200, 500, 750, 1000, 1750]
        Recipe_pres_sp = [0.6, 0.7, 0.8, 0.9, 1.1, 1, 0.9, 0.9]
        for SQ in range(1, len(Recipe_pres) + 1):
            if k <= Recipe_pres[SQ - 1]:
                pressure = Recipe_pres_sp[SQ - 1]
                break
            else:
                pressure = Recipe_pres_sp[-1]

        # SBC - F_discharge
        Recipe_discharge = [500, 510, 650, 660, 750, 760, 850, 860, 950, 960, 1050,
                            1060, 1150, 1160, 1250, 1260, 1350, 1360, 1750]
        Recipe_discharge_sp = [0, 4000, 0, 4000, 0, 4000, 0, 4000, 0, 4000, 0, 4000, 0, 4000, 0, 4000, 0, 4000, 0, 0]
        for SQ in range(1, len(Recipe_discharge) + 1):
            if k <= Recipe_discharge[SQ - 1]:
                F_discharge = -Recipe_discharge_sp[SQ - 1]
                break
            else:
                F_discharge = Recipe_discharge_sp[-1]

        # SBC - Fw
        Recipe_water = [250, 375, 750, 800, 850, 1000, 1250, 1350, 1750]
        Recipe_water_sp = [0, 500, 100, 0, 400, 150, 250, 0, 100]
        for SQ in range(1, len(Recipe_water) + 1):
            if k <= Recipe_water[SQ - 1]:
                Fw = Recipe_water_sp[SQ - 1]
                break
            else:
                Fw = Recipe_water_sp[-1]

        # SBC - F_PAA
        Recipe_PAA = [25, 200, 1000, 1500, 1750]
        Recipe_PAA_sp = [5, 0, 10, 4, 0]
        for SQ in range(1, len(Recipe_PAA) + 1):
            if k <= Recipe_PAA[SQ - 1]:
                Fpaa = Recipe_PAA_sp[SQ - 1]
                break
            else:
                Fpaa = Recipe_PAA_sp[-1]

        # Add PRBS to substrate flow rate  (Fpaa)
        if ctrl_flags.PRBS == 1:
            if k > 500 and np.remainder(k, 100) == 0:
                random_number = np.random.randint(1, 4)
                noise_factor = 1
                if random_number == 1:
                    random_noise = 0
                elif random_number == 2:
                    random_noise = noise_factor
                else:
                    random_noise = -noise_factor
                x.PRBS_noise_addition[k - 1] = random_noise
            Fpaa = x.Fpaa.y[k - 2] if k > 475 else Fpaa
            if k > 500 and np.remainder(k, 100) == 0:
                Fpaa = x.Fpaa.y[k - 2] + x.PRBS_noise_addition[-1]
        else:
            x.PRBS_noise_addition[k - 1] = 0
        xd.NH3_shots.y[k - 1] = 0

    # Process faults
    # 0 - No Faults
    # 1 - Aeration rate fault
    # 2 - Vessel back pressure  fault
    # 3 - Substrate feed rate fault
    # 4 - Base flow-rate fault
    # 5 - Coolant flow-rate fault
    # 6 - All of the above faults

    # Aeration fault
    if ctrl_flags.Faults == 1 or ctrl_flags.Faults == 6:
        if 100 <= k <= 120:
            Fg = 20
            u.Fault_ref = 1
        if 500 <= k <= 550:
            Fg = 20
            u.Fault_ref = 1

    # Pressure fault
    if ctrl_flags.Faults == 2 or ctrl_flags.Faults == 6:
        if 500 <= k <= 520:
            pressure = 2
            u.Fault_ref = 1
        if 1000 <= k <= 1200:
            pressure = 2
            u.Fault_ref = 1

    # Substrate feed fault
    if ctrl_flags.Faults == 3 or ctrl_flags.Faults == 6:
        if 100 <= k <= 150:
            Fs = 2
            u.Fault_ref = 1
        if 380 <= k <= 460:
            Fs = 20
            u.Fault_ref = 1
        if 1000 <= k <= 1070:
            Fs = 20
            u.Fault_ref = 1

    # Base flow-rate  fault
    if ctrl_flags.Faults == 4 or ctrl_flags.Faults == 6:
        if 400 <= k <= 420:
            Fb = 5
            u.Fault_ref = 1
        if 700 <= k <= 800:
            Fb = 10
            u.Fault_ref = 1

    # Coolant water flow-rate fault
    if ctrl_flags.Faults == 5 or ctrl_flags.Faults == 6:
        if 350 <= k <= 450:
            Fc = 2
            u.Fault_ref = 1
        if 1200 <= k <= 1350:
            Fc = 10
            u.Fault_ref = 1

    # Bulidng PID controller for PAA
    # builds the error history.  Samples 1 and 2 are calculated separately
    # because there is only instance of the error available
    if ctrl_flags.Raman_spec == 2:
        PAA_sp = 1200
        if k == 1 or k == 2:
            PAA_err = PAA_sp - x.PAA.y[0]
            PAA_err1 = PAA_sp - x.PAA.y[0]
        else:
            PAA_err = PAA_sp - x.PAA.y[k - 2]
            PAA_err1 = PAA_sp - x.PAA.y[k - 3]

        # builds the temperature history of current and previous two samples
        if k * h >= 10:
            if k == 1 or k == 2:
                temp = x.PAA_pred.y[0]
                temp1 = x.PAA_pred.y[0]
                temp2 = x.PAA_pred.y[0]
            elif k == 3:
                temp = x.PAA_pred.y[1]
                temp1 = x.PAA_pred.y[0]
                temp2 = x.PAA_pred.y[0]
            else:
                temp = x.PAA_pred.y[k - 3]
                temp1 = x.PAA_pred.y[k - 4]
                temp2 = x.PAA_pred.y[k - 5]

            if k == 1:
                Fpaa = PIDSimple3(x.Fpaa.y[0], PAA_err, PAA_err1, temp, temp1, temp2, 0, 150, 0.1, 0.50, 0 * 0.002, h)
            else:
                Fpaa = PIDSimple3(x.Fpaa.y[k - 2], PAA_err, PAA_err1, temp, temp1, temp2,
                                  0, 150, 0.1, 0.50, 0 * 0.002, h)

    # Controller vector
    u.Fg = Fg
    u.RPM = 100
    u.Fs = Fs
    u.Fa = Fa
    u.Fb = Fb
    u.Fc = Fc
    u.Fh = Fh
    u.d1 = ph_on_off
    u.tfl = temp_on_off
    u.Fw = Fw
    u.pressure = pressure
    u.viscosity = viscosity
    u.Fremoved = F_discharge
    u.Fpaa = Fpaa
    u.Foil = Foil
    u.NH3_shots = xd.NH3_shots.y[k - 1]

    # Defining Fault reference
    if not hasattr(u, 'Fault_ref'):
        u.Fault_ref = 0
    return u, x
