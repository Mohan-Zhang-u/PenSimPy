import os
import pandas as pd


def save_csv(run_id, avg_pHs, avg_Ts, penicillin_yields, median_pH, median_T, Xref):
    """
    Save data as csv
    :param run_id:
    :param avg_pHs:
    :param avg_Ts:
    :param penicillin_yields:
    :param median_pH:
    :param median_T:
    :param Xref:
    :return:
    """
    if not os.path.exists("./data"):
        os.mkdir("./data")
    output_dir = f"./data/{run_id}"
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    # df = pd.DataFrame(data={"avg_pH": avg_pHs, "avg_T": avg_Ts, "penicillin_yields": penicillin_yields})
    # file_path = os.path.join(output_dir, 'batch_statistics.csv')
    # df.to_csv(file_path, sep=',', index=True)

    # df = pd.DataFrame(data={"median_pH": median_pH, "median_T": median_T})
    # file_path = os.path.join(output_dir, 'batch_median_trend.csv')
    # df.to_csv(file_path, sep=',', index=False)

    print(f"=== penicillin_yields: {penicillin_yields}")

    time = [round(t, 1) for t in Xref.pH.t]
    df = pd.DataFrame(data={"Batch ID": run_id,
                            "time": time,
                            "pH": Xref.pH.y,
                            "Temperature": Xref.T.y,
                            "Sugar flowrate": Xref.Fs.y,
                            "Aeration rate": Xref.Fg.y,
                            "Acid flowrate": Xref.Fa.y,
                            "Base flowrate": Xref.Fb.y,
                            "Cooling water flowrate": Xref.Fc.y,
                            "Heating water flowrate": Xref.Fh.y,
                            "Water for injection/dilution": Xref.Fw.y,
                            "pressure": Xref.pressure.y,
                            "Discharge rate": Xref.Fremoved.y,
                            "PAA flow": Xref.Fpaa.y,
                            "Oil flow": Xref.Foil.y,
                            "Substrate concentration": Xref.S.y,
                            "Dissolved oxygen concentration": Xref.DO2.y,
                            "Biomass concentration": Xref.X.y,
                            "Penicillin concentration": Xref.P.y,
                            "Vessel volume": Xref.V.y,
                            "Vessel weight": Xref.Wt.y,
                            "Carbon dioxide percent in off-gas": Xref.CO2outgas.y,
                            "Dissolved CO_2": Xref.CO2_d.y,
                            "Oxygen in percent in off-gas": Xref.O2.y})

    df = df.iloc[::5, :]
    file_path = os.path.join(output_dir, f'batch_data_{str(int(penicillin_yields))}.csv')
    df.to_csv(file_path, sep=',', index=False)

    wavenumber = Xref.Raman_Spec.Wavenumber
    df = pd.DataFrame(Xref.Raman_Spec.Intensity, columns=wavenumber)
    df = df[df.columns[::-1]]
    df['peni_concentraion'] = Xref.P.y
    file_path = os.path.join(output_dir, 'raman.csv')
    df.to_csv(file_path, sep=',', index=False)
