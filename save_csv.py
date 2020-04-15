import os
import pandas as pd


def save_csv(run_id, avg_pHs, avg_Ts, penicillin_yields, median_pH, median_T):
    """
    Save data as csv
    :param run_id:
    :param avg_pHs:
    :param avg_Ts:
    :param penicillin_yields:
    :param median_pH:
    :param median_T:
    :return:
    """
    if not os.path.exists("./data"):
        os.mkdir("./data")
    output_dir = f"./data/{run_id}"
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    df = pd.DataFrame(data={"avg_pH": avg_pHs, "avg_T": avg_Ts, "penicillin_yields": penicillin_yields})
    file_path = os.path.join(output_dir, 'batch_statistics.csv')
    df.to_csv(file_path, sep=',', index=True)

    df = pd.DataFrame(data={"median_pH": median_pH, "median_T": median_T})
    file_path = os.path.join(output_dir, 'batch_median_trend.csv')
    df.to_csv(file_path, sep=',', index=False)
