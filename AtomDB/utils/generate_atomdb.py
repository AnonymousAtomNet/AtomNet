import os
import argparse

import pandas as pd
import seaborn as sns


parser = argparse.ArgumentParser(
    description="STM32CUBEMX log profiler concat tools")
parser.add_argument("-d", "--dir", help="log root dir", type=str, default="./")
parser.add_argument("-n", "--name", help="output name",
                    type=str, default="mcu_all")


def main(args):
    # declare output name
    output_csv_name = args.name + ".csv"
    output_xlsx_name = args.name + ".xlsx"

    dfs = []
    for roots, dirs, files in os.walk(args.dir):
        for file in files:
            if file == "report.csv":
                dfs.append(pd.read_csv(os.path.join(roots, file)))

    # concat reports
    all = pd.concat(dfs)

    # Export
    columns = ['name', 'type', 'conv_type', 'groups', 'c_in', 'h_in', 'w_in', 'k1', 'k2',
               'c_out', 'h_out', 'w_out', 'rom', 'flops', 'time', 'flops/ms', 'ms/MFlops']
    # export csv
    # results = results.fillna(0)
    all.reset_index(drop=True).to_csv(
        output_csv_name, index=False, columns=columns)

    # export excel
    # 0. prepare style
    cmap = sns.light_palette("green", as_cmap=True)
    # 1. read csv
    report_excel = pd.read_csv(output_csv_name, delimiter=",")
    # 2. export to excel
    report_excel.reset_index(drop=True).style.hide_index().background_gradient(
        cmap=cmap, subset=["rom", "flops", "time", "flops/ms", "ms/MFlops"]
    ).to_excel(output_xlsx_name, index=False, engine="openpyxl", columns=columns)


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
