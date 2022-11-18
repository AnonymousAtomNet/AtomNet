import os
import argparse
import pandas as pd
import seaborn as sns

parser = argparse.ArgumentParser(description="STM32CUBEMX log profiler")
parser.add_argument(
    "-d", "--dir", help="stm32 output log dir", type=str, default="./")


def main(args):
    # Set TMP_DIR
    TMP_DIR = os.path.join(args.dir, "data_report")
    # Generate csv files from log
    _file_path = os.path.dirname(os.path.abspath(__file__))
    os.system(os.path.join(_file_path, "parse_log.sh") + " " + args.dir)

    # Read csv
    flops = pd.read_csv(
        os.path.join(TMP_DIR, "flops.csv"),
        names=["c_id", "name", "flops", "rom"],
        delimiter=" ",
    )
    shapes_kernel = pd.read_csv(os.path.join(TMP_DIR, "shapes_kernel.csv"), names=[
                                "name", "c_in", "c_out/groups", "k1", "k2"], delimiter=" ",)
    shapes_output = pd.read_csv(os.path.join(TMP_DIR, "shapes_output.csv"), names=[
                                "name", "n", "h_out", "w_out", "c_out"], delimiter=" ",)
    shapes_input = pd.read_csv(os.path.join(TMP_DIR, "shapes_input.csv"), names=[
                                "name", "n", "h_in", "w_in", "c_in"], delimiter=" ",)
    latency = pd.read_csv(os.path.join(TMP_DIR, "latency.csv"), names=[
                          "c_id", "time"], delimiter=" ")

    # Merge report
    results = pd.merge(
        flops,
        shapes_kernel.loc[:, ["name", "c_out/groups", "k1", "k2"]],
        how="left",
        on="name",
    )
    results = pd.merge(results, shapes_output.loc[:, [
                       "name", "c_out", "h_out", "w_out"]], how="left", on="name")
    results = pd.merge(results, shapes_input.loc[:, [
                       "name", "c_in", "h_in", "w_in"]], how="left", on="name")
    results = pd.merge(
        results, latency.loc[:, ["c_id", "time"]], how="left", on="c_id")

    # calculate
    results.loc[results["flops"] != 0,
                ["flops/ms"]] = results["flops"] // results["time"]
    results.loc[results["flops"] != 0,
                ["ms/MFlops"]] = (1e9 * results["time"] // results["flops"]) / 1e3
    results["type"] = results["name"].str.replace('_[0-9]+', '', regex=True)
    results.loc[results["type"] == 'conv2d',
                ["groups"]] = results["c_out"] // results["c_out/groups"]
    results.loc[(results["type"] == 'conv2d') & (
        results["groups"] == results["c_in"]), ["conv_type"]] = "DepthWise"
    results.loc[(results["type"] == 'conv2d') & (
        results["k1"] == 1), ["conv_type"]] = "PointWise"
    results.loc[(results["type"] == 'conv2d') & (
        results["conv_type"].isnull()), ["conv_type"]] = "Ordinary"

    # Export
    columns = ['name', 'type', 'conv_type', 'groups', 'c_in', 'h_in', 'w_in', 'k1', 'k2',
               'c_out', 'h_out', 'w_out', 'rom', 'flops', 'time', 'flops/ms', 'ms/MFlops']
    # export csv
    results.to_csv(
        os.path.join(TMP_DIR, "report.csv"), index=False, columns=columns)

    # export excel
    # 0. prepare style
    cmap = sns.light_palette("green", as_cmap=True)
    # 1. read csv
    report_excel = pd.read_csv(os.path.join(
        TMP_DIR, "report.csv"), delimiter=",")
    # 2. export to excel
    report_excel.style.hide_index().background_gradient(
        cmap=cmap, subset=["rom", "flops", "time", "flops/ms", "ms/MFlops"]
    ).to_excel(os.path.join(TMP_DIR, "report_excel.xlsx"), index=False, engine="openpyxl", columns=columns)


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
