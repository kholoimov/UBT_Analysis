import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import ROOT

from model import extract_plot_data


def plot_residual_histogram(residuals, output_name, title="Track-hit residuals", bins=100):
    plt.figure(figsize=(8, 6))

    if len(residuals) > 0:
        plt.hist(residuals, bins=bins, histtype="step", linewidth=1.8)
        mean_val = np.mean(residuals)
        rms_val = np.std(residuals)
        plt.axvline(mean_val, linestyle="--", label=f"mean = {mean_val:.3f}")
        plt.legend()
        print(f"{title}: N={len(residuals)}, mean={mean_val:.6f}, rms={rms_val:.6f}")
    else:
        print(f"{title}: no entries")

    plt.xlabel(r"Residual distance $\sqrt{(x_{state}-x_{hit})^2 + (y_{state}-y_{hit})^2}$")
    plt.ylabel("Entries")
    plt.title(title)
    plt.grid(True)
    plt.savefig(output_name)
    plt.close()


def plot_residual_vs_momentum_plot(
    res,
    mom,
    xlog=True,
    ylog=True,
    ylim=(-100, 100),
    name="residual_vs_momentum",
    ylabel="Distance between extra state and true hit",
):
    plt.figure(figsize=(8, 6))
    plt.scatter(mom, res, s=5)
    plt.ylabel(ylabel)
    plt.xlabel("Muon momentum [GeV]")
    plt.ylim(ylim)

    if len(res) > 0 and len(mom) > 0:
        if ylog:
            plt.yscale("log")
        if xlog:
            plt.xscale("log")

    plt.title(name)
    plt.grid(True)
    plt.savefig(f"{name}.png")
    plt.close()


def plot_coord_bias_vs_momentum_plot(
    x_distance,
    y_distance,
    mom,
    name="coord_vs_momentum",
    ylabel="Distance between extra state and true hit",
):
    plt.figure(figsize=(8, 6))
    plt.scatter(mom, x_distance, s=5, label="x distance")
    plt.scatter(mom, y_distance, s=5, label="y distance")
    plt.ylabel(ylabel)
    plt.xlabel("Muon momentum [GeV]")
    plt.ylim(-150, 150)
    plt.legend()

    if len(mom) > 0:
        plt.xscale("log")

    plt.title("Distance vs Momentum")
    plt.grid(True)
    plt.savefig(f"{name}.png")
    plt.close()


def plot_residual_vs_state_coordinate(
    x_data,
    y_data,
    xlim=(-250, 250),
    ylim=(-100, 100),
    name="residual distance vs state location",
    ylabel="Residual",
    xlabel="State location",
):
    plt.figure(figsize=(8, 6))
    plt.scatter(x_data, y_data, s=5, label="x distance")

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.ylim(ylim)
    plt.xlim(xlim)

    plt.legend()
    plt.title(name)
    plt.grid(True)
    plt.savefig(f"{name}.png")
    plt.close()


def plot_2d_ROOT_histogram(x_data, y_data, name="UpstreamTagger_Hit_Map"):
    bin_width = 0.05

    xmin = -4.0
    xmax = 4.0
    ymin = -3.0
    ymax = 3.0

    nbx = int((xmax - xmin) / bin_width)
    nby = int((ymax - ymin) / bin_width)

    hXY = ROOT.TH2D(
        "hXY",
        f"{name};X [m];Y [m]",
        nbx, xmin, xmax,
        nby, ymin, ymax,
    )

    for x, y in zip(x_data, y_data):
        hXY.Fill(x * 0.01, y * 0.01)

    ROOT.gStyle.SetOptStat(0)
    ROOT.gStyle.SetPalette(62)

    c = ROOT.TCanvas("c_upstream_xy", name, 900, 700)
    c.SetRightMargin(0.14)
    c.SetLogz(True)

    hXY.Draw("COLZ")
    c.SaveAs(f"{name}_5cm_Passed_ST.pdf")


def make_all_summary_plots(events, output_prefix=""):
    results = extract_plot_data(events)

    global_residuals = results["residuals"]
    global_momentum_for_histogram = results["momenta"]
    global_distance_x = results["dx"]
    global_distance_y = results["dy"]
    global_px = results["px"]
    global_py = results["py"]
    global_pz = results["pz"]

    state_x = results["x_state"]
    state_y = results["y_state"]

    hit_x = results["x_hit"]
    hit_y = results["y_hit"]

    plot_residual_histogram(
        global_residuals,
        f"{output_prefix}track_hit_residuals_all.png",
        title="All matched extra-state vs hit residuals",
        bins=100,
    )

    if len(global_residuals) > 0 and len(global_momentum_for_histogram) > 0:
        plot_residual_vs_momentum_plot(
            global_residuals,
            global_momentum_for_histogram,
            ylim=(1e-2, 1e3),
            name=f"{output_prefix}residual_vs_momentum_state_momentum",
        )

    if len(global_distance_x) > 0 and len(global_momentum_for_histogram) > 0:
        plot_residual_vs_momentum_plot(
            global_distance_x,
            global_momentum_for_histogram,
            ylog=False,
            ylabel="X bias between extra state and true hit",
            ylim=(-200, 200),
            name=f"{output_prefix}X_bias_vs_mom",
        )

    if len(global_distance_y) > 0 and len(global_momentum_for_histogram) > 0:
        plot_residual_vs_momentum_plot(
            global_distance_y,
            global_momentum_for_histogram,
            ylog=False,
            ylabel="Y bias between extra state and true hit",
            name=f"{output_prefix}Y_bias_vs_mom",
        )

    if len(global_distance_x) > 0 and len(global_distance_y) > 0 and len(global_momentum_for_histogram) > 0:
        plot_coord_bias_vs_momentum_plot(
            global_distance_x,
            global_distance_y,
            global_momentum_for_histogram,
            name=f"{output_prefix}bias_vs_mom",
        )

    if len(global_distance_y) > 0 and len(global_py) > 0:
        plot_residual_vs_momentum_plot(
            global_distance_y,
            global_py,
            ylog=False,
            ylabel="Y bias between extra state and true hit",
            name=f"{output_prefix}Y_bias_vs_mom_py",
        )

    if len(global_distance_x) > 0 and len(global_px) > 0:
        plot_residual_vs_momentum_plot(
            global_distance_x,
            global_px,
            ylog=False,
            ylabel="X bias between extra state and true hit",
            name=f"{output_prefix}X_bias_vs_mom_px",
        )

    if len(global_distance_x) > 0 and len(global_pz) > 0:
        plot_residual_vs_momentum_plot(
            global_distance_x,
            global_pz,
            ylog=False,
            ylabel="X bias between extra state and true hit",
            name=f"{output_prefix}X_bias_vs_mom_pz",
        )

    if len(global_distance_x) > 0 and len(global_py) > 0:
        plot_residual_vs_momentum_plot(
            global_distance_x,
            global_py,
            ylog=False,
            ylabel="X bias between extra state and true hit",
            name=f"{output_prefix}X_bias_vs_mom_py",
        )

    plot_residual_vs_state_coordinate(
        state_x,
        global_distance_y,
        ylabel="Y bias between extra state and true hit",
        xlabel="X coordinate of state in ST",
        name=f"{output_prefix}Y_bias_vs_X_location",
    )

    plot_residual_vs_state_coordinate(
        state_y,
        global_distance_y,
        ylabel="Y bias between extra state and true hit",
        xlabel="Y coordinate of state in ST",
        name=f"{output_prefix}Y_bias_vs_Y_location",
    )

    plot_residual_vs_state_coordinate(
        state_x,
        global_distance_x,
        ylabel="X bias between extra state and true hit",
        xlabel="X coordinate of state in ST",
        name=f"{output_prefix}X_bias_vs_X_location",
    )

    plot_residual_vs_state_coordinate(
        state_y,
        global_distance_x,
        ylabel="X bias between extra state and true hit",
        xlabel="Y coordinate of state in ST",
        name=f"{output_prefix}X_bias_vs_Y_location",
    )

    plot_residual_vs_state_coordinate(
        state_x,
        global_distance_x,
        ylim=(-200, 200),
        ylabel="X bias between extra state and true hit",
        xlabel="X coordinate of extra state",
        name=f"{output_prefix}X_bias_vs_X_extra",
    )

    plot_residual_vs_state_coordinate(
        state_y,
        global_distance_x,
        ylim=(-200, 200),
        ylabel="X bias between extra state and true hit",
        xlabel="Y coordinate of extra state",
        name=f"{output_prefix}X_bias_vs_Y_extra",
    )

    plot_residual_vs_state_coordinate(
        state_x,
        global_distance_y,
        ylim=(-200, 200),
        ylabel="Y bias between extra state and true hit",
        xlabel="X coordinate of extra state",
        name=f"{output_prefix}Y_bias_vs_X_extra",
    )

    plot_residual_vs_state_coordinate(
        state_y,
        global_distance_y,
        ylim=(-200, 200),
        ylabel="Y bias between extra state and true hit",
        xlabel="Y coordinate of extra state",
        name=f"{output_prefix}Y_bias_vs_Y_extra",
    )

    plot_residual_vs_state_coordinate(
        hit_y,
        global_distance_y,
        ylim=(-200, 200),
        ylabel="Y bias between extra state and true hit",
        xlabel="Y coordinate of true hit",
        name=f"{output_prefix}Y_bias_vs_Y_true",
    )

    plot_residual_vs_state_coordinate(
        hit_x,
        global_distance_x,
        ylim=(-200, 200),
        ylabel="X bias between extra state and true hit",
        xlabel="X coordinate of true hit",
        name=f"{output_prefix}X_bias_vs_X_true",
    )

    plot_2d_ROOT_histogram(hit_x, hit_y, name=f"{output_prefix}true_hit")
    plot_2d_ROOT_histogram(state_x, state_y, name=f"{output_prefix}extra_state")

    df = pd.DataFrame({
        "x_true": hit_x,
        "y_true": hit_y,
        "momentum": global_momentum_for_histogram,
    })

    low_momentum = df["momentum"] < 10

    plot_2d_ROOT_histogram(
        df["x_true"][low_momentum],
        df["y_true"][low_momentum],
        name=f"{output_prefix}Low_momentum_XY_map",
    )
    plot_2d_ROOT_histogram(
        df["x_true"][~low_momentum],
        df["y_true"][~low_momentum],
        name=f"{output_prefix}High_momentum_XY_map",
    )
