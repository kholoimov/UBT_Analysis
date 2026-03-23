import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import ROOT

from analysis_io import build_output_path
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
    plt.savefig(build_output_path(output_name))
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
    plt.savefig(build_output_path(f"{name}.png"))
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
    plt.savefig(build_output_path(f"{name}.png"))
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
    plt.savefig(build_output_path(f"{name}.png"))
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
    c.SaveAs(build_output_path(f"{name}_5cm_Passed_ST.pdf"))


def plot_event_detector_views(event, event_number, output_prefix=""):
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharex=True)

    all_z = []
    all_x = []
    all_y = []

    if event.UBT_hits:
        ubt_z = [hit.z for hit in event.UBT_hits]
        ubt_x = [hit.x for hit in event.UBT_hits]
        ubt_y = [hit.y for hit in event.UBT_hits]
        axes[0].scatter(ubt_z, ubt_x, s=45, marker="o", color="C1", label="UBT hits")
        axes[1].scatter(ubt_z, ubt_y, s=45, marker="o", color="C1", label="UBT hits")
        all_z.extend(ubt_z)
        all_x.extend(ubt_x)
        all_y.extend(ubt_y)

    plotted_track_label = False
    for itrk, track in enumerate(event.ST_tracks):
        if not track.hits:
            continue

        z_vals = [hit[2] for hit in track.hits]
        x_vals = [hit[0] for hit in track.hits]
        y_vals = [hit[1] for hit in track.hits]
        color = f"C{itrk % 10}"
        label = "Track states" if not plotted_track_label else None

        axes[0].plot(z_vals, x_vals, marker=".", markersize=5, linewidth=1.2, alpha=0.85, color=color, label=label)
        axes[1].plot(z_vals, y_vals, marker=".", markersize=5, linewidth=1.2, alpha=0.85, color=color, label=label)

        plotted_track_label = True
        all_z.extend(z_vals)
        all_x.extend(x_vals)
        all_y.extend(y_vals)

    if event.ExtraStates:
        extra_z = [state.z for state in event.ExtraStates]
        extra_x = [state.x for state in event.ExtraStates]
        extra_y = [state.y for state in event.ExtraStates]
        axes[0].scatter(
            extra_z,
            extra_x,
            s=90,
            marker="x",
            linewidths=2.0,
            color="black",
            label="Extra state",
        )
        axes[1].scatter(
            extra_z,
            extra_y,
            s=90,
            marker="x",
            linewidths=2.0,
            color="black",
            label="Extra state",
        )
        all_z.extend(extra_z)
        all_x.extend(extra_x)
        all_y.extend(extra_y)

        z_reference = all_z if all_z else extra_z
        z_min_ref = min(z_reference)
        z_max_ref = max(z_reference)
        z_span = max(z_max_ref - z_min_ref, 1.0)
        momentum_segment = 0.08 * z_span

        for state in event.ExtraStates:
            if abs(state.pz) < 1e-12:
                continue

            dz_short = momentum_segment
            dz_full = z_span

            short_z = [state.z, state.z + dz_short]
            short_x = [state.x, state.x + (state.px / state.pz) * dz_short]
            short_y = [state.y, state.y + (state.py / state.pz) * dz_short]

            full_z = [state.z - 0.5 * dz_full, state.z + 0.5 * dz_full]
            full_x = [
                state.x + (state.px / state.pz) * (full_z[0] - state.z),
                state.x + (state.px / state.pz) * (full_z[1] - state.z),
            ]
            full_y = [
                state.y + (state.py / state.pz) * (full_z[0] - state.z),
                state.y + (state.py / state.pz) * (full_z[1] - state.z),
            ]

            axes[0].plot(
                full_z,
                full_x,
                linestyle="--",
                linewidth=1.4,
                alpha=0.18,
                color="black",
                label="Linear extrapolation" if state is event.ExtraStates[0] else None,
            )
            axes[1].plot(
                full_z,
                full_y,
                linestyle="--",
                linewidth=1.4,
                alpha=0.18,
                color="black",
                label="Linear extrapolation" if state is event.ExtraStates[0] else None,
            )

            axes[0].plot(
                short_z,
                short_x,
                linewidth=2.2,
                color="red",
                label="Extra-state momentum" if state is event.ExtraStates[0] else None,
            )
            axes[1].plot(
                short_z,
                short_y,
                linewidth=2.2,
                color="red",
                label="Extra-state momentum" if state is event.ExtraStates[0] else None,
            )

            all_z.extend(full_z)
            all_x.extend(full_x)
            all_y.extend(full_y)

    first_state_label_drawn = False
    for track in event.ST_tracks:
        if len(track.hits) < 2:
            continue

        first_hit = track.hits[0]
        second_hit = track.hits[1]

        z0 = first_hit[2]
        x0 = first_hit[0]
        y0 = first_hit[1]
        dz = second_hit[2] - first_hit[2]

        if abs(dz) < 1e-12:
            continue

        slope_x = (second_hit[0] - first_hit[0]) / dz
        slope_y = (second_hit[1] - first_hit[1]) / dz

        z_reference = all_z if all_z else [z0, second_hit[2]]
        z_min_ref = min(z_reference)
        z_max_ref = max(z_reference)
        z_span = max(z_max_ref - z_min_ref, 1.0)

        first_state_z = [z0, z0 + z_span]
        first_state_x = [x0, x0 + slope_x * z_span]
        first_state_y = [y0, y0 + slope_y * z_span]

        axes[0].plot(
            first_state_z,
            first_state_x,
            linestyle=":",
            linewidth=1.8,
            alpha=0.35,
            color="blue",
            label="Linear extrapolation from first track state" if not first_state_label_drawn else None,
        )
        axes[1].plot(
            first_state_z,
            first_state_y,
            linestyle=":",
            linewidth=1.8,
            alpha=0.35,
            color="blue",
            label="Linear extrapolation from first track state" if not first_state_label_drawn else None,
        )

        first_state_label_drawn = True
        all_z.extend(first_state_z)
        all_x.extend(first_state_x)
        all_y.extend(first_state_y)

    axes[0].set_title(f"Event {event_number}: XZ view")
    axes[1].set_title(f"Event {event_number}: YZ view")
    axes[0].set_xlabel("Z")
    axes[1].set_xlabel("Z")
    axes[0].set_ylabel("X")
    axes[1].set_ylabel("Y")

    for ax in axes:
        ax.grid(True)
        ax.legend()

    if all_z:
        z_min = min(all_z)
        z_max = max(all_z)
        z_pad = max((z_max - z_min) * 0.05, 1.0)
        for ax in axes:
            ax.set_xlim(z_min - z_pad, z_max + z_pad)

    if all_x:
        x_min = min(all_x)
        x_max = max(all_x)
        x_pad = max((x_max - x_min) * 0.1, 10.0)
        axes[0].set_ylim(x_min - x_pad, x_max + x_pad)

    if all_y:
        y_min = min(all_y)
        y_max = max(all_y)
        y_pad = max((y_max - y_min) * 0.1, 10.0)
        axes[1].set_ylim(y_min - y_pad, y_max + y_pad)

    fig.tight_layout()
    fig.savefig(build_output_path(f"{output_prefix}event_{event_number}_detector_views.png"))
    plt.close(fig)


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
