import sys
import glob
import math
import os
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp
import ROOT
from concurrent.futures import ProcessPoolExecutor, as_completed
import pandas as pd


# -----------------------------------------------------------------------------
# ROOT access
# -----------------------------------------------------------------------------
def get_ROOT():
    import ROOT
    return ROOT


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------
def expand_patterns(patterns):
    files = []
    for pattern in patterns:
        matched = glob.glob(pattern)
        if matched:
            files.extend(sorted(matched))
        else:
            print(f"Warning: no files matched pattern '{pattern}'")
    return list(dict.fromkeys(files))


def get_all_track_points(track):
    """
    Read stored fitted states from the GenFit track object.
    These are used only for visualisation / station coverage checks.
    No propagation is done here.
    """
    points = []
    n = track.getNumPoints()

    for i in range(n):
        try:
            state = track.getFittedState(i)
            pos = state.getPos()
            mom = state.getMom()
        except Exception:
            continue

        points.append((
            pos.X(),
            pos.Y(),
            pos.Z(),
            mom.X(),
            mom.Y(),
            mom.Z(),
        ))

    return points


def get_vector3_components(v):
    """
    Robust reader for TVector3-like objects.
    """
    try:
        return float(v.X()), float(v.Y()), float(v.Z())
    except Exception:
        return float(v.x()), float(v.y()), float(v.z())


def get_saved_reference_state(pos_branch, mom_branch, track_index):
    """
    Read saved per-track reference state from separate TVector3 branches.

    Returns
    -------
    (x, y, z, px, py, pz)
    """
    pos = pos_branch[track_index]
    mom = mom_branch[track_index]

    print("pos_X: ", pos.x())

    x, y, z = get_vector3_components(pos)
    px, py, pz = get_vector3_components(mom)

    return x, y, z, px, py, pz


def extrapolate_linearly_from_saved_state(x0, y0, z0, px0, py0, pz0, z_target):
    """
    Straight-line propagation from a saved state:
        x(z) = x0 + (px/pz) * (z - z0)
        y(z) = y0 + (py/pz) * (z - z0)

    Momentum is kept constant.
    """
    if pz0 is None or abs(pz0) < 1e-12:
        return x0, y0, z0, px0, py0, pz0, False

    dz = float(z_target) - float(z0)

    xp = float(x0) + float(px0 / pz0) * dz
    yp = float(y0) + float(py0 / pz0) * dz
    zp = float(z_target)

    return xp, yp, zp, px0, py0, pz0, True


def has_hits_in_all_stations(all_points, station_z=(8407.0, 8607.0, 9307.0, 9507.0), tolerance=100.0):
    station_hits = [False] * len(station_z)

    for p in all_points:
        z = p[2]
        for i, zs in enumerate(station_z):
            if abs(z - zs) < tolerance:
                station_hits[i] = True

    return all(station_hits)


def track_passes_selection_from_saved_state(ref_state, all_points, p_min=1.0, nmeas_min=25):
    """
    Selection based on saved state + stored track points.
    No GenFit propagation or fit-quality requirement.
    """
    try:
        _, _, _, px, py, pz = ref_state
    except Exception:
        return False, {
            "reason": "no_reference_state",
            "p": None,
            "n_meas": None,
        }

    p = math.sqrt(px * px + py * py + pz * pz)
    n_meas = len(all_points)
    has_both = has_hits_in_all_stations(all_points)

    passes = True
    reason = "passed"

    if not (p > p_min):
        passes = False
        reason = "momentum"
    elif not (n_meas > nmeas_min):
        passes = False
        reason = "n_measurements"
    elif not has_both:
        passes = False
        reason = "spectrometer_z_windows"

    info = {
        "reason": reason,
        "p": p,
        "n_meas": n_meas,
    }
    return passes, info


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

    plt.xlabel(r"Residual distance $\sqrt{(x_{prop}-x_{hit})^2 + (y_{prop}-y_{hit})^2}$")
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
    ylabel="Distance between extrapolated track and true hit",
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
    ylabel="Distance between extrapolated track and true hit",
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


def plot_xy_all_tracks(hit_x, hit_y, hit_z, corresponding_track_ids, track_results, event_number, output_prefix=""):
    plt.figure(figsize=(9, 9))

    for i in range(len(hit_x)):
        if i == 0:
            plt.scatter(hit_x[i], hit_y[i], label="UBT hit", color="C1")
        else:
            plt.scatter(hit_x[i], hit_y[i], color="C1")

        plt.annotate(
            str(corresponding_track_ids[i]),
            (hit_x[i], hit_y[i]),
            textcoords="offset points",
            xytext=(5, 5),
            fontsize=8,
        )

    first_prop_label = True
    for trk in track_results:
        if not trk["valid"]:
            continue

        prop_x = trk["prop_x"]
        prop_y = trk["prop_y"]
        mcid = trk["MCID"]

        if len(prop_x) > 0:
            label = "Selected track propagated" if first_prop_label else None
            plt.scatter(prop_x, prop_y, marker="x", s=60, label=label)
            first_prop_label = False

            plt.annotate(
                str(mcid),
                (prop_x[0], prop_y[0]),
                textcoords="offset points",
                xytext=(5, 5),
                fontsize=8,
            )

    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title(f"Event {event_number}: XY plane")
    plt.axis("equal")
    plt.grid(True)
    plt.legend()
    plt.xlim(-500, 500)
    plt.ylim(-500, 500)
    plt.savefig(f"{output_prefix}XY_event_{event_number}.png")
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


def plot_xz_yz_by_detector(
    hit_x,
    hit_y,
    hit_z,
    corresponding_track_ids,
    track_results,
    event_number,
    detector_zs=(3200.0, 8800.0),
    output_prefix="",
):
    plt.figure(figsize=(10, 7))

    for i in range(len(hit_x)):
        plt.scatter(hit_z[i], hit_x[i], color="C1")
        plt.annotate(
            str(corresponding_track_ids[i]),
            (hit_z[i], hit_x[i]),
            textcoords="offset points",
            xytext=(5, 5),
            fontsize=8,
        )

    first_track_label = True
    first_points_label = True
    for trk in track_results:
        if not trk["valid"]:
            continue

        prop_x = trk["prop_x"]
        prop_z = trk["prop_z"]
        all_points = trk["all_points"]
        x_tot = [p[0] for p in all_points]
        z_tot = [p[2] for p in all_points]
        mcid = trk["MCID"]

        if len(prop_x) > 0:
            order = sorted(range(len(prop_z)), key=lambda i: prop_z[i])
            z_sorted = [prop_z[i] for i in order]
            x_sorted = [prop_x[i] for i in order]

            label_track = "Selected track propagation" if first_track_label else None
            label_points = "Track fitted states" if first_points_label else None

            plt.scatter(z_sorted, x_sorted, marker="x", alpha=0.8, label=label_track)
            plt.scatter(z_tot, x_tot, marker="o", alpha=0.8, label=label_points)

            first_track_label = False
            first_points_label = False

            plt.annotate(
                str(mcid),
                (z_tot[0], x_tot[0]),
                textcoords="offset points",
                xytext=(5, 5),
                fontsize=8,
            )

    plt.xlabel("Z")
    plt.ylabel("X")
    plt.title(f"Event {event_number}: XZ plane")
    plt.grid(True)
    plt.legend()
    plt.savefig(f"{output_prefix}XZ_event_{event_number}.png")
    plt.close()

    plt.figure(figsize=(10, 7))

    for i in range(len(hit_y)):
        plt.scatter(hit_z[i], hit_y[i], color="C1")
        plt.annotate(
            str(corresponding_track_ids[i]),
            (hit_z[i], hit_y[i]),
            textcoords="offset points",
            xytext=(5, 5),
            fontsize=8,
        )

    first_track_label = True
    first_points_label = True
    for trk in track_results:
        if not trk["valid"]:
            continue

        prop_y = trk["prop_y"]
        prop_z = trk["prop_z"]
        all_points = trk["all_points"]
        y_tot = [p[1] for p in all_points]
        z_tot = [p[2] for p in all_points]
        mcid = trk["MCID"]

        if len(prop_y) > 0:
            order = sorted(range(len(prop_z)), key=lambda i: prop_z[i])
            z_sorted = [prop_z[i] for i in order]
            y_sorted = [prop_y[i] for i in order]

            label_track = "Selected track propagation" if first_track_label else None
            label_points = "Track fitted states" if first_points_label else None

            plt.scatter(z_sorted, y_sorted, marker="x", alpha=0.8, label=label_track)
            plt.scatter(z_tot, y_tot, marker="o", alpha=0.8, label=label_points)

            first_track_label = False
            first_points_label = False

            plt.annotate(
                str(mcid),
                (z_tot[0], y_tot[0]),
                textcoords="offset points",
                xytext=(5, 5),
                fontsize=8,
            )

    plt.xlabel("Z")
    plt.ylabel("Y")
    plt.title(f"Event {event_number}: YZ plane")
    plt.grid(True)
    plt.legend()
    plt.savefig(f"{output_prefix}YZ_event_{event_number}.png")
    plt.close()


def get_branch_object(tree_or_chain, branch_name):
    try:
        return getattr(tree_or_chain, branch_name)
    except Exception:
        return None


# -----------------------------------------------------------------------------
# Workers
# -----------------------------------------------------------------------------
def scan_pair_for_events_with_tracks(args):
    (
        pair_index,
        track_file,
        hit_file,
        track_tree_name,
        hit_tree_name,
        track_branch_name,
    ) = args

    ROOT = get_ROOT()

    track_chain = ROOT.TChain(track_tree_name)
    track_chain.Add(track_file)

    hit_chain = ROOT.TChain(hit_tree_name)
    hit_chain.Add(hit_file)

    n_track_entries = int(track_chain.GetEntries())
    n_hit_entries = int(hit_chain.GetEntries())
    n_common = min(n_track_entries, n_hit_entries)

    found_local_events = []

    for event_number in range(n_common):
        ok = track_chain.GetEntry(event_number)
        if ok <= 0:
            continue

        fit_tracks = get_branch_object(track_chain, track_branch_name)
        if fit_tracks is None:
            continue

        try:
            n_tracks = fit_tracks.size()
        except Exception:
            continue

        if n_tracks > 0:
            found_local_events.append(event_number)

    return {
        "pair_index": pair_index,
        "track_file": track_file,
        "hit_file": hit_file,
        "n_common": n_common,
        "events_with_tracks": found_local_events,
    }


def analyze_selected_event_in_pair(args):
    (
        pair_index,
        local_event_number,
        global_event_number,
        track_file,
        hit_file,
        track_tree_name,
        hit_tree_name,
        track_branch_name,
        hit_branch_name,
        track_state_pos_branch_name,
        track_state_mom_branch_name,
        detector_zs,
        output_prefix_base,
        verbose,
    ) = args

    ROOT = get_ROOT()

    track_chain = ROOT.TChain(track_tree_name)
    track_chain.Add(track_file)

    hit_chain = ROOT.TChain(hit_tree_name)
    hit_chain.Add(hit_file)

    ok1 = track_chain.GetEntry(local_event_number)
    ok2 = hit_chain.GetEntry(local_event_number)

    empty_result = {
        "global_event_number": global_event_number,
        "success": False,
        "residuals": [],
        "momenta": [],
        "dx": [],
        "dy": [],
        "ref_px": [],
        "ref_py": [],
        "ref_pz": [],
        "x_state": [],
        "y_state": [],
        "z_state": [],
        "x_xp": [],
        "y_xp": [],
    }

    if ok1 <= 0 or ok2 <= 0:
        return empty_result

    fit_tracks = get_branch_object(track_chain, track_branch_name)
    upstream_points = get_branch_object(hit_chain, hit_branch_name)
    mc_trackIDs = get_branch_object(track_chain, "fitTrack2MC")
    saved_state_pos = get_branch_object(track_chain, track_state_pos_branch_name)
    saved_state_mom = get_branch_object(track_chain, track_state_mom_branch_name)

    if fit_tracks is None or upstream_points is None or saved_state_pos is None or saved_state_mom is None:
        return empty_result

    try:
        n_tracks = fit_tracks.size()
        n_hits = upstream_points.size()
        n_pos = saved_state_pos.size()
        n_mom = saved_state_mom.size()
    except Exception:
        print("some of arrays are broken")
        return empty_result

    n_tracks = min(n_tracks, n_pos, n_mom)

    hit_x = []
    hit_y = []
    hit_z = []
    hit_px = []
    hit_py = []
    hit_pz = []
    corresponding_track_ids = []

    mom_ubt = ROOT.TVector3()
    for i in range(n_hits):
        try:
            hit = upstream_points[i]
            hit_x.append(hit.GetX())
            hit_y.append(hit.GetY())
            hit_z.append(hit.GetZ())

            corresponding_track_ids.append(hit.GetTrackID())
            hit.Momentum(mom_ubt)

            sx = mom_ubt.x()
            sy = mom_ubt.y()
            sz = mom_ubt.z()

            hit_px.append(sx)
            hit_py.append(sy)
            hit_pz.append(sz)
        except Exception:
            pass

    track_results = []
    event_residuals = []
    event_momenta = []
    event_dx = []
    event_dy = []

    event_ref_px = []
    event_ref_py = []
    event_ref_pz = []

    x_state = []
    y_state = []
    z_state = []

    x_extrapolated = []
    y_extrapolated = []

    for itrk in range(n_tracks):
        result = {
            "valid": False,
            "x_ref": None,
            "y_ref": None,
            "z_ref": None,
            "px_ref": None,
            "py_ref": None,
            "pz_ref": None,
            "prop_x": [],
            "prop_y": [],
            "prop_z": [],
            "all_points": [],
            "MCID": None,
        }

        try:
            track = fit_tracks[itrk]
        except Exception:
            track_results.append(result)
            continue

        if not track:
            track_results.append(result)
            continue

        try:
            all_points = get_all_track_points(track)
        except Exception:
            track_results.append(result)
            continue

        try:
            ref_state = get_saved_reference_state(saved_state_pos, saved_state_mom, itrk)
        except Exception:
            track_results.append(result)
            continue

        # passes, cut_info = track_passes_selection_from_saved_state(ref_state, all_points)
        # print("CUT PASSED:", passes)

        mcid = itrk
        if mc_trackIDs is not None:
            try:
                mcid = mc_trackIDs[itrk]
            except Exception:
                pass

        # if not passes:
        #     track_results.append(result)
        #     continue

        print("ref state: ", ref_state)
        x_ref, y_ref, z_ref, px_ref, py_ref, pz_ref = ref_state
        p_ref_mag = math.sqrt(px_ref * px_ref + py_ref * py_ref + pz_ref * pz_ref)

        result["valid"] = True
        result["x_ref"] = x_ref
        result["y_ref"] = y_ref
        result["z_ref"] = z_ref
        result["px_ref"] = px_ref
        result["py_ref"] = py_ref
        result["pz_ref"] = pz_ref
        result["all_points"] = all_points
        result["MCID"] = mcid

        for ihit in range(n_hits):
            zh = hit_z[ihit]
            xh = hit_x[ihit]
            yh = hit_y[ihit]
            hit_mcid = corresponding_track_ids[ihit]

            xp, yp, zp, pxp, pyp, pzp, ok = extrapolate_linearly_from_saved_state(
                x_ref, y_ref, z_ref,
                px_ref, py_ref, pz_ref,
                zh,
            )
            if xp is None or yp is None:
                continue

            result["prop_x"].append(xp)
            result["prop_y"].append(yp)
            result["prop_z"].append(zp)

            if hit_mcid == mcid:
                dx = xp - xh
                dy = yp - yh
                dist = math.sqrt(dx * dx + dy * dy)

                event_residuals.append(dist)
                event_dx.append(dx)
                event_dy.append(dy)

                event_ref_px.append(px_ref)
                event_ref_py.append(py_ref)
                event_ref_pz.append(pz_ref)

                total_momentum_ubt = np.sqrt(hit_px[ihit] ** 2 + hit_py[ihit] ** 2 + hit_pz[ihit] ** 2)
                total_momentum_state = np.sqrt(px_ref ** 2 + py_ref ** 2 + pz_ref ** 2) if pxp is not None else p_ref_mag
                print("comparison state:", total_momentum_state, "UBT:", total_momentum_ubt)

                event_momenta.append(total_momentum_state)

                x_state.append(x_ref)
                y_state.append(y_ref)
                z_state.append(z_ref)

                x_extrapolated.append(xp)
                y_extrapolated.append(yp)

        track_results.append(result)

    if n_hits > 0:
        event_prefix = f"{output_prefix_base}pair_{pair_index}_global_{global_event_number}_"
        # plot_xy_all_tracks(hit_x, hit_y, hit_z, corresponding_track_ids, track_results, global_event_number, output_prefix=event_prefix)
        # plot_xz_yz_by_detector(hit_x, hit_y, hit_z, corresponding_track_ids, track_results, global_event_number, detector_zs=detector_zs, output_prefix=event_prefix)

    return {
        "global_event_number": global_event_number,
        "success": True,
        "residuals": event_residuals,
        "momenta": event_momenta,
        "dx": event_dx,
        "dy": event_dy,
        "ref_px": event_ref_px,
        "ref_py": event_ref_py,
        "ref_pz": event_ref_pz,
        "x_state": x_state,
        "y_state": y_state,
        "z_state": z_state,
        "x_xp": x_extrapolated,
        "y_xp": y_extrapolated,
    }


def make_all_summary_plots(results, output_prefix=""):
    global_residuals = results["residuals"]
    global_momentum_for_histogram = results["momenta"]
    global_distance_x = results["dx"]
    global_distance_y = results["dy"]
    global_px = results["px"]
    global_py = results["py"]
    global_pz = results["pz"]

    state_x = results["x_state"]
    state_y = results["y_state"]
    state_z = results["z_state"]

    x_xp = results["x_xp"]
    y_xp = results["y_xp"]

    plot_residual_histogram(
        global_residuals,
        f"{output_prefix}track_hit_residuals_all.png",
        title="All matched track-hit residuals",
        bins=100,
    )

    if len(global_residuals) > 0 and len(global_momentum_for_histogram) > 0:
        plot_residual_vs_momentum_plot(
            global_residuals,
            global_momentum_for_histogram,
            ylim=(1e-2, 1e3),
            name=f"{output_prefix}residual_vs_momentum_TOKANUT_momentum_in_UBT",
        )

    if len(global_distance_x) > 0 and len(global_momentum_for_histogram) > 0:
        plot_residual_vs_momentum_plot(
            global_distance_x,
            global_momentum_for_histogram,
            ylog=False,
            ylabel="X bias between x-ted track and true hit",
            ylim=(-200, 200),
            name=f"{output_prefix}X_bias_vs_mom",
        )

    if len(global_distance_y) > 0 and len(global_momentum_for_histogram) > 0:
        plot_residual_vs_momentum_plot(
            global_distance_y,
            global_momentum_for_histogram,
            ylog=False,
            ylabel="Y bias between x-ted track and true hit",
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
            ylabel="Y bias between x-ted track and true hit",
            name=f"{output_prefix}Y_bias_vs_mom_py",
        )

    if len(global_distance_x) > 0 and len(global_px) > 0:
        plot_residual_vs_momentum_plot(
            global_distance_x,
            global_px,
            ylog=False,
            ylabel="X bias between x-ted track and true hit",
            name=f"{output_prefix}X_bias_vs_mom_px",
        )

    if len(global_distance_x) > 0 and len(global_pz) > 0:
        plot_residual_vs_momentum_plot(
            global_distance_x,
            global_pz,
            ylog=False,
            ylabel="X bias between x-ted track and true hit",
            name=f"{output_prefix}X_bias_vs_mom_pz",
        )

    if len(global_distance_x) > 0 and len(global_py) > 0:
        plot_residual_vs_momentum_plot(
            global_distance_x,
            global_py,
            ylog=False,
            ylabel="X bias between x-ted track and true hit",
            name=f"{output_prefix}X_bias_vs_mom_py",
        )

    plot_residual_vs_state_coordinate(
        state_x,
        global_distance_y,
        ylabel="Y bias between x-ted track and true hit",
        xlabel="X coordinate of state in ST",
        name=f"{output_prefix}Y_bias_vs_X_location",
    )

    plot_residual_vs_state_coordinate(
        state_y,
        global_distance_y,
        ylabel="Y bias between x-ted track and true hit",
        xlabel="Y coordinate of state in ST",
        name=f"{output_prefix}Y_bias_vs_Y_location",
    )

    plot_residual_vs_state_coordinate(
        state_x,
        global_distance_x,
        ylabel="X bias between x-ted track and true hit",
        xlabel="X coordinate of state in ST",
        name=f"{output_prefix}X_bias_vs_X_location",
    )

    plot_residual_vs_state_coordinate(
        state_y,
        global_distance_x,
        ylabel="X bias between x-ted track and true hit",
        xlabel="Y coordinate of state in ST",
        name=f"{output_prefix}X_bias_vs_Y_location",
    )

    plot_residual_vs_state_coordinate(
        x_xp,
        global_distance_x,
        ylim=(-200, 200),
        ylabel="X bias between x-ted track and true hit",
        xlabel="X coordinate of state in UBT",
        name=f"{output_prefix}X_bias_vs_X_extrapolated",
    )

    plot_residual_vs_state_coordinate(
        y_xp,
        global_distance_x,
        ylim=(-200, 200),
        ylabel="X bias between x-ted track and true hit",
        xlabel="Y coordinate of state in UBT",
        name=f"{output_prefix}X_bias_vs_Y_extrapolated",
    )

    plot_residual_vs_state_coordinate(
        x_xp,
        global_distance_y,
        ylim=(-200, 200),
        ylabel="Y bias between x-ted track and true hit",
        xlabel="X coordinate of state in UBT",
        name=f"{output_prefix}Y_bias_vs_X_extrapolated",
    )

    plot_residual_vs_state_coordinate(
        y_xp,
        global_distance_y,
        ylim=(-200, 200),
        ylabel="Y bias between x-ted track and true hit",
        xlabel="Y coordinate of state in UBT",
        name=f"{output_prefix}Y_bias_vs_Y_extrapolated",
    )

    x_true = []
    y_true = []
    for i in range(len(global_distance_y)):
        y_true.append(y_xp[i] - global_distance_y[i])
        x_true.append(x_xp[i] - global_distance_x[i])

    plot_residual_vs_state_coordinate(
        y_true,
        global_distance_y,
        ylim=(-200, 200),
        ylabel="Y bias between x-ted track and true hit",
        xlabel="Y coordinate of state in UBT",
        name=f"{output_prefix}Y_bias_vs_Y_true",
    )

    plot_residual_vs_state_coordinate(
        x_true,
        global_distance_x,
        ylim=(-200, 200),
        ylabel="X bias between x-ted track and true hit",
        xlabel="X coordinate of state in UBT",
        name=f"{output_prefix}X_bias_vs_X_true",
    )

    plot_2d_ROOT_histogram(x_true, y_true)
    plot_2d_ROOT_histogram(x_xp, y_xp, name="extrapolated")

    df = pd.DataFrame({
        "x_true": x_true,
        "y_true": y_true,
        "momentum": global_momentum_for_histogram,
    })

    low_momentum = df["momentum"] < 10

    plot_2d_ROOT_histogram(df["x_true"][low_momentum], df["y_true"][low_momentum], name="Low_momentum_XY_map")
    plot_2d_ROOT_histogram(df["x_true"][~low_momentum], df["y_true"][~low_momentum], name="High_momentum_XY_map")


# -----------------------------------------------------------------------------
# Save / load processed results
# -----------------------------------------------------------------------------
def save_analysis_results(output_prefix, residuals, momenta, dx, dy, px, py, pz, x_state, y_state, z_state, x_xp, y_xp):
    filename = f"{output_prefix}analysis_results.npz"
    np.savez(
        filename,
        residuals=np.asarray(residuals, dtype=float),
        momenta=np.asarray(momenta, dtype=float),
        dx=np.asarray(dx, dtype=float),
        dy=np.asarray(dy, dtype=float),
        px=np.asarray(px, dtype=float),
        py=np.asarray(py, dtype=float),
        pz=np.asarray(pz, dtype=float),
        x_state=np.asarray(x_state, dtype=float),
        y_state=np.asarray(y_state, dtype=float),
        z_state=np.asarray(z_state, dtype=float),
        x_xp=np.asarray(x_xp, dtype=float),
        y_xp=np.asarray(y_xp, dtype=float),
    )

    print(f"Saved processed results to: {filename}")


def load_analysis_results(filename):
    data = np.load(filename)
    return {
        "residuals": data["residuals"],
        "momenta": data["momenta"],
        "dx": data["dx"],
        "dy": data["dy"],
        "px": data["px"],
        "py": data["py"],
        "pz": data["pz"],
        "x_state": data["x_state"],
        "y_state": data["y_state"],
        "z_state": data["z_state"],
        "x_xp": data["x_xp"],
        "y_xp": data["y_xp"],
    }


# -----------------------------------------------------------------------------
# Driver
# -----------------------------------------------------------------------------
def inspect_and_plot_all_tracks_parallel(
    track_file_patterns,
    hit_file_patterns,
    track_tree_name="ship_reco_sim",
    hit_tree_name="cbmsim",
    track_branch_name="FitTracks",
    hit_branch_name="UpstreamTaggerPoint",
    track_state_pos_branch_name="PropagatedPos",
    track_state_mom_branch_name="PropagatedMom",
    detector_zs=(3000.0, 9000.0),
    max_events_with_tracks=1,
    workers=4,
    output_prefix="",
    verbose=False,
    save_processed=True,
):
    track_files = expand_patterns(track_file_patterns)
    hit_files = expand_patterns(hit_file_patterns)

    if not track_files:
        print("Error: no track files found")
        return
    if not hit_files:
        print("Error: no hit files found")
        return

    if len(track_files) != len(hit_files):
        raise RuntimeError(
            "This parallel version assumes a 1:1 mapping between reco and sim files.\n"
            f"Found {len(track_files)} track files and {len(hit_files)} hit files."
        )

    scan_args = [
        (i, track_files[i], hit_files[i], track_tree_name, hit_tree_name, track_branch_name)
        for i in range(len(track_files))
    ]

    print("\n" + "=" * 80)
    print("FIRST PASS: SCANNING FILE PAIRS FOR EVENTS WITH TRACKS")
    print("=" * 80)

    ctx = mp.get_context("spawn")
    scan_results = []

    with ProcessPoolExecutor(max_workers=min(workers, len(scan_args)), mp_context=ctx) as executor:
        futures = [executor.submit(scan_pair_for_events_with_tracks, arg) for arg in scan_args]
        for fut in as_completed(futures):
            scan_results.append(fut.result())

    scan_results.sort(key=lambda x: x["pair_index"])

    pair_offsets = {}
    running_offset = 0
    for r in scan_results:
        pair_offsets[r["pair_index"]] = running_offset
        running_offset += r["n_common"]

    candidates = []
    total_entries = 0
    for r in scan_results:
        total_entries += r["n_common"]
        offset = pair_offsets[r["pair_index"]]
        for local_evt in r["events_with_tracks"]:
            candidates.append({
                "pair_index": r["pair_index"],
                "track_file": r["track_file"],
                "hit_file": r["hit_file"],
                "local_event_number": local_evt,
                "global_event_number": offset + local_evt,
            })

    candidates.sort(key=lambda x: x["global_event_number"])

    print(f"\nTotal entries scanned: {total_entries}")
    print(f"Total candidate events with tracks: {len(candidates)}")

    if not candidates:
        print("No events with tracks found.")
        return

    selected = candidates[:max_events_with_tracks]

    print("\n" + "=" * 80)
    print("SECOND PASS: ANALYZING SELECTED EVENTS IN PARALLEL")
    print("=" * 80)

    analyze_args = [
        (
            s["pair_index"],
            s["local_event_number"],
            s["global_event_number"],
            s["track_file"],
            s["hit_file"],
            track_tree_name,
            hit_tree_name,
            track_branch_name,
            hit_branch_name,
            track_state_pos_branch_name,
            track_state_mom_branch_name,
            detector_zs,
            output_prefix,
            verbose,
        )
        for s in selected
    ]

    analysis_results = []
    with ProcessPoolExecutor(max_workers=min(workers, len(analyze_args)), mp_context=ctx) as executor:
        futures = [executor.submit(analyze_selected_event_in_pair, arg) for arg in analyze_args]
        for fut in as_completed(futures):
            analysis_results.append(fut.result())

    analysis_results.sort(key=lambda x: x["global_event_number"])

    global_residuals = []
    global_momentum_for_histogram = []
    global_distance_x = []
    global_distance_y = []
    global_px = []
    global_py = []
    global_pz = []

    x_state = []
    y_state = []
    z_state = []

    x_xp = []
    y_xp = []

    for res in analysis_results:
        if not res["success"]:
            continue
        global_residuals.extend(res["residuals"])
        global_momentum_for_histogram.extend(res["momenta"])
        global_distance_x.extend(res["dx"])
        global_distance_y.extend(res["dy"])
        global_px.extend(res["ref_px"])
        global_py.extend(res["ref_py"])
        global_pz.extend(res["ref_pz"])

        x_state.extend(res["x_state"])
        y_state.extend(res["y_state"])
        z_state.extend(res["z_state"])

        x_xp.extend(res["x_xp"])
        y_xp.extend(res["y_xp"])

    results = {
        "residuals": np.asarray(global_residuals, dtype=float),
        "momenta": np.asarray(global_momentum_for_histogram, dtype=float),
        "dx": np.asarray(global_distance_x, dtype=float),
        "dy": np.asarray(global_distance_y, dtype=float),
        "px": np.asarray(global_px, dtype=float),
        "py": np.asarray(global_py, dtype=float),
        "pz": np.asarray(global_pz, dtype=float),
        "x_state": np.asarray(x_state, dtype=float),
        "y_state": np.asarray(y_state, dtype=float),
        "z_state": np.asarray(z_state, dtype=float),
        "x_xp": np.asarray(x_xp, dtype=float),
        "y_xp": np.asarray(y_xp, dtype=float),
    }

    make_all_summary_plots(results, output_prefix=output_prefix)

    if save_processed:
        save_analysis_results(
            output_prefix,
            results["residuals"],
            results["momenta"],
            results["dx"],
            results["dy"],
            results["px"],
            results["py"],
            results["pz"],
            results["x_state"],
            results["y_state"],
            results["z_state"],
            results["x_xp"],
            results["y_xp"],
        )


def plot_from_saved_file(saved_results_file, output_prefix=""):
    print(f"Loading processed results from: {saved_results_file}")
    results = load_analysis_results(saved_results_file)
    make_all_summary_plots(results, output_prefix=output_prefix)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)

    usage = (
        "Usage:\n"
        "  Analyze ROOT files and save processed arrays:\n"
        "    python plot_all_tracks_vs_hits_parallel.py <track_files/wildcards> <hit_files/wildcards> "
        "[max_events_with_tracks] [workers] [output_prefix] [track_state_pos_branch] [track_state_mom_branch]\n\n"
        "  Replot from saved processed file only:\n"
        "    python plot_all_tracks_vs_hits_parallel.py --load <saved_results.npz> [output_prefix]\n"
    )

    if len(sys.argv) < 2:
        print(usage)
        sys.exit(1)

    if sys.argv[1] == "--load":
        if len(sys.argv) < 3:
            print(usage)
            sys.exit(1)

        saved_results_file = sys.argv[2]
        output_prefix = sys.argv[3] if len(sys.argv) > 3 else ""
        plot_from_saved_file(saved_results_file, output_prefix=output_prefix)
        sys.exit(0)

    if len(sys.argv) < 3:
        print(usage)
        sys.exit(1)

    track_patterns = [sys.argv[1]]
    hit_patterns = [sys.argv[2]]
    max_events_with_tracks = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    workers = int(sys.argv[4]) if len(sys.argv) > 4 else 4
    output_prefix = sys.argv[5] if len(sys.argv) > 5 else ""
    track_state_pos_branch_name = sys.argv[6] if len(sys.argv) > 6 else "PropagatedPos"
    track_state_mom_branch_name = sys.argv[7] if len(sys.argv) > 7 else "PropagatedMom"

    inspect_and_plot_all_tracks_parallel(
        track_file_patterns=track_patterns,
        hit_file_patterns=hit_patterns,
        track_state_pos_branch_name=track_state_pos_branch_name,
        track_state_mom_branch_name=track_state_mom_branch_name,
        max_events_with_tracks=max_events_with_tracks,
        workers=workers,
        output_prefix=output_prefix,
        verbose=False,
        save_processed=True,
    )