import math

import matplotlib.pyplot as plt
import numpy as np

from analysis_io import build_output_path
from root_utils import (
    expand_patterns,
    get_ROOT,
    get_branch_object,
    get_collection_item,
    get_collection_size,
)
from track_state import get_all_track_points


def _get_track_id(point):
    for attr in ("GetTrackID", "getTrackID"):
        try:
            return int(getattr(point, attr)())
        except Exception:
            pass

    for attr in ("fTrackID", "trackID", "TrackID"):
        try:
            value = getattr(point, attr)
            return int(value() if callable(value) else value)
        except Exception:
            pass

    return None


def _fill_momentum(point, vector):
    try:
        point.Momentum(vector)
        return float(vector.x()), float(vector.y()), float(vector.z())
    except Exception:
        pass

    getters = (("GetPx", "GetPy", "GetPz"), ("Px", "Py", "Pz"))
    for names in getters:
        try:
            return tuple(float(getattr(point, name)()) for name in names)
        except Exception:
            pass

    return None


def _plot_histogram(values, output_name, title, xlabel):
    plt.figure(figsize=(8, 6))
    if len(values) > 0:
        plt.hist(values, bins=100, histtype="step", linewidth=1.8)
        mean_val = np.mean(values)
        rms_val = np.std(values)
        plt.axvline(mean_val, linestyle="--", label=f"mean = {mean_val:.3f}")
        plt.legend()
        print(f"{title}: N={len(values)}, mean={mean_val:.6f}, rms={rms_val:.6f}")
    else:
        print(f"{title}: no entries")

    plt.xlabel(xlabel)
    plt.ylabel("Entries")
    plt.title(title)
    plt.grid(True)
    plt.savefig(build_output_path(output_name))
    plt.close()


def _plot_scatter(true_values, reco_values, output_name, title, xlabel, ylabel):
    plt.figure(figsize=(8, 6))
    plt.scatter(true_values, reco_values, s=8)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)

    if len(true_values) > 0 and len(reco_values) > 0:
        min_val = min(float(np.min(true_values)), float(np.min(reco_values)))
        max_val = max(float(np.max(true_values)), float(np.max(reco_values)))
        plt.plot([min_val, max_val], [min_val, max_val], linestyle="--", color="black", linewidth=1.0)

    plt.savefig(build_output_path(output_name))
    plt.close()


def CompareTrackMomentum(
    track_file_patterns,
    hit_file_patterns,
    track_tree_name="ship_reco_sim",
    hit_tree_name="cbmsim",
    track_branch_name="FitTracks",
    straw_branch_name="strawtubesPoint",
    max_events_with_tracks=1,
    workers=4,
    output_prefix="",
):
    del workers

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
            "Momentum comparison assumes a 1:1 mapping between reco and sim files.\n"
            f"Found {len(track_files)} track files and {len(hit_files)} hit files."
        )

    ROOT = get_ROOT()
    true_px = []
    true_py = []
    true_pz = []
    reco_px = []
    reco_py = []
    reco_pz = []
    processed_events = 0

    for pair_index, (track_file, hit_file) in enumerate(zip(track_files, hit_files)):
        track_chain = ROOT.TChain(track_tree_name)
        hit_chain = ROOT.TChain(hit_tree_name)
        track_chain.Add(track_file)
        hit_chain.Add(hit_file)

        n_events = min(int(track_chain.GetEntries()), int(hit_chain.GetEntries()))
        for event_number in range(n_events):
            if processed_events >= max_events_with_tracks:
                break

            if track_chain.GetEntry(event_number) <= 0 or hit_chain.GetEntry(event_number) <= 0:
                continue

            fit_tracks = get_branch_object(track_chain, track_branch_name)
            straw_points = get_branch_object(hit_chain, straw_branch_name)
            if straw_points is None:
                straw_points = get_branch_object(hit_chain, "StrawtubesPoint")
            mc_track_ids = get_branch_object(track_chain, "fitTrack2MC")

            if fit_tracks is None or straw_points is None:
                continue

            straw_hits_by_mcid = {}
            n_straw = get_collection_size(straw_points)
            mom_true = ROOT.TVector3()
            for i in range(n_straw):
                hit = get_collection_item(straw_points, i)
                if hit is None:
                    continue
                mcid = _get_track_id(hit)
                if mcid is None:
                    continue
                momentum = _fill_momentum(hit, mom_true)
                if momentum is None:
                    continue
                straw_hits_by_mcid.setdefault(mcid, []).append(
                    {
                        "z": float(hit.GetZ()),
                        "px": momentum[0],
                        "py": momentum[1],
                        "pz": momentum[2],
                    }
                )

            n_tracks = get_collection_size(fit_tracks)
            found_track_in_event = False
            for itrk in range(n_tracks):
                track = get_collection_item(fit_tracks, itrk)
                if track is None:
                    continue

                mcid = itrk
                if mc_track_ids is not None:
                    mcid_obj = get_collection_item(mc_track_ids, itrk)
                    if mcid_obj is not None:
                        mcid = int(mcid_obj)

                reco_points = get_all_track_points(track)
                if not reco_points or len(reco_points[0]) < 6:
                    continue

                matched_straw_hits = straw_hits_by_mcid.get(mcid, [])
                if not matched_straw_hits:
                    continue

                first_straw_hit = min(matched_straw_hits, key=lambda hit: hit["z"])
                first_reco_state = reco_points[0]

                reco_px.append(float(first_reco_state[3]))
                reco_py.append(float(first_reco_state[4]))
                reco_pz.append(float(first_reco_state[5]))

                true_px.append(float(first_straw_hit["px"]))
                true_py.append(float(first_straw_hit["py"]))
                true_pz.append(float(first_straw_hit["pz"]))
                found_track_in_event = True

            if found_track_in_event:
                processed_events += 1

        if processed_events >= max_events_with_tracks:
            break

    reco_p = np.sqrt(np.asarray(reco_px) ** 2 + np.asarray(reco_py) ** 2 + np.asarray(reco_pz) ** 2)
    true_p = np.sqrt(np.asarray(true_px) ** 2 + np.asarray(true_py) ** 2 + np.asarray(true_pz) ** 2)

    _plot_histogram(reco_p - true_p, f"{output_prefix}momentum_resolution_p.png", "Momentum resolution", "p_reco - p_true")
    _plot_histogram(np.asarray(reco_px) - np.asarray(true_px), f"{output_prefix}momentum_resolution_px.png", "Px resolution", "px_reco - px_true")
    _plot_histogram(np.asarray(reco_py) - np.asarray(true_py), f"{output_prefix}momentum_resolution_py.png", "Py resolution", "py_reco - py_true")
    _plot_histogram(np.asarray(reco_pz) - np.asarray(true_pz), f"{output_prefix}momentum_resolution_pz.png", "Pz resolution", "pz_reco - pz_true")

    _plot_scatter(true_p, reco_p, f"{output_prefix}momentum_true_vs_reco_p.png", "Reco vs true momentum", "p_true", "p_reco")
    _plot_scatter(np.asarray(true_px), np.asarray(reco_px), f"{output_prefix}momentum_true_vs_reco_px.png", "Reco vs true px", "px_true", "px_reco")
    _plot_scatter(np.asarray(true_py), np.asarray(reco_py), f"{output_prefix}momentum_true_vs_reco_py.png", "Reco vs true py", "py_true", "py_reco")
    _plot_scatter(np.asarray(true_pz), np.asarray(reco_pz), f"{output_prefix}momentum_true_vs_reco_pz.png", "Reco vs true pz", "pz_true", "pz_reco")

    print(f"Processed events for momentum comparison: {processed_events}")
    print(f"Matched tracks for momentum comparison: {len(reco_p)}")
