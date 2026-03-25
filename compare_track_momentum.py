import math
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from analysis_io import build_output_path
from root_utils import (
    expand_patterns,
    get_ROOT,
    get_branch_object,
    get_collection_item,
    get_collection_size,
    get_vector3_components,
)
from track_state import get_all_track_points, get_saved_reference_state, extrapolate_linearly_from_state


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


def _get_point_xyz(point):
    getters = (("GetX", "GetY", "GetZ"), ("X", "Y", "Z"))
    for names in getters:
        try:
            return tuple(float(getattr(point, name)()) for name in names)
        except Exception:
            pass
    return None


def _plot_histogram(values, output_name, title, xlabel, range = None):
    plt.figure(figsize=(8, 6))
    if len(values) > 0:
        plt.hist(values, bins=40, histtype="step", linewidth=1.8, range = range)
        mean_val = np.mean(values)
        rms_val = np.std(values)
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


def _plot_resolution_vs_true_momentum(
    true_momentum,
    relative_resolution,
    output_name,
    title,
    ylabel,
    bins=20,
    max_abs_relative_resolution=0.8,
):
    plt.figure(figsize=(9, 6))
    ax = plt.gca()
    ax_hist = ax.twinx()

    true_momentum = np.asarray(true_momentum, dtype=float)
    relative_resolution = np.asarray(relative_resolution, dtype=float)

    valid = np.isfinite(true_momentum) & np.isfinite(relative_resolution)
    valid &= np.abs(relative_resolution) <= max_abs_relative_resolution
    true_momentum = true_momentum[valid]
    relative_resolution = relative_resolution[valid]
    positive_momentum = true_momentum[true_momentum > 0]
    positive_resolution = relative_resolution[true_momentum > 0]

    if len(positive_momentum) > 0:
        if np.min(positive_momentum) == np.max(positive_momentum):
            bin_edges = np.array([positive_momentum[0] * 0.9, positive_momentum[0] * 1.1])
        else:
            bin_edges = np.geomspace(np.min(positive_momentum), np.max(positive_momentum), bins + 1)

        hist_counts, hist_edges = np.histogram(positive_momentum, bins=bin_edges)
        ax_hist.stairs(
            hist_counts,
            hist_edges,
            fill=True,
            facecolor="lightgray",
            edgecolor="gray",
            alpha=0.35,
            linewidth=1.2,
            zorder=0,
        )
        ax_hist.set_ylabel("Momentum entries", color="gray")
        ax_hist.tick_params(axis="y", colors="gray")
        ax_hist.grid(False)

        bin_indices = np.digitize(positive_momentum, bin_edges) - 1
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        resolution_points = []
        resolution_errors = []
        x_points = []

        for idx, center in enumerate(bin_centers):
            in_bin = bin_indices == idx
            if not np.any(in_bin):
                continue
            values = positive_resolution[in_bin]
            if len(values) < 2:
                resolution = float(np.std(values))
                error = 0.0
            else:
                resolution = float(np.std(values))
                error = resolution / math.sqrt(2.0 * (len(values) - 1))
            x_points.append(float(center))
            resolution_points.append(resolution)
            resolution_errors.append(error)

        if x_points:
            ax.errorbar(
                x_points,
                resolution_points,
                yerr=resolution_errors,
                fmt="o-",
                color="firebrick",
                markersize=5,
                linewidth=1.5,
                capsize=2,
                zorder=3,
            )
            print(f"{title}: plotted {len(x_points)} momentum bins")
        else:
            print(f"{title}: no populated bins")
    elif len(true_momentum) > 0:
        print(f"{title}: no positive momentum entries for log binning")
    else:
        print(f"{title}: no entries")

    if len(positive_momentum) > 0:
        ax.set_xscale("log")

    ax.set_xlabel("True momentum")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    plt.savefig(build_output_path(output_name))
    plt.close()


def _plot_track_state_example(
    reco_points,
    straw_hits,
    output_name,
    title,
):
    if not reco_points:
        return

    reco_points = sorted(reco_points, key=lambda point: point[2])
    first_state = reco_points[0]
    last_state = reco_points[-1]

    straw_hits = sorted(straw_hits, key=lambda hit: hit["z"])

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True)
    views = (
        (0, "x", "X [cm]", "XZ view"),
        (1, "y", "Y [cm]", "YZ view"),
    )

    for axis, coord_index, coord_label, view_title in views:
        ax = axes[axis]
        reco_coord_index = 0 if coord_index == "x" else 1
        truth_key = coord_index

        reco_z = [point[2] for point in reco_points]
        reco_coord = [point[reco_coord_index] for point in reco_points]
        ax.plot(reco_z, reco_coord, color="tab:blue", linewidth=1.7, label="STTrack states")

        if straw_hits:
            ax.scatter(
                [hit["z"] for hit in straw_hits],
                [hit[truth_key] for hit in straw_hits],
                s=18,
                color="black",
                alpha=0.65,
                label="StrawTubes MC hits",
            )

        ax.scatter(first_state[2], first_state[reco_coord_index], s=70, color="tab:green", zorder=4, label="First state used in code")
        ax.scatter(last_state[2], last_state[reco_coord_index], s=70, color="tab:red", zorder=4, label="Last state (max Z)")

        ax.annotate(
            "first state",
            xy=(first_state[2], first_state[reco_coord_index]),
            xytext=(8, 8),
            textcoords="offset points",
            color="tab:green",
        )
        ax.annotate(
            "last state",
            xy=(last_state[2], last_state[reco_coord_index]),
            xytext=(8, -14),
            textcoords="offset points",
            color="tab:red",
        )

        ax.set_ylim(-500, 500)
        ax.set_xlabel("Z [cm]")
        ax.set_ylabel(coord_label)
        ax.set_title(view_title)
        ax.grid(True, alpha=0.3)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
    # fig.suptitle(title)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(build_output_path(output_name))
    plt.close(fig)


def _plot_detector_truth_example(
    reco_points,
    ubt_hits,
    timedet_hits,
    output_name,
    extrapolated_ubt_hit=None,
    matched_timedet_hit=None,
):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True)
    views = (
        (0, "x", "X [cm]", "XZ view"),
        (1, "y", "Y [cm]", "YZ view"),
    )
    detector_specs = (
        ("UBT", 3272.0, 10.0, 200.0, "tab:blue"),
        ("ST 1", 8407.0, 30.0, 250.0, "tab:green"),
        ("ST 2", 8607.0, 30.0, 250.0, "tab:green"),
        ("ST 3", 9307.0, 30.0, 250.0, "tab:green"),
        ("ST 4", 9507.0, 30.0, 250.0, "tab:green"),
        ("TimeDet", 9590.0, 10.0, 300.0, "tab:orange"),
    )

    sorted_reco_points = sorted(reco_points, key=lambda point: point[2]) if reco_points else []
    first_state = sorted_reco_points[0] if sorted_reco_points else None
    last_state = sorted_reco_points[-1] if sorted_reco_points else None

    for axis, coord_key, coord_label, view_title in views:
        ax = axes[axis]
        reco_coord_index = 0 if coord_key == "x" else 1

        for detector_name, detector_z, detector_zwidth, detector_half_size, detector_color in detector_specs:
            ax.add_patch(
                Rectangle(
                    (detector_z - 0.5 * detector_zwidth, -detector_half_size),
                    detector_zwidth,
                    2.0 * detector_half_size,
                    fill=False,
                    edgecolor=detector_color,
                    linewidth=1.0,
                    alpha=0.55,
                )
            )
            ax.text(
                detector_z,
                detector_half_size + 20.0,
                detector_name,
                color=detector_color,
                ha="center",
                va="bottom",
                fontsize=8,
                alpha=0.85,
            )

        if sorted_reco_points:
            ax.plot(
                [point[2] for point in sorted_reco_points],
                [point[reco_coord_index] for point in sorted_reco_points],
                color="tab:green",
                linewidth=1.7,
                label="STTrack states",
            )

        if ubt_hits:
            ax.scatter(
                [hit["z"] for hit in ubt_hits],
                [hit[coord_key] for hit in ubt_hits],
                s=24,
                color="tab:blue",
                alpha=0.8,
                label="UBT hits",
            )

        if timedet_hits:
            ax.scatter(
                [hit["z"] for hit in timedet_hits],
                [hit[coord_key] for hit in timedet_hits],
                s=24,
                color="tab:orange",
                alpha=0.8,
                label="TimeDetector points",
            )

        if first_state is not None and extrapolated_ubt_hit is not None:
            extrapolated_coord = extrapolated_ubt_hit[coord_key]
            ax.plot(
                [extrapolated_ubt_hit["z"], first_state[2]],
                [extrapolated_coord, first_state[reco_coord_index]],
                color="tab:red",
                linestyle="--",
                linewidth=1.6,
                zorder=5,
                label="Extrapolated UBT -> first ST state",
            )

        if last_state is not None and matched_timedet_hit is not None:
            timedet_coord = matched_timedet_hit[coord_key]
            ax.plot(
                [last_state[2], matched_timedet_hit["z"]],
                [last_state[reco_coord_index], timedet_coord],
                color="tab:purple",
                linestyle="-.",
                linewidth=1.6,
                zorder=5,
                label="Last ST state -> TimeDet hit",
            )

        ax.set_xlabel("Z [cm]")
        ax.set_ylabel(coord_label)
        ax.set_title(view_title)
        ax.set_ylim(-500, 500)
        ax.grid(True, alpha=0.3)

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(build_output_path(output_name))
    plt.close(fig)


def _save_example_track_plot(
    track_files,
    hit_files,
    track_tree_name,
    hit_tree_name,
    track_branch_name,
    straw_branch_name,
    output_prefix="",
):
    ROOT = get_ROOT()

    for track_file, hit_file in zip(track_files, hit_files):
        track_chain = ROOT.TChain(track_tree_name)
        hit_chain = ROOT.TChain(hit_tree_name)
        track_chain.Add(track_file)
        hit_chain.Add(hit_file)

        n_events = min(int(track_chain.GetEntries()), int(hit_chain.GetEntries()))
        for event_number in range(n_events):
            if track_chain.GetEntry(event_number) <= 0 or hit_chain.GetEntry(event_number) <= 0:
                continue

            fit_tracks = get_branch_object(track_chain, track_branch_name)
            straw_points = get_branch_object(hit_chain, straw_branch_name)
            if straw_points is None:
                straw_points = get_branch_object(hit_chain, "StrawtubesPoint")
            ubt_points = get_branch_object(hit_chain, "UpstreamTaggerPoint")
            timedet_points = get_branch_object(hit_chain, "TimeDetPoint")
            mc_track_ids = get_branch_object(track_chain, "fitTrack2MC")
            saved_state_pos = get_branch_object(track_chain, "PropagatedPos")
            saved_state_mom = get_branch_object(track_chain, "PropagatedMom")

            if fit_tracks is None or straw_points is None:
                continue

            straw_hits_by_mcid = {}
            ubt_hits_by_mcid = {}
            timedet_hits_by_mcid = {}
            ubt_hits_all = []
            timedet_hits_all = []
            n_straw = get_collection_size(straw_points)
            for i in range(n_straw):
                hit = get_collection_item(straw_points, i)
                if hit is None:
                    continue
                mcid = _get_track_id(hit)
                if mcid is None:
                    continue
                straw_hits_by_mcid.setdefault(mcid, []).append(
                    {
                        "x": float(hit.GetX()),
                        "y": float(hit.GetY()),
                        "z": float(hit.GetZ()),
                    }
                )

            for i in range(get_collection_size(ubt_points)):
                hit = get_collection_item(ubt_points, i)
                coords = _get_point_xyz(hit) if hit is not None else None
                mcid = _get_track_id(hit) if hit is not None else None
                if coords is None:
                    continue
                point_dict = {"x": coords[0], "y": coords[1], "z": coords[2]}
                ubt_hits_all.append(point_dict)
                if mcid is not None:
                    ubt_hits_by_mcid.setdefault(mcid, []).append(point_dict)

            for i in range(get_collection_size(timedet_points)):
                hit = get_collection_item(timedet_points, i)
                coords = _get_point_xyz(hit) if hit is not None else None
                mcid = _get_track_id(hit) if hit is not None else None
                if coords is None:
                    continue
                point_dict = {"x": coords[0], "y": coords[1], "z": coords[2]}
                timedet_hits_all.append(point_dict)
                if mcid is not None:
                    timedet_hits_by_mcid.setdefault(mcid, []).append(point_dict)

            n_tracks = get_collection_size(fit_tracks)
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
                if len(reco_points) < 2:
                    continue

                matched_straw_hits = straw_hits_by_mcid.get(mcid, [])
                if not matched_straw_hits:
                    continue

                matched_ubt_hits = ubt_hits_by_mcid.get(mcid, [])
                matched_timedet_hits = timedet_hits_by_mcid.get(mcid, [])
                last_ubt_hit = max(matched_ubt_hits, key=lambda hit: hit["z"]) if matched_ubt_hits else None
                matched_timedet_hit = min(matched_timedet_hits, key=lambda hit: abs(hit["z"] - 105.0)) if matched_timedet_hits else None
                extrapolated_ubt_hit = None

                if (
                    last_ubt_hit is not None
                    and saved_state_pos is not None
                    and saved_state_mom is not None
                ):
                    try:
                        saved_ref_state = get_saved_reference_state(
                            saved_state_pos,
                            saved_state_mom,
                            itrk,
                            get_vector3_components,
                        )
                        extrapolated_state = extrapolate_linearly_from_state(
                            saved_ref_state[0],
                            saved_ref_state[1],
                            saved_ref_state[2],
                            saved_ref_state[3],
                            saved_ref_state[4],
                            saved_ref_state[5],
                            last_ubt_hit["z"],
                        )
                        if extrapolated_state is not None:
                            extrapolated_ubt_hit = {
                                "x": float(extrapolated_state[0]),
                                "y": float(extrapolated_state[1]),
                                "z": float(extrapolated_state[2]),
                            }
                    except Exception:
                        extrapolated_ubt_hit = None

                title = f"Track state example: event {event_number}, track {itrk}, MCID {mcid}"
                output_name = f"{output_prefix}track_state_example_event_{event_number}_track_{itrk}.png"
                _plot_track_state_example(reco_points, matched_straw_hits, output_name, title)
                print(f"Saved example track-state plot: {output_name}")

                detector_output_name = f"{output_prefix}detector_truth_example_event_{event_number}_track_{itrk}.png"
                _plot_detector_truth_example(
                    reco_points,
                    ubt_hits_all,
                    timedet_hits_all,
                    detector_output_name,
                    extrapolated_ubt_hit=extrapolated_ubt_hit,
                    matched_timedet_hit=matched_timedet_hit,
                )
                print(f"Saved detector-truth example plot: {detector_output_name}")
                return

    print("No matched track found for example track-state plot.")


def _process_momentum_chunk(args):
    (
        chunk_index,
        track_file,
        hit_file,
        track_tree_name,
        hit_tree_name,
        track_branch_name,
        straw_branch_name,
        event_start,
        event_end,
    ) = args

    ROOT = get_ROOT()
    track_chain = ROOT.TChain(track_tree_name)
    hit_chain = ROOT.TChain(hit_tree_name)
    track_chain.Add(track_file)
    hit_chain.Add(hit_file)

    true_px = []
    true_py = []
    true_pz = []
    reco_px = []
    reco_py = []
    reco_pz = []
    true_px_last = []
    true_py_last = []
    true_pz_last = []
    reco_px_last = []
    reco_py_last = []
    reco_pz_last = []
    processed_events = 0

    for event_number in range(event_start, event_end):
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
            last_straw_hit = max(matched_straw_hits, key=lambda hit: hit["z"])
            first_reco_state = reco_points[0]
            last_reco_state = max(reco_points, key=lambda point: point[2])

            reco_px.append(float(first_reco_state[3]))
            reco_py.append(float(first_reco_state[4]))
            reco_pz.append(float(first_reco_state[5]))

            true_px.append(float(first_straw_hit["px"]))
            true_py.append(float(first_straw_hit["py"]))
            true_pz.append(float(first_straw_hit["pz"]))

            reco_px_last.append(float(last_reco_state[3]))
            reco_py_last.append(float(last_reco_state[4]))
            reco_pz_last.append(float(last_reco_state[5]))

            true_px_last.append(float(last_straw_hit["px"]))
            true_py_last.append(float(last_straw_hit["py"]))
            true_pz_last.append(float(last_straw_hit["pz"]))
            found_track_in_event = True

        if found_track_in_event:
            processed_events += 1

    return {
        "chunk_index": chunk_index,
        "processed_events": processed_events,
        "true_px": true_px,
        "true_py": true_py,
        "true_pz": true_pz,
        "reco_px": reco_px,
        "reco_py": reco_py,
        "reco_pz": reco_pz,
        "true_px_last": true_px_last,
        "true_py_last": true_py_last,
        "true_pz_last": true_pz_last,
        "reco_px_last": reco_px_last,
        "reco_py_last": reco_py_last,
        "reco_pz_last": reco_pz_last,
    }


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
    save_track_example=False,
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
    true_px_last = []
    true_py_last = []
    true_pz_last = []
    reco_px_last = []
    reco_py_last = []
    reco_pz_last = []
    processed_events = 0
    chunk_args = []
    chunk_index = 0
    chunk_size_divisor = max(1, workers)
    for track_file, hit_file in zip(track_files, hit_files):
        track_chain = ROOT.TChain(track_tree_name)
        hit_chain = ROOT.TChain(hit_tree_name)
        track_chain.Add(track_file)
        hit_chain.Add(hit_file)
        n_events = min(int(track_chain.GetEntries()), int(hit_chain.GetEntries()))
        if n_events <= 0:
            continue

        chunk_size = max(1, math.ceil(n_events / chunk_size_divisor))
        for event_start in range(0, n_events, chunk_size):
            event_end = min(event_start + chunk_size, n_events)
            chunk_args.append(
                (
                    chunk_index,
                    track_file,
                    hit_file,
                    track_tree_name,
                    hit_tree_name,
                    track_branch_name,
                    straw_branch_name,
                    event_start,
                    event_end,
                )
            )
            chunk_index += 1

    chunk_results = []
    if chunk_args:
        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=min(workers, len(chunk_args)), mp_context=ctx) as executor:
            futures = [executor.submit(_process_momentum_chunk, arg) for arg in chunk_args]
            for fut in as_completed(futures):
                chunk_results.append(fut.result())

    chunk_results.sort(key=lambda item: item["chunk_index"])

    for result in chunk_results:
        if processed_events >= max_events_with_tracks:
            break

        processed_events += result["processed_events"]
        true_px.extend(result["true_px"])
        true_py.extend(result["true_py"])
        true_pz.extend(result["true_pz"])
        reco_px.extend(result["reco_px"])
        reco_py.extend(result["reco_py"])
        reco_pz.extend(result["reco_pz"])
        true_px_last.extend(result["true_px_last"])
        true_py_last.extend(result["true_py_last"])
        true_pz_last.extend(result["true_pz_last"])
        reco_px_last.extend(result["reco_px_last"])
        reco_py_last.extend(result["reco_py_last"])
        reco_pz_last.extend(result["reco_pz_last"])

    reco_p = np.sqrt(np.asarray(reco_px) ** 2 + np.asarray(reco_py) ** 2 + np.asarray(reco_pz) ** 2)
    true_p = np.sqrt(np.asarray(true_px) ** 2 + np.asarray(true_py) ** 2 + np.asarray(true_pz) ** 2)
    reco_p_last = np.sqrt(np.asarray(reco_px_last) ** 2 + np.asarray(reco_py_last) ** 2 + np.asarray(reco_pz_last) ** 2)
    true_p_last = np.sqrt(np.asarray(true_px_last) ** 2 + np.asarray(true_py_last) ** 2 + np.asarray(true_pz_last) ** 2)
    rel_p = np.divide(reco_p - true_p, true_p, out=np.zeros_like(reco_p), where=true_p != 0)
    rel_px = np.divide(np.asarray(reco_px) - np.asarray(true_px), np.asarray(true_px), out=np.zeros_like(np.asarray(reco_px)), where=np.asarray(true_px) != 0)
    rel_py = np.divide(np.asarray(reco_py) - np.asarray(true_py), np.asarray(true_py), out=np.zeros_like(np.asarray(reco_py)), where=np.asarray(true_py) != 0)
    rel_pz = np.divide(np.asarray(reco_pz) - np.asarray(true_pz), np.asarray(true_pz), out=np.zeros_like(np.asarray(reco_pz)), where=np.asarray(true_pz) != 0)
    rel_p_last = np.divide(reco_p_last - true_p_last, true_p_last, out=np.zeros_like(reco_p_last), where=true_p_last != 0)
    rel_px_last = np.divide(np.asarray(reco_px_last) - np.asarray(true_px_last), np.asarray(true_px_last), out=np.zeros_like(np.asarray(reco_px_last)), where=np.asarray(true_px_last) != 0)
    rel_py_last = np.divide(np.asarray(reco_py_last) - np.asarray(true_py_last), np.asarray(true_py_last), out=np.zeros_like(np.asarray(reco_py_last)), where=np.asarray(true_py_last) != 0)
    rel_pz_last = np.divide(np.asarray(reco_pz_last) - np.asarray(true_pz_last), np.asarray(true_pz_last), out=np.zeros_like(np.asarray(reco_pz_last)), where=np.asarray(true_pz_last) != 0)

    histograms_range = {
        "x": [-0.15, 0.15],
        "y": [-0.15, 0.15],
        "z": [-1, 5],
        "mag": [-1, 5],
        "rel": [-1, 1],
    }

    _plot_histogram(reco_p - true_p, f"{output_prefix}momentum_resolution_p.png", "Momentum resolution", "p_reco - p_true", range = histograms_range["mag"])
    _plot_histogram(np.asarray(reco_px) - np.asarray(true_px), f"{output_prefix}momentum_resolution_px.png", "Px resolution", "px_reco - px_true", range = histograms_range["x"])
    _plot_histogram(np.asarray(reco_py) - np.asarray(true_py), f"{output_prefix}momentum_resolution_py.png", "Py resolution", "py_reco - py_true", range = histograms_range["y"])
    _plot_histogram(np.asarray(reco_pz) - np.asarray(true_pz), f"{output_prefix}momentum_resolution_pz.png", "Pz resolution", "pz_reco - pz_true", range = histograms_range["z"])
    _plot_histogram(rel_p, f"{output_prefix}momentum_relative_resolution_p.png", "Relative momentum resolution", "(p_reco - p_true) / p_true", range = histograms_range["rel"])
    _plot_histogram(rel_px, f"{output_prefix}momentum_relative_resolution_px.png", "Relative px resolution", "(px_reco - px_true) / px_true", range = histograms_range["rel"])
    _plot_histogram(rel_py, f"{output_prefix}momentum_relative_resolution_py.png", "Relative py resolution", "(py_reco - py_true) / py_true", range = histograms_range["rel"])
    _plot_histogram(rel_pz, f"{output_prefix}momentum_relative_resolution_pz.png", "Relative pz resolution", "(pz_reco - pz_true) / pz_true", range = histograms_range["rel"])
    _plot_resolution_vs_true_momentum(true_p, rel_p, f"{output_prefix}momentum_relative_resolution_vs_true_p.png", "Relative momentum resolution vs true momentum", "RMS[(p_reco - p_true) / p_true]")
    _plot_resolution_vs_true_momentum(true_p, rel_px, f"{output_prefix}momentum_relative_resolution_vs_true_px.png", "Relative px resolution vs true momentum", "RMS[(px_reco - px_true) / px_true]")
    _plot_resolution_vs_true_momentum(true_p, rel_py, f"{output_prefix}momentum_relative_resolution_vs_true_py.png", "Relative py resolution vs true momentum", "RMS[(py_reco - py_true) / py_true]")
    _plot_resolution_vs_true_momentum(true_p, rel_pz, f"{output_prefix}momentum_relative_resolution_vs_true_pz.png", "Relative pz resolution vs true momentum", "RMS[(pz_reco - pz_true) / pz_true]")

    _plot_scatter(true_p, reco_p, f"{output_prefix}momentum_true_vs_reco_p.png", "Reco vs true momentum", "p_true", "p_reco")
    _plot_scatter(np.asarray(true_px), np.asarray(reco_px), f"{output_prefix}momentum_true_vs_reco_px.png", "Reco vs true px", "px_true", "px_reco")
    _plot_scatter(np.asarray(true_py), np.asarray(reco_py), f"{output_prefix}momentum_true_vs_reco_py.png", "Reco vs true py", "py_true", "py_reco")
    _plot_scatter(np.asarray(true_pz), np.asarray(reco_pz), f"{output_prefix}momentum_true_vs_reco_pz.png", "Reco vs true pz", "pz_true", "pz_reco")

    _plot_histogram(reco_p_last - true_p_last, f"{output_prefix}momentum_resolution_last_p.png", "Momentum resolution at last state", "p_reco(last) - p_true(last)", range = histograms_range["mag"])
    _plot_histogram(np.asarray(reco_px_last) - np.asarray(true_px_last), f"{output_prefix}momentum_resolution_last_px.png", "Px resolution at last state", "px_reco(last) - px_true(last)", range = histograms_range["x"])
    _plot_histogram(np.asarray(reco_py_last) - np.asarray(true_py_last), f"{output_prefix}momentum_resolution_last_py.png", "Py resolution at last state", "py_reco(last) - py_true(last)", range = histograms_range["y"])
    _plot_histogram(np.asarray(reco_pz_last) - np.asarray(true_pz_last), f"{output_prefix}momentum_resolution_last_pz.png", "Pz resolution at last state", "pz_reco(last) - pz_true(last)", range = histograms_range["z"])
    _plot_histogram(rel_p_last, f"{output_prefix}momentum_relative_resolution_last_p.png", "Relative momentum resolution at last state", "(p_reco(last) - p_true(last)) / p_true(last)", range = histograms_range["rel"])
    _plot_histogram(rel_px_last, f"{output_prefix}momentum_relative_resolution_last_px.png", "Relative px resolution at last state", "(px_reco(last) - px_true(last)) / px_true(last)", range = histograms_range["rel"])
    _plot_histogram(rel_py_last, f"{output_prefix}momentum_relative_resolution_last_py.png", "Relative py resolution at last state", "(py_reco(last) - py_true(last)) / py_true(last)", range = histograms_range["rel"])
    _plot_histogram(rel_pz_last, f"{output_prefix}momentum_relative_resolution_last_pz.png", "Relative pz resolution at last state", "(pz_reco(last) - pz_true(last)) / pz_true(last)", range = histograms_range["rel"])
    _plot_resolution_vs_true_momentum(true_p_last, rel_p_last, f"{output_prefix}momentum_relative_resolution_vs_true_last_p.png", "Relative momentum resolution vs true momentum at last state", "RMS[(p_reco(last) - p_true(last)) / p_true(last)]")
    _plot_resolution_vs_true_momentum(true_p_last, rel_px_last, f"{output_prefix}momentum_relative_resolution_vs_true_last_px.png", "Relative px resolution vs true momentum at last state", "RMS[(px_reco(last) - px_true(last)) / px_true(last)]")
    _plot_resolution_vs_true_momentum(true_p_last, rel_py_last, f"{output_prefix}momentum_relative_resolution_vs_true_last_py.png", "Relative py resolution vs true momentum at last state", "RMS[(py_reco(last) - py_true(last)) / py_true(last)]")
    _plot_resolution_vs_true_momentum(true_p_last, rel_pz_last, f"{output_prefix}momentum_relative_resolution_vs_true_last_pz.png", "Relative pz resolution vs true momentum at last state", "RMS[(pz_reco(last) - pz_true(last)) / pz_true(last)]")

    _plot_scatter(true_p_last, reco_p_last, f"{output_prefix}momentum_true_vs_reco_last_p.png", "Reco vs true momentum at last state", "p_true(last)", "p_reco(last)")
    _plot_scatter(np.asarray(true_px_last), np.asarray(reco_px_last), f"{output_prefix}momentum_true_vs_reco_last_px.png", "Reco vs true px at last state", "px_true(last)", "px_reco(last)")
    _plot_scatter(np.asarray(true_py_last), np.asarray(reco_py_last), f"{output_prefix}momentum_true_vs_reco_last_py.png", "Reco vs true py at last state", "py_true(last)", "py_reco(last)")
    _plot_scatter(np.asarray(true_pz_last), np.asarray(reco_pz_last), f"{output_prefix}momentum_true_vs_reco_last_pz.png", "Reco vs true pz at last state", "pz_true(last)", "pz_reco(last)")

    if save_track_example:
        _save_example_track_plot(
            track_files,
            hit_files,
            track_tree_name,
            hit_tree_name,
            track_branch_name,
            straw_branch_name,
            output_prefix=output_prefix,
        )

    print(f"Processed events for momentum comparison: {processed_events}")
    print(f"Matched tracks for momentum comparison: {len(reco_p)}")
