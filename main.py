import sys
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np

from root_utils import expand_patterns
from workers import scan_pair_for_events_with_tracks, analyze_selected_event_in_pair
from plotting import make_all_summary_plots, plot_event_detector_views
from analysis_io import save_analysis_results, load_analysis_results

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

    events = []
    for res in analysis_results:
        if not res["success"]:
            continue
        plot_event_detector_views(
            res["event"],
            res["global_event_number"],
            output_prefix=output_prefix,
        )
        events.append(res["event"])

    make_all_summary_plots(events, output_prefix=output_prefix)

    if save_processed:
        save_analysis_results(output_prefix, events)


def plot_from_saved_file(saved_results_file, output_prefix=""):
    print(f"Loading processed results from: {saved_results_file}")
    events = load_analysis_results(saved_results_file)
    for event_number, event in enumerate(events):
        plot_event_detector_views(event, event_number, output_prefix=output_prefix)
    make_all_summary_plots(events, output_prefix=output_prefix)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)

    usage = (
        "Usage:\n"
        "  Analyze ROOT files and save processed arrays:\n"
        "    python main.py <track_files/wildcards> <hit_files/wildcards> "
        "[max_events_with_tracks] [workers] [output_prefix] [track_state_pos_branch] [track_state_mom_branch]\n\n"
        "  Replot from saved processed file only:\n"
        "    python main.py --load <saved_results.npz> [output_prefix]\n"
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
