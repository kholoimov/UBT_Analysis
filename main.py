import sys
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import Counter
import numpy as np

from root_utils import expand_patterns
from workers import scan_pair_for_events_with_tracks, analyze_selected_event_in_pair
from plotting import make_all_summary_plots, plot_event_detector_views
from analysis_io import save_analysis_results, load_analysis_results

def inspect_and_plot_all_tracks_parallel(
    track_file_patterns,
    hit_file_patterns,
    track_tree_name="cbmsim",
    hit_tree_name="cbmsim",
    track_branch_name="FitTracks",
    hit_branch_name="UpstreamTaggerPoint",
    track_state_pos_branch_name="PropagatedPos",
    track_state_mom_branch_name="PropagatedMom",
    detector_zs=None,
    max_events_with_tracks=1,
    workers=4,
    output_prefix="",
    verbose=False,
    trace_hits=False,
    save_processed=True,
):
    def debug_log(message):
        if verbose:
            print(f"[DEBUG][MAIN] {message}")

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
    debug_log(f"starting second pass for {len(selected)} selected events")

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
            trace_hits,
        )
        for s in selected
    ]

    analysis_results = []
    with ProcessPoolExecutor(max_workers=min(workers, len(analyze_args)), mp_context=ctx) as executor:
        futures = [executor.submit(analyze_selected_event_in_pair, arg) for arg in analyze_args]
        for fut in as_completed(futures):
            analysis_results.append(fut.result())
            debug_log(f"collected worker result {len(analysis_results)}/{len(analyze_args)}")
    debug_log("all worker futures completed")

    analysis_results.sort(key=lambda x: x["global_event_number"])
    step_two_failure_counts = Counter()
    for res in analysis_results:
        step_two_failure_counts.update(res.get("failure_counts", {}))

    events = []
    counter = 0
    for res in analysis_results:
        if not res["success"]:
            debug_log(f"skipping failed event {res['global_event_number']}")
            continue
        if counter < 5:
            debug_log(f"plotting detector view for event {res['global_event_number']}")
            plot_event_detector_views(
                res["event"],
                res["global_event_number"],
                output_prefix=output_prefix,
            )
            debug_log(f"finished detector view for event {res['global_event_number']}")
            counter += 1
        events.append(res["event"])

    if step_two_failure_counts:
        print("\nStep 2 failure summary:")
        for reason, count in sorted(step_two_failure_counts.items()):
            print(f"  {reason}: {count}")
    else:
        print("\nStep 2 failure summary: no failures recorded.")

    debug_log(f"starting summary plots for {len(events)} events")
    make_all_summary_plots(events, output_prefix=output_prefix)
    debug_log("finished summary plots")

    if save_processed:
        debug_log("starting save of processed results")
        save_analysis_results(output_prefix, events)
        debug_log("finished save of processed results")


def plot_from_saved_file(saved_results_file, output_prefix=""):
    print(f"Loading processed results from: {saved_results_file}")
    events = load_analysis_results(saved_results_file)
    for event_number, event in enumerate(events):
        plot_event_detector_views(event, event_number, output_prefix=output_prefix)
        if event_number >= 100: break
    make_all_summary_plots(events, output_prefix=output_prefix)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)

    usage = (
        "Usage:\n"
        "  Analyze ROOT files and save processed arrays:\n"
        "    python main.py [--debug] [--trace-hits] <track_files/wildcards> <hit_files/wildcards> "
        "[max_events_with_tracks] [workers] [output_prefix]\n\n"
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

    args = sys.argv[1:]
    debug = False
    trace_hits = False
    if "--debug" in args:
        debug = True
        args = [arg for arg in args if arg != "--debug"]
    if "--trace-hits" in args:
        trace_hits = True
        args = [arg for arg in args if arg != "--trace-hits"]

    if len(args) < 2:
        print(usage)
        sys.exit(1)

    track_patterns = [args[0]]
    hit_patterns = [args[1]]
    max_events_with_tracks = int(args[2]) if len(args) > 2 else 1
    workers = int(args[3]) if len(args) > 3 else 4
    output_prefix = args[4] if len(args) > 4 else ""

    inspect_and_plot_all_tracks_parallel(
        track_file_patterns=track_patterns,
        hit_file_patterns=hit_patterns,
        max_events_with_tracks=max_events_with_tracks,
        workers=workers,
        output_prefix=output_prefix,
        verbose=debug,
        trace_hits=trace_hits,
        save_processed=True,
    )
