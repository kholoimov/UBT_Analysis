import argparse
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np

from root_utils import expand_patterns
from workers import scan_pair_for_events_with_tracks, analyze_selected_event_in_pair
from plotting import make_all_summary_plots, plot_event_detector_views
from analysis_io import save_analysis_results, load_analysis_results
from investigate_tracks import InvestigateTracks
from compare_track_momentum import CompareTrackMomentum

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
    save_detector_view=False,
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
    counter = 0
    for res in analysis_results:
        if not res["success"]:
            continue
        if save_detector_view and counter < 100:
            plot_event_detector_views(
                res["event"],
                res["global_event_number"],
                output_prefix=output_prefix,
            )
            counter += 1
        events.append(res["event"])

    make_all_summary_plots(events, output_prefix=output_prefix)

    if save_processed:
        save_analysis_results(output_prefix, events)


def plot_from_saved_file(saved_results_file, output_prefix="", save_detector_view=False):
    print(f"Loading processed results from: {saved_results_file}")
    events = load_analysis_results(saved_results_file)
    if save_detector_view:
        for event_number, event in enumerate(events):
            plot_event_detector_views(event, event_number, output_prefix=output_prefix)
            if event_number >= 100:
                break
    make_all_summary_plots(events, output_prefix=output_prefix)


def build_parser():
    parser = argparse.ArgumentParser(
        description="UBT analysis utilities",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command")

    def add_analysis_args(subparser, include_detector_view=True):
        subparser.add_argument("track_files", help="Track ROOT files or wildcard pattern")
        subparser.add_argument("hit_files", help="Hit ROOT files or wildcard pattern")
        subparser.add_argument("max_events_with_tracks", nargs="?", type=int, default=1)
        subparser.add_argument("workers", nargs="?", type=int, default=4)
        subparser.add_argument("output_prefix", nargs="?", default="")
        if include_detector_view:
            subparser.add_argument(
                "--save-detector-view",
                action="store_true",
                help="Save detector-view plots",
            )

    analyze_parser = subparsers.add_parser("analyze", help="Run the main analysis workflow")
    add_analysis_args(analyze_parser, include_detector_view=True)

    timing_parser = subparsers.add_parser("timing-resolution", help="Run the timing-resolution workflow")
    add_analysis_args(timing_parser, include_detector_view=True)

    investigate_parser = subparsers.add_parser("investigate-tracks", help="Run InvestigateTracks")
    add_analysis_args(investigate_parser, include_detector_view=False)

    compare_parser = subparsers.add_parser("compare-track-momentum", help="Compare reconstructed and true track momentum")
    add_analysis_args(compare_parser, include_detector_view=False)
    compare_parser.add_argument(
        "--save-track-example",
        action="store_true",
        help="Save one example track-state plot showing first and last fitted states",
    )

    load_parser = subparsers.add_parser("load", help="Replot from a saved processed file")
    load_parser.add_argument("saved_results_file", help="Saved analysis results pickle")
    load_parser.add_argument("output_prefix", nargs="?", default="")
    load_parser.add_argument(
        "--save-detector-view",
        action="store_true",
        help="Save detector-view plots",
    )

    return parser


def parse_args():
    parser = build_parser()
    import sys

    argv = sys.argv[1:]
    if not argv:
        parser.print_help()
        raise SystemExit(1)

    # Backward compatibility:
    # `python main.py file1 file2 ...` -> `analyze`
    # `python main.py --load file ...` -> `load`
    # `python main.py --timing-resolution ...` -> `timing-resolution`
    # `python main.py --investigate-tracks ...` -> `investigate-tracks`
    # `python main.py --compare-track-momentum ...` -> `compare-track-momentum`
    legacy_flag_map = {
        "--load": "load",
        "--timing-resolution": "timing-resolution",
        "--investigate-tracks": "investigate-tracks",
        "--compare-track-momentum": "compare-track-momentum",
    }

    if argv[0] in legacy_flag_map:
        argv = [legacy_flag_map[argv[0]], *argv[1:]]
    elif argv[0] not in {"analyze", "timing-resolution", "investigate-tracks", "compare-track-momentum", "load"}:
        argv = ["analyze", *argv]

    return parser.parse_args(argv)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    args = parse_args()

    if args.command == "load":
        plot_from_saved_file(
            args.saved_results_file,
            output_prefix=args.output_prefix,
            save_detector_view=args.save_detector_view,
        )
    elif args.command == "investigate-tracks":
        InvestigateTracks(
            track_file_patterns=[args.track_files],
            hit_file_patterns=[args.hit_files],
            max_events_with_tracks=args.max_events_with_tracks,
            workers=args.workers,
            output_prefix=args.output_prefix,
        )
    elif args.command == "compare-track-momentum":
        CompareTrackMomentum(
            track_file_patterns=[args.track_files],
            hit_file_patterns=[args.hit_files],
            max_events_with_tracks=args.max_events_with_tracks,
            workers=args.workers,
            output_prefix=args.output_prefix,
            save_track_example=args.save_track_example,
        )
    else:
        inspect_and_plot_all_tracks_parallel(
            track_file_patterns=[args.track_files],
            hit_file_patterns=[args.hit_files],
            max_events_with_tracks=args.max_events_with_tracks,
            workers=args.workers,
            output_prefix=args.output_prefix,
            save_detector_view=getattr(args, "save_detector_view", False),
            verbose=False,
            save_processed=True,
        )
