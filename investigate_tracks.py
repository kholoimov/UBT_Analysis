import math

from root_utils import (
    get_ROOT,
    get_branch_object,
    get_collection_item,
    get_collection_size,
    get_vector3_components,
)
from track_state import (
    get_all_track_points,
    get_saved_reference_state,
    extrapolate_linearly_from_state,
)

from model import EventInformation, MomentumVector, Residual, STTrack, TimingMeasurement


def InvestigateTracks(
    track_file_patterns,
    hit_file_patterns=None,
    track_tree_name="ship_reco_sim",
    max_events_with_tracks=1,
    workers=4,
    output_prefix="",
):
    del hit_file_patterns, max_events_with_tracks, workers, output_prefix
    ROOT = get_ROOT()

    track_chain = ROOT.TChain(track_tree_name)
    track_chain.Add(track_file_patterns)

    TOTAL_EVENTS = int(track_chain.GetEntries())

    print("TOTAL EVENTS: ", TOTAL_EVENTS)

    for i in range(TOTAL_EVENTS):
        i = 129465
        track_chain.GetEntry(i)

        fit_tracks = get_branch_object(track_chain, "FitTracks")

        if fit_tracks is None:
            return

        for tr in fit_tracks:
            number_of_measurements = tr.getNumPointsWithMeasurement()

            fitStatus = tr.getFitStatus()
            ndf = fitStatus.getNdf()
            chi2 = fitStatus.getChi2()

            print("Track FIT Status: chi2/NDS = ", chi2 / ndf)
            print(f"TRACK {i} with N_measurements = {number_of_measurements}")

            print(f"TRACK {i} with N states = {tr.getNumPoints()}")
            print("Track Lenght: ", fitStatus.getTrackLen())

        break
