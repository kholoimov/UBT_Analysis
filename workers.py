import math
from collections import Counter

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

from model import EventInformation, MomentumVector, Residual, STTrack

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

        n_tracks = get_collection_size(fit_tracks)

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

    failure_counts = Counter()

    def record_failure(reason, message):
        failure_counts[reason] += 1
        if verbose:
            print(
                f"[DEBUG][STEP2][pair={pair_index}][global_event={global_event_number}] "
                f"{reason}: {message}"
            )

    empty_result = {
        "global_event_number": global_event_number,
        "success": False,
        "event": EventInformation(),
        "failure_counts": {},
    }

    if ok1 <= 0:
        record_failure(
            "track_entry_read_failed",
            f"could not read track entry for local event {local_event_number}",
        )
    if ok2 <= 0:
        record_failure(
            "hit_entry_read_failed",
            f"could not read hit entry for local event {local_event_number}",
        )
    if ok1 <= 0 or ok2 <= 0:
        empty_result["failure_counts"] = dict(failure_counts)
        return empty_result

    fit_tracks = get_branch_object(track_chain, track_branch_name)
    upstream_points = get_branch_object(hit_chain, hit_branch_name)
    mc_trackIDs = get_branch_object(track_chain, "fitTrack2MC")
    saved_state_pos = get_branch_object(track_chain, track_state_pos_branch_name)
    saved_state_mom = get_branch_object(track_chain, track_state_mom_branch_name)

    if fit_tracks is None:
        record_failure("missing_fit_tracks", f"branch '{track_branch_name}' is missing")
    if upstream_points is None:
        record_failure("missing_upstream_points", f"branch '{hit_branch_name}' is missing")
    if saved_state_pos is None:
        record_failure(
            "missing_saved_state_pos",
            f"branch '{track_state_pos_branch_name}' is missing",
        )
    if saved_state_mom is None:
        record_failure(
            "missing_saved_state_mom",
            f"branch '{track_state_mom_branch_name}' is missing",
        )

    if (
        fit_tracks is None
        or upstream_points is None
        or saved_state_pos is None
        or saved_state_mom is None
    ):
        empty_result["failure_counts"] = dict(failure_counts)
        return empty_result

    n_tracks = get_collection_size(fit_tracks)
    n_hits = get_collection_size(upstream_points)

    event_info = EventInformation()
    ubt_hits_by_mcid = {}

    # -------------------------------------------------------------------------
    # Read UBT hits
    # -------------------------------------------------------------------------
    mom_ubt = ROOT.TVector3()
    for i in range(n_hits):
        try:
            hit = get_collection_item(upstream_points, i)
            if hit is None:
                continue
            hit.Momentum(mom_ubt)

            ubt_hit = MomentumVector(
                x=hit.GetX(),
                y=hit.GetY(),
                z=hit.GetZ(),
                mcid=hit.GetTrackID(),
                px=mom_ubt.x(),
                py=mom_ubt.y(),
                pz=mom_ubt.z(),
            )
            event_info.addUBTHit(ubt_hit)
            ubt_hits_by_mcid.setdefault(ubt_hit.mcid, []).append(ubt_hit)

        except Exception as exc:
            record_failure("ubt_hit_read_failed", f"failed to read UBT hit {i}: {exc}")
            continue

    # -------------------------------------------------------------------------
    # Read stored track points and extrapolate the saved extra state back to UBT
    # -------------------------------------------------------------------------
    for itrk in range(n_tracks):
        try:
            track = get_collection_item(fit_tracks, itrk)
        except Exception as exc:
            record_failure("track_access_failed", f"failed to access track {itrk}: {exc}")
            continue

        if not track:
            record_failure("null_track", f"null track at index {itrk}")
            continue

        mcid = itrk
        if mc_trackIDs is not None:
            try:
                mcid_obj = get_collection_item(mc_trackIDs, itrk)
                if mcid_obj is not None:
                    mcid = int(mcid_obj)
            except Exception as exc:
                record_failure("mcid_read_failed", f"failed to read mcid for track {itrk}: {exc}")

        try:
            saved_ref_state = get_saved_reference_state(
                saved_state_pos,
                saved_state_mom,
                itrk,
                get_vector3_components,
            )
        except Exception as exc:
            record_failure(
                "saved_extra_state_read_failed",
                f"failed to read saved extra state for track {itrk}: {exc}",
            )
            continue

        saved_state = MomentumVector(
            x=saved_ref_state[0],
            y=saved_ref_state[1],
            z=saved_ref_state[2],
            mcid=mcid,
            px=saved_ref_state[3],
            py=saved_ref_state[4],
            pz=saved_ref_state[5],
        )
        event_info.addExtraState(saved_state)

        st_track = STTrack(mcid=mcid)

        try:
            all_points = get_all_track_points(track)
            for point in all_points:
                # Supports tuples/lists like (x, y, z, ...)
                # or objects with x/y/z attributes.
                if hasattr(point, "x") and hasattr(point, "y") and hasattr(point, "z"):
                    st_track.add_hit(point.x, point.y, point.z)
                elif len(point) >= 3:
                    st_track.add_hit(point[0], point[1], point[2])
        except Exception as exc:
            record_failure(
                "track_points_extract_failed",
                f"failed to extract points for track {itrk}: {exc}",
            )

        event_info.addSTTrack(st_track)

        matched_hits = ubt_hits_by_mcid.get(mcid, [])
        for hit in matched_hits:
            extrapolated_state = extrapolate_linearly_from_state(
                saved_ref_state[0],
                saved_ref_state[1],
                saved_ref_state[2],
                saved_ref_state[3],
                saved_ref_state[4],
                saved_ref_state[5],
                hit.z,
            )
            if extrapolated_state is None:
                record_failure(
                    "linear_extrapolation_failed",
                    f"track {itrk} (mcid={mcid}) could not extrapolate to z={hit.z}",
                )
                continue

            x_ref, y_ref, z_ref, px_ref, py_ref, pz_ref = extrapolated_state
            extrapolated_state_vector = MomentumVector(
                x=x_ref,
                y=y_ref,
                z=z_ref,
                mcid=mcid,
                px=px_ref,
                py=py_ref,
                pz=pz_ref,
            )
            dx = extrapolated_state_vector.x - hit.x
            dy = extrapolated_state_vector.y - hit.y
            dist = math.sqrt(dx * dx + dy * dy)

            if verbose:
                print("REF X = ", extrapolated_state_vector.x, " UBT hit = ", hit.x)
                print("REF Y = ", extrapolated_state_vector.y, " UBT hit = ", hit.y)
                print("REF Z = ", extrapolated_state_vector.z, " UBT hit = ", hit.z)
                print("=" * 80)

            event_info.addResidual(
                Residual(
                    mcid=extrapolated_state_vector.mcid,
                    dx=dx,
                    dy=dy,
                    dist=dist,
                    state=extrapolated_state_vector,
                    hit=hit,
                )
            )

    return {
        "global_event_number": global_event_number,
        "success": True,
        "event": event_info,
        "failure_counts": dict(failure_counts),
    }
