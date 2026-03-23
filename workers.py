import math
import numpy as np

from root_utils import get_ROOT, get_branch_object, get_vector3_components
from track_state import (
    get_all_track_points,
    get_saved_reference_state,
    track_passes_selection_from_saved_state,
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
        "event": EventInformation(),
    }

    if ok1 <= 0 or ok2 <= 0:
        return empty_result

    fit_tracks = get_branch_object(track_chain, track_branch_name)
    upstream_points = get_branch_object(hit_chain, hit_branch_name)
    mc_trackIDs = get_branch_object(track_chain, "fitTrack2MC")
    saved_state_pos = get_branch_object(track_chain, track_state_pos_branch_name)
    saved_state_mom = get_branch_object(track_chain, track_state_mom_branch_name)

    if (
        fit_tracks is None
        or upstream_points is None
        or saved_state_pos is None
        or saved_state_mom is None
    ):
        return empty_result

    try:
        n_tracks = fit_tracks.size()
        n_hits = upstream_points.size()
        n_pos = saved_state_pos.size()
        n_mom = saved_state_mom.size()
    except Exception:
        return empty_result

    n_tracks = min(n_tracks, n_pos, n_mom)

    event_info = EventInformation()

    # -------------------------------------------------------------------------
    # Read UBT hits
    # -------------------------------------------------------------------------
    mom_ubt = ROOT.TVector3()
    for i in range(n_hits):
        try:
            hit = upstream_points[i]
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

        except Exception as exc:
            if verbose:
                print(f"[WARN] Failed to read UBT hit {i}: {exc}")
            continue

    # -------------------------------------------------------------------------
    # Read saved reference states and optionally track points
    # -------------------------------------------------------------------------
    for itrk in range(n_tracks):
        try:
            track = fit_tracks[itrk]
        except Exception as exc:
            if verbose:
                print(f"[WARN] Failed to access track {itrk}: {exc}")
            continue

        if not track:
            if verbose:
                print(f"[WARN] Null track at index {itrk}")
            continue

        try:
            ref_state = get_saved_reference_state(
                saved_state_pos,
                saved_state_mom,
                itrk,
                get_vector3_components,
            )
        except Exception as exc:
            if verbose:
                print(f"[WARN] Failed to read saved state for track {itrk}: {exc}")
            continue

        mcid = itrk
        if mc_trackIDs is not None:
            try:
                mcid = mc_trackIDs[itrk]
            except Exception as exc:
                if verbose:
                    print(f"[WARN] Failed to read mcid for track {itrk}: {exc}")

        x_ref, y_ref, z_ref, px_ref, py_ref, pz_ref = ref_state

        state = MomentumVector(
            x=x_ref,
            y=y_ref,
            z=z_ref,
            mcid=mcid,
            px=px_ref,
            py=py_ref,
            pz=pz_ref,
        )
        event_info.addExtraState(state)

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
            if verbose:
                print(f"[WARN] Failed to extract points for track {itrk}: {exc}")

        event_info.addSTTrack(st_track)

    # -------------------------------------------------------------------------
    # Match ExtraStates to UBT hits by mcid and store residuals in EventInformation
    # -------------------------------------------------------------------------
    for state in event_info.ExtraStates:
        for hit in event_info.UBT_hits:
            if hit.mcid != state.mcid:
                continue

            dx = state.x - hit.x
            dy = state.y - hit.y
            dist = math.sqrt(dx * dx + dy * dy)

            if verbose:
                print("REF X = ", state.x, " UBT hit = ", hit.x)
                print("REF Y = ", state.y, " UBT hit = ", hit.y)
                print("REF Z = ", state.z, " UBT hit = ", hit.z)
                print("=" * 80)

            event_info.addResidual(
                Residual(
                    mcid=state.mcid,
                    dx=dx,
                    dy=dy,
                    dist=dist,
                    state=state,
                    hit=hit,
                )
            )

    return {
        "global_event_number": global_event_number,
        "success": True,
        "event": event_info,
    }
