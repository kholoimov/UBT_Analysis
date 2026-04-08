import math

from root_utils import (
    get_ROOT,
    get_branch_object,
    get_collection_item,
    get_collection_size,
)
from track_state import (
    get_all_track_points,
    extrapolate_linearly_from_state,
)

from model import EventInformation, MomentumVector, Residual, STTrack, TimingMeasurement

MUON_MASS_GEV = 0.1056583755
SPEED_OF_LIGHT_CM_PER_NS = 29.9792458


def _get_point_time_ns(point):
    for attr in ("fTime",):
        try:
            value = getattr(point, attr)
            return float(value() if callable(value) else value)
        except Exception:
            pass

    for method_name in ("GetTime",):
        try:
            return float(getattr(point, method_name)())
        except Exception:
            pass

    return None


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


def _calculate_beta_from_momentum(px, py, pz, mass_gev=MUON_MASS_GEV):
    p = math.sqrt(px * px + py * py + pz * pz)
    if p <= 0.0:
        return None

    energy = math.sqrt(p * p + mass_gev * mass_gev)
    if energy <= 0.0:
        return None

    beta = p / energy
    if beta <= 0.0 or beta >= 1.0e6:
        return None
    return beta

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

    empty_result = {
        "global_event_number": global_event_number,
        "success": False,
        "event": EventInformation(),
    }

    if ok1 <= 0 or ok2 <= 0:
        return empty_result

    fit_tracks = get_branch_object(track_chain, track_branch_name)
    upstream_points = get_branch_object(hit_chain, hit_branch_name)
    straw_points = get_branch_object(hit_chain, "strawtubesPoint")
    if straw_points is None:
        straw_points = get_branch_object(hit_chain, "StrawtubesPoint")
    timedet_points = get_branch_object(hit_chain, "TimeDetPoint")
    mc_trackIDs = get_branch_object(track_chain, "fitTrack2MC")

    if fit_tracks is None or upstream_points is None:
        return empty_result

    n_tracks = get_collection_size(fit_tracks)
    n_hits = get_collection_size(upstream_points)
    n_straw_hits = get_collection_size(straw_points)
    n_timedet_hits = get_collection_size(timedet_points)

    event_info = EventInformation()
    ubt_hits_by_mcid = {}
    straw_hits_by_mcid = {}
    timedet_hits_by_mcid = {}

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
                mcid=_get_track_id(hit),
                px=mom_ubt.x(),
                py=mom_ubt.y(),
                pz=mom_ubt.z(),
                time_ns=_get_point_time_ns(hit),
            )
            event_info.addUBTHit(ubt_hit)
            if ubt_hit.mcid is not None:
                ubt_hits_by_mcid.setdefault(ubt_hit.mcid, []).append(ubt_hit)

        except Exception as exc:
            if verbose:
                print(f"[WARN] Failed to read UBT hit {i}: {exc}")
            continue

    for i in range(n_straw_hits):
        try:
            hit = get_collection_item(straw_points, i)
            if hit is None:
                continue

            mcid = _get_track_id(hit)
            if mcid is None:
                continue

            straw_hit = MomentumVector(
                x=hit.GetX(),
                y=hit.GetY(),
                z=hit.GetZ(),
                mcid=mcid,
                time_ns=_get_point_time_ns(hit),
            )
            straw_hits_by_mcid.setdefault(mcid, []).append(straw_hit)
        except Exception as exc:
            if verbose:
                print(f"[WARN] Failed to read StrawTubes hit {i}: {exc}")
            continue

    for i in range(n_timedet_hits):
        try:
            hit = get_collection_item(timedet_points, i)
            if hit is None:
                continue

            mcid = _get_track_id(hit)
            if mcid is None:
                continue

            timedet_hit = MomentumVector(
                x=hit.GetX(),
                y=hit.GetY(),
                z=hit.GetZ(),
                mcid=mcid,
                time_ns=_get_point_time_ns(hit),
            )
            timedet_hits_by_mcid.setdefault(mcid, []).append(timedet_hit)
        except Exception as exc:
            if verbose:
                print(f"[WARN] Failed to read TimeDet hit {i}: {exc}")
            continue

    # -------------------------------------------------------------------------
    # Read stored track points and extrapolate the first ST state back to UBT
    # -------------------------------------------------------------------------
    for itrk in range(n_tracks):
        try:
            track = get_collection_item(fit_tracks, itrk)
        except Exception as exc:
            if verbose:
                print(f"[WARN] Failed to access track {itrk}: {exc}")
            continue

        if not track:
            if verbose:
                print(f"[WARN] Null track at index {itrk}")
            continue

        mcid = itrk
        if mc_trackIDs is not None:
            try:
                mcid_obj = get_collection_item(mc_trackIDs, itrk)
                if mcid_obj is not None:
                    mcid = int(mcid_obj)
            except Exception as exc:
                if verbose:
                    print(f"[WARN] Failed to read mcid for track {itrk}: {exc}")

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

        if all_points:
            first_st_state = all_points[0]
            last_st_state = max(all_points, key=lambda point: point[2])
        else:
            first_st_state = None
            last_st_state = None

        matched_hits = ubt_hits_by_mcid.get(mcid, [])
        last_ubt_hit = max(matched_hits, key=lambda hit: hit.z) if matched_hits else None
        extrapolated_last_ubt_hit = None
        for hit in matched_hits:
            if first_st_state is None or len(first_st_state) < 6:
                if verbose:
                    print(
                        f"[WARN] Linear extrapolation from first ST state failed for track {itrk} "
                        f"(mcid={mcid}): no valid first state"
                    )
                break

            extrapolated_state = extrapolate_linearly_from_state(
                first_st_state[0],
                first_st_state[1],
                first_st_state[2],
                first_st_state[3],
                first_st_state[4],
                first_st_state[5],
                hit.z,
            )
            if extrapolated_state is None:
                if verbose:
                    print(
                        f"[WARN] Linear extrapolation from first ST state failed for track {itrk} "
                        f"(mcid={mcid}) to z={hit.z}"
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

            if last_ubt_hit is not None and hit is last_ubt_hit:
                extrapolated_last_ubt_hit = extrapolated_state_vector

        timedet_hit = None
        matched_timedet_hits = timedet_hits_by_mcid.get(mcid, [])
        if matched_timedet_hits:
            timedet_hit = min(matched_timedet_hits, key=lambda hit: abs(hit.z - 105.0))

        if (
            last_ubt_hit is not None
            and timedet_hit is not None
            and last_ubt_hit.time_ns is not None
            and timedet_hit.time_ns is not None
            and first_st_state is not None
            and last_st_state is not None
            and len(first_st_state) >= 6
            and extrapolated_last_ubt_hit is not None
        ):
            true_time_ns = timedet_hit.time_ns - last_ubt_hit.time_ns

            first_state_vector = MomentumVector(
                x=first_st_state[0],
                y=first_st_state[1],
                z=first_st_state[2],
                mcid=mcid,
                px=first_st_state[3],
                py=first_st_state[4],
                pz=first_st_state[5],
            )
            last_state_vector = MomentumVector(
                x=last_st_state[0],
                y=last_st_state[1],
                z=last_st_state[2],
                mcid=mcid,
                px=last_st_state[3] if len(last_st_state) >= 6 else 0.0,
                py=last_st_state[4] if len(last_st_state) >= 6 else 0.0,
                pz=last_st_state[5] if len(last_st_state) >= 6 else 0.0,
            )

            beta = _calculate_beta_from_momentum(
                first_state_vector.px,
                first_state_vector.py,
                first_state_vector.pz,
            )

            fit_status = None
            track_length_cm = None
            try:
                fit_status = track.getFitStatus()
            except Exception:
                fit_status = None

            if fit_status is not None:
                try:
                    track_length_cm = float(fit_status.getTrackLen())
                except Exception:
                    track_length_cm = None

            if beta is not None and beta > 0.0 and track_length_cm is not None and track_length_cm > 0.0:
                dx = first_state_vector.x - extrapolated_last_ubt_hit.x
                dy = first_state_vector.y - extrapolated_last_ubt_hit.y
                dz = first_state_vector.z - extrapolated_last_ubt_hit.z
                ubt_to_first_state_distance_cm = math.sqrt(dx * dx + dy * dy + dz * dz)
                dx_end = timedet_hit.x - last_state_vector.x
                dy_end = timedet_hit.y - last_state_vector.y
                dz_end = timedet_hit.z - last_state_vector.z
                last_state_to_timedet_distance_cm = math.sqrt(dx_end * dx_end + dy_end * dy_end + dz_end * dz_end)
                total_distance_cm = (
                    ubt_to_first_state_distance_cm
                    + track_length_cm
                    + last_state_to_timedet_distance_cm
                )

                reco_time_ns = total_distance_cm / (beta * SPEED_OF_LIGHT_CM_PER_NS)

                delta_time_ns = reco_time_ns - true_time_ns
                
                if (delta_time_ns < 0):
                    print("\n")
                    print("=" * 100)
                    print("UBT hit X: ", last_ubt_hit.x, " Y: ", last_ubt_hit.y, " Z: ", last_ubt_hit.z, " TIME: ", last_ubt_hit.time_ns)
                    print("TimeDet hit X: ", timedet_hit.x, " Y: ", timedet_hit.y, " Z: ", timedet_hit.z, " TIME: ", timedet_hit.time_ns)

                    print("UBT RECO HIT X: ", extrapolated_last_ubt_hit.x, " Y: ", extrapolated_last_ubt_hit.y, " Z: ", extrapolated_last_ubt_hit.z)
                    print("First ST state X: ", first_state_vector.x, " Y: ", first_state_vector.y, " Z: ", first_state_vector.z)
                    print("ST momentum PX: ", first_state_vector.px, " PY: ", first_state_vector.py, " PZ: ", first_state_vector.pz)
                    print("Track length in ST [cm]: ", track_length_cm)
                    print("Distance UBT -> first ST state [cm]: ", ubt_to_first_state_distance_cm)
                    print("Distance last ST state -> TimeDet [cm]: ", last_state_to_timedet_distance_cm)
                    print("Total distance [cm]: ", total_distance_cm)
                    print("Beta: ", beta)

                    print("TRUE TIME: ", true_time_ns)
                    print("RECO TIME: ", reco_time_ns)
                    print("=" * 100)
                    print("\n")

                event_info.addTimingMeasurement(
                    TimingMeasurement(
                        mcid=mcid,
                        true_time_ns=true_time_ns,
                        reco_time_ns=reco_time_ns,
                        delta_time_ns=delta_time_ns,
                        distance_cm=total_distance_cm,
                        beta=beta,
                        ubt_hit=last_ubt_hit,
                        st_state=first_state_vector,
                        extrapolated_hit=extrapolated_last_ubt_hit,
                    )
                )

    return {
        "global_event_number": global_event_number,
        "success": True,
        "event": event_info,
    }
