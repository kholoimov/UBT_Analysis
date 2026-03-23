import math


def get_all_track_points(track):
    """
    Read stored fitted states from the GenFit track object.
    Used only for visualisation / station coverage checks.
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


def propagate_track_to_z(track, z_target, use_closest_state=True):
    """
    Propagate a fitted GENFIT track to a plane z = z_target.

    Returns
    -------
    (x, y, z, px, py, pz) on success, otherwise None.

    Notes
    -----
    This uses the track's existing fitted representation when available,
    instead of reconstructing a new rep from a guessed PDG.
    """
    try:
        import ROOT
    except Exception:
        return None

    if track is None:
        return None

    try:
        fit_status = track.getFitStatus()
        if not fit_status or not fit_status.isFitConverged():
            return None
    except Exception:
        return None

    try:
        rep = track.getCardinalRep()
    except Exception:
        rep = None

    if rep is None:
        return None

    try:
        n_points = int(track.getNumPoints())
    except Exception:
        return None

    if n_points <= 0:
        return None

    source_pos = None
    source_mom = None
    best_dz = None

    if use_closest_state:
        for i in range(n_points):
            try:
                state = track.getFittedState(i)
                pos = state.getPos()
                mom = state.getMom()
            except Exception:
                continue

            try:
                dz = abs(float(pos.Z()) - float(z_target))
            except Exception:
                continue

            if best_dz is None or dz < best_dz:
                best_dz = dz
                source_pos = ROOT.TVector3(float(pos.X()), float(pos.Y()), float(pos.Z()))
                source_mom = ROOT.TVector3(float(mom.X()), float(mom.Y()), float(mom.Z()))
    else:
        try:
            state = track.getFittedState()
            pos = state.getPos()
            mom = state.getMom()
            source_pos = ROOT.TVector3(float(pos.X()), float(pos.Y()), float(pos.Z()))
            source_mom = ROOT.TVector3(float(mom.X()), float(mom.Y()), float(mom.Z()))
        except Exception:
            return None

    if source_pos is None or source_mom is None:
        return None

    try:
        state_on_plane = ROOT.genfit.StateOnPlane(rep)
        rep.setPosMom(state_on_plane, source_pos, source_mom)
    except Exception:
        return None

    try:
        plane = ROOT.genfit.SharedPlanePtr(
            ROOT.genfit.DetPlane(
                ROOT.TVector3(0.0, 0.0, float(z_target)),
                ROOT.TVector3(1.0, 0.0, 0.0),
                ROOT.TVector3(0.0, 1.0, 0.0),
            )
        )
        rep.extrapolateToPlane(state_on_plane, plane)
    except Exception:
        return None

    try:
        pos = state_on_plane.getPos()
        mom = state_on_plane.getMom()
        return (
            float(pos.X()),
            float(pos.Y()),
            float(pos.Z()),
            float(mom.X()),
            float(mom.Y()),
            float(mom.Z()),
        )
    except Exception:
        return None


def get_saved_reference_state(pos_branch, mom_branch, track_index, get_vector3_components):
    """
    Read saved per-track extra state from separate TVector3 branches.
    Returns:
        x, y, z, px, py, pz
    """
    pos = pos_branch[track_index]
    mom = mom_branch[track_index]

    x, y, z = get_vector3_components(pos)
    px, py, pz = get_vector3_components(mom)

    return x, y, z, px, py, pz


def has_hits_in_all_stations(
    all_points,
    station_z=(8407.0, 8607.0, 9307.0, 9507.0),
    tolerance=100.0
):
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
    No propagation is used.
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
