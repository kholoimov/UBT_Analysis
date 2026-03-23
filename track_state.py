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


def extrapolate_track_linearly_to_z(track, z_target):
    """
    Linearly extrapolate a track to z_target from the first stored fitted state.

    Returns
    -------
    (x, y, z, px, py, pz) on success, otherwise None.
    """
    if track is None:
        return None

    points = get_all_track_points(track)
    if not points:
        return None

    first = points[0]
    x0, y0, z0 = first[0], first[1], first[2]

    if len(first) >= 6:
        px0, py0, pz0 = first[3], first[4], first[5]
    elif len(points) >= 2:
        second = points[1]
        px0 = second[0] - first[0]
        py0 = second[1] - first[1]
        pz0 = second[2] - first[2]
    else:
        return None

    if abs(pz0) < 1e-12:
        return None

    dz = float(z_target) - float(z0)
    x = float(x0) + float(px0 / pz0) * dz
    y = float(y0) + float(py0 / pz0) * dz

    return x, y, float(z_target), float(px0), float(py0), float(pz0)


def extrapolate_linearly_from_state(x0, y0, z0, px0, py0, pz0, z_target):
    """
    Linearly extrapolate a state to z_target.

    Returns
    -------
    (x, y, z, px, py, pz) on success, otherwise None.
    """
    if abs(pz0) < 1e-12:
        return None

    dz = float(z_target) - float(z0)
    x = float(x0) + float(px0 / pz0) * dz
    y = float(y0) + float(py0 / pz0) * dz

    return x, y, float(z_target), float(px0), float(py0), float(pz0)


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
