from dataclasses import dataclass, field

import numpy as np


@dataclass
class MomentumVector:
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    mcid: int = -1
    px: float = 0.0
    py: float = 0.0
    pz: float = 0.0

    def p_mag(self):
        return float(np.sqrt(self.px * self.px + self.py * self.py + self.pz * self.pz))


@dataclass
class STTrack:
    mcid: int = -1
    hits: list[tuple[float, float, float]] = field(default_factory=list)

    def add_hit(self, x, y, z):
        self.hits.append((float(x), float(y), float(z)))


@dataclass
class Residual:
    mcid: int
    dx: float
    dy: float
    dist: float
    state: MomentumVector
    hit: MomentumVector


@dataclass
class EventInformation:
    UBT_hits: list[MomentumVector] = field(default_factory=list)
    ExtraStates: list[MomentumVector] = field(default_factory=list)
    ST_tracks: list[STTrack] = field(default_factory=list)
    residuals: list[Residual] = field(default_factory=list)

    def addUBTHit(self, hit):
        self.UBT_hits.append(hit)

    def addExtraState(self, state):
        self.ExtraStates.append(state)

    def addSTTrack(self, track):
        self.ST_tracks.append(track)

    def addResidual(self, residual):
        self.residuals.append(residual)


def _collect_residual_field(events, getter):
    values = []
    for event in events:
        for residual in event.residuals:
            values.append(getter(residual))
    return np.asarray(values, dtype=float)


def extract_plot_data(events):
    return {
        "residuals": _collect_residual_field(events, lambda r: r.dist),
        "momenta": _collect_residual_field(events, lambda r: r.state.p_mag()),
        "dx": _collect_residual_field(events, lambda r: r.dx),
        "dy": _collect_residual_field(events, lambda r: r.dy),
        "px": _collect_residual_field(events, lambda r: r.state.px),
        "py": _collect_residual_field(events, lambda r: r.state.py),
        "pz": _collect_residual_field(events, lambda r: r.state.pz),
        "x_state": _collect_residual_field(events, lambda r: r.state.x),
        "y_state": _collect_residual_field(events, lambda r: r.state.y),
        "z_state": _collect_residual_field(events, lambda r: r.state.z),
        "x_hit": _collect_residual_field(events, lambda r: r.hit.x),
        "y_hit": _collect_residual_field(events, lambda r: r.hit.y),
        "z_hit": _collect_residual_field(events, lambda r: r.hit.z),
    }
