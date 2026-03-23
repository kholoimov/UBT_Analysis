import glob


def get_ROOT():
    import ROOT
    return ROOT


def expand_patterns(patterns):
    files = []
    for pattern in patterns:
        matched = glob.glob(pattern)
        if matched:
            files.extend(sorted(matched))
        else:
            print(f"Warning: no files matched pattern '{pattern}'")
    return list(dict.fromkeys(files))


def get_branch_object(tree_or_chain, branch_name):
    try:
        return getattr(tree_or_chain, branch_name)
    except Exception:
        return None


def get_vector3_components(v):
    try:
        return float(v.X()), float(v.Y()), float(v.Z())
    except Exception:
        return float(v.x()), float(v.y()), float(v.z())