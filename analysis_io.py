import os
import pickle


OUTPUT_DIR = "output"


def build_output_path(name):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    return os.path.join(OUTPUT_DIR, name)


def save_analysis_results(output_prefix, events):
    filename = build_output_path(f"{output_prefix}analysis_results.pkl")
    with open(filename, "wb") as f:
        pickle.dump(events, f)

    print(f"Saved processed results to: {filename}")


def load_analysis_results(filename):
    if not os.path.exists(filename):
        candidate = build_output_path(filename)
        if os.path.exists(candidate):
            filename = candidate
    with open(filename, "rb") as f:
        return pickle.load(f)
