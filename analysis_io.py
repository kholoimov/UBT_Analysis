import pickle


def save_analysis_results(output_prefix, events):
    filename = f"{output_prefix}analysis_results.pkl"
    with open(filename, "wb") as f:
        pickle.dump(events, f)

    print(f"Saved processed results to: {filename}")


def load_analysis_results(filename):
    with open(filename, "rb") as f:
        return pickle.load(f)