import os
import pandas as pd

# Resolves to <project_root>/data/placement_data.csv regardless of the
# machine / OS the project is run on.
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_PATH = os.path.join(BASE_DIR, "data", "placement_data.csv")


def load_data(path=DATA_PATH):
    df = pd.read_csv(path)
    return df


def get_summary(df):
    return {
        "rows": df.shape[0],
        "columns": df.shape[1],
        "target": "PlacementStatus"
    }

if __name__ == "__main__":
    df = load_data()
    print(get_summary(df))