import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.data.load_data import load_data
import matplotlib
matplotlib.use("Agg")   # Non-interactive backend — no popup windows
import matplotlib.pyplot as plt
import seaborn as sns

# Charts directory — relative to project root
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CHARTS_DIR = os.path.join(BASE_DIR, "app", "static", "charts")
os.makedirs(CHARTS_DIR, exist_ok=True)


def basic_eda(df):
    print("First five rows")
    print(df.head())
    print("-" * 89)
    print("Last five rows")
    print(df.tail())
    print("-" * 89)
    print("from row 25 to 35")
    print(df.iloc[25:36])
    print("-" * 89)
    print("random rows")
    print(df.sample(10))
    print("-" * 89)
    print("Column Names")
    print(df.columns)
    print("-" * 89)
    print("datatypes:")
    print(df.dtypes)
    print("-" * 89)
    print("Complete Information")
    print(df.info())
    print("Data Types")
    print(df.describe())
    print("Columns null values")
    missing = df.isnull().sum()
    print(missing[missing > 0])
    print("Target variable status")
    print(df["PlacementStatus"].value_counts())

    count = df["PlacementStatus"].value_counts()
    plt.figure(figsize=(6, 5))
    plt.title("Distribution of Placement Status")
    plt.bar(count.index, count.values)
    plt.xlabel("Placement Status")
    plt.ylabel("Count")
    plt.savefig(os.path.join(CHARTS_DIR, "placement_status.png"))
    plt.close()


def univariate(df):
    # CGPA Histogram
    plt.figure(figsize=(6, 5))
    plt.hist(df["CGPA"], bins=10, edgecolor="black")
    plt.title("Histogram of CGPA")
    plt.xlabel("CGPA")
    plt.ylabel("Frequency")
    plt.savefig(os.path.join(CHARTS_DIR, "CGPA.png"))
    plt.close()

    # Gender Pie Chart
    gendercount = df["Gender"].value_counts()
    plt.figure(figsize=(6, 5))
    plt.pie(
        gendercount,
        labels=gendercount.index,
        autopct="%1.1f%%",
        startangle=90
    )
    plt.title("Gender Distribution Piechart")
    plt.savefig(os.path.join(CHARTS_DIR, "gender_distribution.png"))
    plt.close()


def bivariate(df):
    # CGPA vs Aptitude Test Score scatter
    plt.figure(figsize=(6, 5))
    plt.scatter(df["CGPA"], df["AptitudeTestScore"])
    plt.title("CGPA vs Aptitude Test Score")
    plt.xlabel("CGPA")
    plt.ylabel("Aptitude Test Score")
    plt.savefig(os.path.join(CHARTS_DIR, "cgpa_aptitudescore_scatter.png"))
    plt.close()

    # CGPA vs Placement Status Boxplot
    placed = df[df["PlacementStatus"] == 1]["CGPA"]
    not_placed = df[df["PlacementStatus"] == 0]["CGPA"]
    plt.figure(figsize=(6, 5))
    plt.boxplot([placed, not_placed])
    plt.xticks([1, 2], ["Placed", "Not Placed"])
    plt.title("CGPA vs Placement Status")
    plt.xlabel("Placement Status")
    plt.ylabel("CGPA")
    plt.savefig(os.path.join(CHARTS_DIR, "boxplot.png"))
    plt.close()


def multivariate(df):
    data = df[["CGPA", "AptitudeTestScore", "PlacementStatus"]]
    correlation = data.corr()
    plt.figure(figsize=(6, 5))
    sns.heatmap(correlation, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Heatmap")
    plt.savefig(os.path.join(CHARTS_DIR, "Heatmap.png"))
    plt.close()


if __name__ == "__main__":
    df = load_data()
    basic_eda(df)
    univariate(df)
    bivariate(df)
    multivariate(df)
    print("\nAll charts saved to:", CHARTS_DIR)