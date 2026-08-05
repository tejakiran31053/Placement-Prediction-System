from scipy.signal import correlate
from scipy.stats import multivariate_t

from src.data.load_data import load_data
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

def basic_eda(df):
    print("First five rows")
    print(df.head())
    print("-----------------------------------------------------------------------------------------")
    print("Last five rows")
    print(df.tail())
    print("-----------------------------------------------------------------------------------------")
    print("from row 25 to 35")
    print(df.iloc[25:36])
    print("-----------------------------------------------------------------------------------------")
    print("random rows")
    print(df.sample(10))
    print("-----------------------------------------------------------------------------------------")
    print("Column Names")
    print(df.columns)
    print("-----------------------------------------------------------------------------------------")
    print("datatypes:")
    print(df.dtypes)
    print("-----------------------------------------------------------------------------------------")
    print("Complete Information")
    print(df.info())
    print("Data Types")
    print(df.describe())
    print("Colums null values")
    missing = df.isnull().sum()
    print(missing[missing > 0])
    print("Target variable status")
    print(df["PlacementStatus"].value_counts())

    count = df["PlacementStatus"].value_counts()

    plt.figure(figsize=(6,5))
    plt.title("Distribution of Placement Status")
    plt.bar(count.index, count.values)
    plt.xlabel("Placement Status")
    plt.ylabel("Count")
    plt.savefig(r"C:\Users\lenovo\PycharmProjects\PlacementPredictionSystem\app\static\charts")
    plt.show()


def univariate(df):
    plt.figure(figsize=(6,5))
    plt.hist(df["CGPA"], bins=10, edgecolor="black")
    plt.title("Histogram of CGPA")
    plt.xlabel("CGPA")
    plt.ylabel("Frequency")
    plt.savefig(r"C:\Users\lenovo\PycharmProjects\PlacementPredictionSystem\app\static\charts/cgpa.png")
    plt.show()
    gendercount=df["Gender"].value_counts()
    plt.figure(figsize=(6,5))
    plt.pie(
        gendercount,
        labels=gendercount.index,
        autopct="%1.1f%%",
        startangle=90
    )
    plt.title("Gender Distribution Piechart")
    plt.savefig(r"C:\Users\lenovo\PycharmProjects\PlacementPredictionSystem\app\static\charts/cgpa_aptitudescore_scatter")
    plt.show()


def bivariate(df):
        plt.figure(figsize=(6,5))
        plt.scatter(df["CGPA"], df["AptitudeTestScore"])
        plt.title("CGPA vs Aptitude Test Score")
        plt.xlabel("CGPA")
        plt.ylabel("Aptitude Test Score")
        plt.savefig(r"C:\Users\lenovo\PycharmProjects\PlacementPredictionSystem\app\static\charts/gender_distribution.png")
        plt.show()
        plt.close()
        plt.figure(figsize=(6,5))


        placed = df[df["PlacementStatus"] == 1]["CGPA"]
        not_placed = df[df["PlacementStatus"] == 0]["CGPA"]
        plt.boxplot([placed, not_placed], label=["placed", "not placed"])
        plt.title("CGPA vs Placement Status")
        plt.xlabel("Placement Status")
        plt.ylabel("CGPA")
        plt.savefig(r"C:\Users\lenovo\PycharmProjects\PlacementPredictionSystem\app\static\charts/Boxplot.png")
        plt.show()




def multivariate(df):
    data = df[["CGPA", "AptitudeTestScore", "PlacementStatus"]]
    correlation = data.corr()
    plt.figure(figsize=(6, 5))
    sns.heatmap(correlation,
                annot=True,
                cmap="coolwarm",
                fmt=".2f")
    plt.title("Correlation heatmap")
    plt.savefig(r"C:\Users\lenovo\PycharmProjects\PlacementPredictionSystem\app\static\charts/Heatmap.png")
    plt.show()
    plt.close()



if __name__ == "__main__":
    df = load_data()
    basic_eda(df)
    univariate(df)
    bivariate(df)
    multivariate(df)