from src.data.load_data import load_data
from src.data.preprocess import (
    handle_missing_values,
    standardize_data
)
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
import numpy as np


def create_model(eps, min_samples):
    model = DBSCAN(eps=eps, min_samples=min_samples)
    return model


def train_model(model, X):
    model.fit(X)
    print("\nDBSCAN Clustering completed!")
    return model


def evaluate_model(model, X):
    labels = model.labels_
    non_noise = labels != -1

    if len(set(labels[non_noise])) >= 2:
        score = silhouette_score(X[non_noise], labels[non_noise])
        print("\nSilhouette Score:")
        print(score)
    else:
        print("\nSilhouette Score cannot be calculated.")

    return labels


def main():
    df = load_data()
    print("Dataset Shape:")
    print(df.shape)

    # Sample Data
    df = df.sample(n=500, random_state=42)

    # Select 4 Features
    features = ["CGPA", "AttendancePercent", "CodingTestScore", "MockInterviewScore"]
    X = df[features].copy()
    print("\nSelected Features:", features)

    # Missing Values
    X, _, _ = handle_missing_values(X, X, features)

    # Standardization
    X, _, _ = standardize_data(X, X, features)

    # DBSCAN Parameters
    eps = 0.8
    min_samples = 5
    print("\nDBSCAN Parameters:")
    print("eps:", eps)
    print("min_samples:", min_samples)

    model = create_model(eps, min_samples)
    model = train_model(model, X)
    labels = evaluate_model(model, X)

    # Number of Clusters
    clusters = set(labels)
    if -1 in clusters:
        clusters.remove(-1)
    print("\nNumber of Clusters:", len(clusters))

    # Number of Noise Points
    noise_count = np.sum(labels == -1)
    print("Number of Noise Points:", noise_count)

    # Cluster Distribution
    df["Cluster"] = labels
    print("\nCluster Distribution:")
    print(df["Cluster"].value_counts().sort_index())


if __name__ == "__main__":
    main()
