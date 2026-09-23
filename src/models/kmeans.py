import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import os

# Load dataset using dynamic path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_PATH = os.path.join(BASE_DIR, "data", "placement_data.csv")

df = pd.read_csv(DATA_PATH)

# Show available columns
print("Available columns:")
print(df.columns.tolist())

# Features to use
features = ["CGPA", "Internships", "Projects", "CodingTestScore"]

# Keep only features that exist in the dataset
features = [feature for feature in features if feature in df.columns]
print("\nFeatures used:", features)

if len(features) == 0:
    raise ValueError("None of the selected features exist in the dataset.")

# Select features and remove missing values
X = df[features].dropna()
print("\nSelected features:")
print(X.head())

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Create K-Means model
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)

# Assign clusters
clusters = kmeans.fit_predict(X_scaled)

# Add cluster column
X = X.copy()
X["Cluster"] = clusters

print("\nCluster Assignments:")
print(X.head(10))

# Get cluster centers
centers_scaled = kmeans.cluster_centers_
centers = scaler.inverse_transform(centers_scaled)
centers_df = pd.DataFrame(centers, columns=features)

print("\nCluster Centers:")
print(centers_df)
