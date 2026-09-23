from sklearn.datasets import make_moons
from sklearn.cluster import DBSCAN, KMeans
import matplotlib.pyplot as plt


def create_data():
    X, _ = make_moons(n_samples=300, noise=0.06, random_state=42)
    return X


def create_kmeans_model():
    model = KMeans(n_clusters=2, n_init=10, random_state=42)
    return model


def create_dbscan_model():
    model = DBSCAN(eps=0.2, min_samples=5)
    return model


def train_model(model, X):
    model.fit(X)
    return model.labels_


def display_results(X, kmeans_labels, dbscan_labels):
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))

    ax[0].scatter(X[:, 0], X[:, 1], c=kmeans_labels)
    ax[0].set_title("K-Means")

    ax[1].scatter(X[:, 0], X[:, 1], c=dbscan_labels)
    ax[1].set_title("DBSCAN")

    plt.show()


def main():
    X = create_data()

    kmeans_model = create_kmeans_model()
    dbscan_model = create_dbscan_model()

    kmeans_labels = train_model(kmeans_model, X)
    dbscan_labels = train_model(dbscan_model, X)

    display_results(X, kmeans_labels, dbscan_labels)


if __name__ == "__main__":
    main()
