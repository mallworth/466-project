import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.metrics.cluster import rand_score
from sklearn.decomposition import PCA
from scipy import stats

'''
Test effectivness of using audio features to cluster songs by genre using DBSCAN
:param D: Pandas df
'''
def dbscan(D):
    # Select numeric attributes (These correspond to audio features)
    gt = D.dropna(axis=0).to_numpy()[:,-1]
    D = D.select_dtypes(include=['number']).dropna()
    scaler = StandardScaler()
    D = scaler.fit_transform(D)
    max_rand = (0, None, None, None)
    min_rand = (1, None, None, None)

    # Run through a range of hyperparameters
    for epsilon in np.arange(2, 3.5, 0.5):
        for samples in range(8, 13): 
            clusters = DBSCAN(eps=epsilon, min_samples=samples).fit(D)
            rand = rand_score(gt, clusters.labels_)
            print(rand)
            if rand > max_rand[0]:
                max_rand = (rand, epsilon, samples, len(set(clusters.labels_)))
            if rand < min_rand[0]:
                min_rand = (rand, epsilon, samples, len(set(clusters.labels_)))

    print(f'''Maximum Rand Score:
\tEpsilon: {max_rand[1]}\tMin Samples: {max_rand[2]}\tRand Score: {max_rand[0]} Cluster Count: {max_rand[3]}
Minimum Rand Score:
\tEpsilon: {min_rand[1]}\tMin Samples: {min_rand[2]}\tRand Score: {min_rand[0]} Cluster Count: {max_rand[3]}''')


'''
Create a visualization of the results of DBSCAN
NOTE: Best hyperparams are default
:param D: Pandas df
:param e: Epsilon value for DBSCAN
:param samples: Minimum sample value for DBSCAN
'''
def visualize_clustering(D, e=2, samples=12):
    D = D.select_dtypes(include=['number']).dropna()
    D = D[(np.abs(stats.zscore(D)) < 3).all(axis=1)]

    # Scale data & perform Principal Component Analysis (Decompose into 2 vectors for visualization)
    scaler = StandardScaler()
    D = scaler.fit_transform(D)
    pca = PCA(n_components=2)
    pca_D = pca.fit_transform(D)

    # Run DBSCAN 
    res = DBSCAN(eps=e, min_samples=samples)
    clusters = res.fit_predict(D)
    df = pd.DataFrame(pca_D, columns=["x", "y"])
    df["Cluster"] = clusters

    # Separate outliers with cluster
    outliers = df[df["Cluster"] == -1]
    cluster = df[df["Cluster"] != -1]

    # Plot
    plt.figure(figsize=(8, 8))
    plt.scatter(cluster["x"], cluster["y"], marker=".", s=5, c="deepskyblue", label="Cluster points")
    plt.scatter(outliers["x"], outliers["y"], marker="x", s=10, c="darkorange", label="Outliers")
    plt.xticks([])
    plt.yticks([])
    plt.show()


'''
Create a visualization of the actual clustering of the dataset
:param D: Pandas df
'''
def visualize_gt(D):
    D = D.dropna(axis=0)
    gt = D["music_genre"]
    # Drop non-numeric attributes and outliers
    D = D.select_dtypes(include=['number'])
    D = D[(np.abs(stats.zscore(D)) < 3).all(axis=1)]
    # Match genres to remaining datapoints
    gt = gt.loc[D.index]

    # Scale data & perform Principal Component Analysis (Decompose into 2 vectors for visualization)
    scaler = StandardScaler()
    D = scaler.fit_transform(D)
    pca = PCA(n_components=2)
    pca_D = pca.fit_transform(D)

    df = pd.DataFrame(pca_D, columns=["x", "y"])
    df["Genre"] = gt.values

    genre_colors = {
        "Country": "red",
        "Hip-Hop": "darkorange",
        "Jazz": "limegreen",
        "Classical": "deepskyblue",
        "Electronic": "royalblue",
        "Rap": "deeppink",
        "Rock": "darkviolet",
        "Anime": "yellow",
        "Alternative": "gold",
        "Blues": "lightcoral"
    }

    # Plot 
    colors = [genre_colors[genre] for genre in df["Genre"]]
    plt.figure(figsize=(8,8))
    plt.scatter(x=df["x"], y=df["y"], marker=".", s=5, c=colors)
    legend = [mpatches.Patch(color=color, label=genre) for genre, color in genre_colors.items()]
    plt.legend(handles=legend, title="Genres")
    # Removed axis numbers because scale is not easy to understand...collapsed several dimensions into 2
    plt.xticks([])
    plt.yticks([])
    plt.show()

if __name__ == '__main__':
    D = pd.read_csv("data/music_genre.csv")
    # Uncomment to test hyperparameters for DBSCAN
    # dbscan(D)

    # Uncomment to visualize clustering by DBSCAN
    # visualize_clustering(D)

    # Uncomment to visualize clustering of actual data
    # visualize_gt(D)
