import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.metrics.cluster import rand_score
from sklearn.decomposition import PCA
from scipy import stats

def gmm_clustering(D):
    """
    Test effectiveness of using audio features to cluster songs by genre using GMM
    :param D: Pandas df
    """
    gt = D.dropna(axis=0).to_numpy()[:,-1]
    D = D.select_dtypes(include=['number']).dropna()
    scaler = StandardScaler()
    D = scaler.fit_transform(D)
    max_rand = (0, None, None)
    min_rand = (1, None, None)

    for n_components in range(8, 13):
        for covariance_type in ['full', 'tied', 'diag', 'spherical']:
            gmm = GaussianMixture(n_components=n_components, 
                                covariance_type=covariance_type,
                                random_state=42)
            clusters = gmm.fit_predict(D)
            rand = rand_score(gt, clusters)
            print(f"Components: {n_components}, Covariance: {covariance_type}, Rand Score: {rand}")
            
            if rand > max_rand[0]:
                max_rand = (rand, n_components, covariance_type)
            if rand < min_rand[0]:
                min_rand = (rand, n_components, covariance_type)

    print(f'''Maximum Rand Score:
\tComponents: {max_rand[1]}\tCovariance: {max_rand[2]}\tRand Score: {max_rand[0]}
Minimum Rand Score:
\tComponents: {min_rand[1]}\tCovariance: {min_rand[2]}\tRand Score: {min_rand[0]}''')

def visualize_gmm_clustering(D, n_components=10, covariance_type='full'):
    """
    Create a visualization of the results of GMM clustering
    :param D: Pandas df
    :param n_components: Number of Gaussian components
    :param covariance_type: Type of covariance parameter
    """
    D = D.select_dtypes(include=['number']).dropna()
    D = D.drop(columns=["popularity", "instance_id"])
    D = D[(np.abs(stats.zscore(D)) < 3).all(axis=1)]

    scaler = StandardScaler()
    D = scaler.fit_transform(D)
    pca = PCA(n_components=2)
    pca_D = pca.fit_transform(D)

    gmm = GaussianMixture(n_components=n_components, 
                         covariance_type=covariance_type,
                         random_state=42)
    clusters = gmm.fit_predict(D)
    
    df = pd.DataFrame(pca_D, columns=["x", "y"])
    df["Cluster"] = clusters

    # Plot
    plt.figure(figsize=(8, 8))
    scatter = plt.scatter(df["x"], df["y"], marker=".", s=5, c=df["Cluster"], cmap='tab10')
    plt.colorbar(scatter, label='Cluster')
    plt.title(f'GMM Clustering (n_components={n_components}, {covariance_type})')
    plt.xticks([])
    plt.yticks([])
    plt.show()

def visualize_gt(D):
    """
    Create a visualization of the actual clustering of the dataset
    :param D: Pandas df
    """
    D = D.dropna(axis=0)
    gt = D["music_genre"]
    D = D.select_dtypes(include=["number"])
    D = D.drop(columns=["popularity", "instance_id"])
    D = D[(np.abs(stats.zscore(D)) < 3).all(axis=1)]
    gt = gt.loc[D.index]

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

    colors = [genre_colors[genre] for genre in df["Genre"]]
    plt.figure(figsize=(8,8))
    plt.scatter(x=df["x"], y=df["y"], marker=".", s=5, c=colors)
    legend = [mpatches.Patch(color=color, label=genre) for genre, color in genre_colors.items()]
    plt.legend(handles=legend, loc="lower left")
    plt.xticks([])
    plt.yticks([])
    plt.show()

if __name__ == '__main__':
    D = pd.read_csv("data/music_genre.csv")
    gmm_clustering(D)
    visualize_gmm_clustering(D)
    visualize_gt(D)
