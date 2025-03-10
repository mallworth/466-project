import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import os

def safe_read_csv(file_path):
    """Enhanced CSV reading with error handling"""
    try:
        # Read the CSV file, skipping the first row (header) which contains the dataset name
        df = pd.read_csv(file_path, header=1, low_memory=False)
        return df
    except Exception as e:
        print(f"Error reading {file_path} with header=1: {e}")
        try:
            # Fallback to standard reading if skipping header fails
            df = pd.read_csv(file_path, low_memory=False)
            return df
        except Exception as e2:
            print(f"Fallback reading also failed: {e2}")
            return None

def kmeans_clustering():
    """Apply K-means clustering to identify music groups"""
    data_dir = './data/processed'
    output_dir = './results/kmeans'
    os.makedirs(output_dir, exist_ok=True)
    
    # Try to load the most appropriate dataset for clustering
    datasets = {}
    
    # List all processed files
    if not os.path.exists(data_dir):
        print(f"Directory {data_dir} does not exist. Run data_preprocessing.py first.")
        return
        
    processed_files = [f for f in os.listdir(data_dir) if f.endswith('_processed.csv')]
    
    if not processed_files:
        print("No processed datasets found. Run data_preprocessing.py first.")
        return
    
    print(f"Found {len(processed_files)} processed datasets")
    
    # Try each file until we find one with sufficient audio features
    for file in processed_files:
        file_path = os.path.join(data_dir, file)
        try:
            df = pd.read_csv(file_path)
            print(f"Using {file} for clustering")
            
            # Select features for clustering
            audio_features = ['danceability', 'energy', 'loudness', 'speechiness', 
                            'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']
            
            # Keep only features that exist in the dataframe
            features = [f for f in audio_features if f in df.columns]
            
            # Ensure features are numeric
            for feature in features:
                if not pd.api.types.is_numeric_dtype(df[feature]):
                    try:
                        df[feature] = pd.to_numeric(df[feature], errors='coerce')
                    except:
                        print(f"Could not convert {feature} to numeric, removing from features")
                        features.remove(feature)
            
            if len(features) >= 3:
                datasets[file.replace('_processed.csv', '')] = df
                break
            else:
                print(f"Not enough audio features in {file}, trying next dataset")
        except Exception as e:
            print(f"Error processing {file}: {e}")
    
    if not datasets:
        print("No suitable datasets found for clustering")
        return
    
    # Use the first dataset found
    dataset_name = list(datasets.keys())[0]
    df = datasets[dataset_name]
    
    print(f"Using {len(features)} features for clustering: {features}")
    
    # Prepare data
    X = df[features].fillna(0)
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Find optimal number of clusters using the elbow method
    inertias = []
    silhouette_scores = []
    k_range = range(2, 11)
    
    for k in k_range:
        # Create and fit KMeans model
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X_scaled)
        
        # Calculate inertia (sum of squared distances)
        inertias.append(kmeans.inertia_)
        
        # Calculate silhouette score
        labels = kmeans.labels_
        silhouette_avg = silhouette_score(X_scaled, labels)
        silhouette_scores.append(silhouette_avg)
        
        print(f"K={k}, Inertia={kmeans.inertia_:.2f}, Silhouette Score={silhouette_avg:.4f}")
    
    # Plot elbow method results
    plt.figure(figsize=(12, 6))
    plt.plot(k_range, inertias, 'bo-')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Inertia')
    plt.title('Elbow Method for Optimal k')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'elbow_method.png'))
    plt.close()
    
    # Plot silhouette scores
    plt.figure(figsize=(12, 6))
    plt.plot(k_range, silhouette_scores, 'ro-')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Score for Optimal k')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'silhouette_scores.png'))
    plt.close()
    
    # Select optimal k based on silhouette score
    optimal_k = k_range[np.argmax(silhouette_scores)]
    print(f"Optimal number of clusters based on silhouette score: {optimal_k}")
    
    # Perform clustering with optimal k
    final_kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    final_kmeans.fit(X_scaled)
    
    # Add cluster labels to original data
    df['cluster'] = final_kmeans.labels_
    
    # Analyze clusters
    cluster_analysis = df.groupby('cluster')[features].mean()
    print("\nCluster centers (feature means):")
    print(cluster_analysis)
    
    # Save cluster analysis to CSV
    cluster_analysis.to_csv(os.path.join(output_dir, 'cluster_analysis.csv'))
    
    # Visualize clusters
    # Use PCA to reduce to 2 dimensions for visualization if more than 2 features
    if len(features) > 2:
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)
        
        plt.figure(figsize=(12, 10))
        scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=final_kmeans.labels_, cmap='viridis', s=50, alpha=0.8)
        plt.colorbar(scatter, label='Cluster')
        plt.title(f'PCA of {optimal_k} Music Clusters')
        plt.xlabel('PCA Component 1')
        plt.ylabel('PCA Component 2')
        plt.savefig(os.path.join(output_dir, 'clusters_pca.png'))
        plt.close()
        
        # Explain variance captured by PCA
        explained_variance = pca.explained_variance_ratio_
        print(f"Variance explained by PCA components: {explained_variance}")
        
        # Save variance explained to text file
        with open(os.path.join(output_dir, 'pca_variance.txt'), 'w') as f:
            f.write(f"Variance explained by PCA components: {explained_variance}")
    else:
        # If only 2 features, plot directly
        plt.figure(figsize=(12, 10))
        scatter = plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=final_kmeans.labels_, cmap='viridis', s=50, alpha=0.8)
        plt.colorbar(scatter, label='Cluster')
        plt.title(f'Clustering of Music based on {features[0]} and {features[1]}')
        plt.xlabel(features[0])
        plt.ylabel(features[1])
        plt.savefig(os.path.join(output_dir, 'clusters_direct.png'))
        plt.close()
    
    # Create radar charts for each cluster
    for cluster_id in range(optimal_k):
        # Create a figure without the problematic subplot_kw parameter
        plt.figure(figsize=(10, 10))
        ax = plt.subplot(111, polar=True)  # Create polar subplot separately
        
        # Get center for this cluster
        values = cluster_analysis.loc[cluster_id].values
        
        # Normalize values for radar chart
        min_vals = cluster_analysis.min().values
        max_vals = cluster_analysis.max().values
        if np.all(max_vals == min_vals):
            # Avoid division by zero
            normalized_values = np.zeros_like(values)
        else:
            range_vals = max_vals - min_vals
            range_vals = np.where(range_vals == 0, 1, range_vals)  # Avoid division by zero
            normalized_values = (values - min_vals) / range_vals
        
        # Set up angles for radar chart
        N = len(features)
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]  # Close the polygon
        
        # Add the values for the last point to close the polygon
        values_plot = np.append(normalized_values, normalized_values[0])
        
        # Plot radar chart
        ax.plot(angles, values_plot, 'o-', linewidth=2)
        ax.fill(angles, values_plot, alpha=0.25)
        ax.set_thetagrids(np.degrees(angles[:-1]), features)
        plt.title(f'Cluster {cluster_id} Characteristics')
        plt.savefig(os.path.join(output_dir, f'cluster_{cluster_id}_radar.png'))
        plt.close()
    
    # Analyze cluster composition
    if 'genre' in df.columns:
        # Analyze genre distribution in each cluster
        genre_distribution = pd.crosstab(df['cluster'], df['genre'], normalize='index')
        genre_distribution.to_csv(os.path.join(output_dir, 'cluster_genre_distribution.csv'))
        
        # Find dominant genre in each cluster
        dominant_genres = {}
        for cluster in range(optimal_k):
            if cluster in genre_distribution.index:
                dominant_genre = genre_distribution.loc[cluster].idxmax()
                dominant_pct = genre_distribution.loc[cluster].max() * 100
                dominant_genres[cluster] = (dominant_genre, dominant_pct)
                print(f"Cluster {cluster} dominant genre: {dominant_genre} ({dominant_pct:.1f}%)")
    
    # Get representative songs from each cluster
    song_info_cols = ['track_name', 'name', 'title', 'artist_name', 'artists', 'artist']
    available_cols = [col for col in song_info_cols if col in df.columns]
    
    if available_cols:
        # For each cluster, find 5 songs closest to centroid
        representative_songs = {}
        
        for cluster_id in range(optimal_k):
            # Get songs in this cluster
            cluster_songs = df[df['cluster'] == cluster_id]
            
            if len(cluster_songs) == 0:
                continue
                
            # Get cluster center
            center = final_kmeans.cluster_centers_[cluster_id]
            
            # Calculate distance to center for each song
            distances = []
            for idx, row in cluster_songs.iterrows():
                song_features = X_scaled[df.index.get_loc(idx)]
                distance = np.linalg.norm(song_features - center)
                distances.append((idx, distance))
            
            # Sort by distance and get top 5
            songs = sorted(distances, key=lambda x: x[1])[:5]
            representative_songs[cluster_id] = [(idx, df.loc[idx, available_cols].tolist(), dist) 
                                              for idx, dist in songs]
            
            print(f"\nCluster {cluster_id} representative songs:")
            for _, song_info, _ in representative_songs[cluster_id]:
                print(f"  {' - '.join(str(x) for x in song_info if pd.notna(x))}")
    
    # Save clustered data
    df.to_csv(os.path.join(output_dir, f'{dataset_name}_clustered.csv'), index=False)
    
    # Write summary report
    with open(os.path.join(output_dir, 'clustering_summary.txt'), 'w') as f:
        f.write(f"Dataset: {dataset_name}\n")
        f.write(f"Features used: {features}\n")
        f.write(f"Optimal number of clusters: {optimal_k}\n\n")
        f.write("Cluster sizes:\n")
        cluster_sizes = df['cluster'].value_counts().sort_index()
        for cluster, size in cluster_sizes.items():
            f.write(f"  Cluster {cluster}: {size} songs\n")
        
        f.write("\nCluster centers (feature means):\n")
        f.write(cluster_analysis.to_string())
    
    print(f"\nClustering complete. Results saved to {output_dir}")
    return df, final_kmeans

if __name__ == "__main__":
    kmeans_clustering()
