import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import folium
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, davies_bouldin_score

class GeospatialClustering:
    """
    A class to perform geospatial clustering using K-Means, DBSCAN, Agglomerative Clustering, 
    and Gaussian Mixture Models (GMM). It includes methods for preprocessing, clustering, 
    evaluating, and visualizing results.
    """
    
    def __init__(self, file_path):
        """
        Initializes the GeospatialClustering class.
        
        Args:
            file_path (str): Path to the CSV dataset.
        """
        self.df = pd.read_csv(file_path)
        self.scaler = StandardScaler()
        self.best_k = None
    
    def preprocess_data(self):
        """
        Cleans the dataset by removing missing and duplicate values, 
        removes outliers, and normalizes latitude and longitude using StandardScaler.
        
        Returns:
            None
        """
        self.df.dropna(inplace=True)
        self.df.drop_duplicates(inplace=True)
        self.df[['longitude', 'latitude']] = self.df['Longitude;Latitude'].str.split(';', expand=True).astype(float)

        # Outlier Detection using Z-Score
        z_scores = np.abs((self.df[['latitude', 'longitude']] - self.df[['latitude', 'longitude']].mean()) / 
                        self.df[['latitude', 'longitude']].std())
        outliers = (z_scores > 3).any(axis=1)

        print(f"Number of Outliers Removed: {outliers.sum()}")
        self.df = self.df[~outliers]  # Remove outliers

        # Normalize Data
        self.df[['latitude', 'longitude']] = self.scaler.fit_transform(self.df[['latitude', 'longitude']])
    
    def plot_elbow_method(self, max_k=10):
        """
        Determines the optimal number of clusters using the elbow method.
        
        Args:
            max_k (int, optional): Maximum number of clusters to test. Defaults to 10.
        
        Returns:
            None
        """
        inertia = []
        k_values = range(1, max_k)
        for k in k_values:
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(self.df[['latitude', 'longitude']])
            inertia.append(kmeans.inertia_)
        
        plt.figure(figsize=(8, 6))
        plt.plot(k_values, inertia, marker='o', linestyle='--')
        plt.xlabel('Number of Clusters (k)')
        plt.ylabel('Inertia')
        plt.title('Elbow Method for Optimal k')
        plt.grid()
        plt.show()
    
    def cluster_data(self, best_k):
        """
        Applies K-Means, DBSCAN, Agglomerative Clustering, and GMM to the dataset.
        
        Args:
            best_k (int): Optimal number of clusters determined from the elbow method.
        
        Returns:
            None
        """
        self.best_k = best_k
        self.df['kmeans'] = KMeans(n_clusters=best_k, random_state=42).fit_predict(self.df[['latitude', 'longitude']])
        self.df['dbscan'] = DBSCAN(eps=0.3, min_samples=5).fit_predict(self.df[['latitude', 'longitude']])
        self.df['agglo'] = AgglomerativeClustering(n_clusters=best_k).fit_predict(self.df[['latitude', 'longitude']])
        self.df['gmm'] = GaussianMixture(n_components=best_k, random_state=42).fit_predict(self.df[['latitude', 'longitude']])
    
    def evaluate_clusters(self):
        """
        Computes clustering evaluation metrics (Silhouette Score & Davies-Bouldin Score) and writes them to a file.
        
        Returns:
            None
        """
        silhouette_scores = {}
        davies_bouldin_scores = {}
        
        for method in ['kmeans', 'dbscan', 'agglo', 'gmm']:
            labels = self.df[method]
            
            if method == 'dbscan' and len(set(labels)) == 1:
                silhouette_scores['DBSCAN'] = -1
                davies_bouldin_scores['DBSCAN'] = -1
            else:
                silhouette_scores[method] = silhouette_score(self.df[['latitude', 'longitude']], labels)
                davies_bouldin_scores[method] = davies_bouldin_score(self.df[['latitude', 'longitude']], labels)
        
        with open("check.txt", "w") as f:
            f.write(f"Silhouette Scores: {silhouette_scores}\n")
            f.write(f"Davies-Bouldin Scores: {davies_bouldin_scores}\n")
    
    def visualize_clusters(self):
        """
        Plots the clustered geospatial data for different clustering methods.
        
        Returns:
            None
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 12))
        clustering_methods = ['kmeans', 'dbscan', 'agglo', 'gmm']
        
        for ax, method in zip(axes.flatten(), clustering_methods):
            scatter = ax.scatter(self.df['longitude'], self.df['latitude'], c=self.df[method], cmap='viridis', alpha=0.5)
            ax.set_title(f'{method.upper()} Clustering')
            ax.set_xlabel('Longitude')
            ax.set_ylabel('Latitude')
            ax.grid()
            plt.colorbar(scatter, ax=ax)
        
        plt.tight_layout()
        plt.show()
    
    def generate_folium_map(self):
        """
        Creates a folium map with clustered points and saves it as an HTML file.
        
        Returns:
            None
        """
        map_center = [self.df['latitude'].mean(), self.df['longitude'].mean()]
        m = folium.Map(location=map_center, zoom_start=6)
        
        for _, row in self.df.iterrows():
            folium.CircleMarker(
                location=[row['latitude'], row['longitude']],
                radius=3,
                color='blue',
                fill=True,
                fill_color='blue'
            ).add_to(m)
        
        m.save('map.html')
        print("Map saved as 'map.html'. Open in a browser to view.")
    
    def generate_folium_map_after_clustering(self):
        """
        Creates a folium map with clustered points for K-Means and saves it as an HTML file.
        
        Returns:
            None
        """
        map_center = [self.df['latitude'].mean(), self.df['longitude'].mean()]
        m = folium.Map(location=map_center, zoom_start=6)
        
        colors = ['red', 'blue', 'green', 'purple', 'orange', 'darkred', 'lightred', 'beige', 'darkblue', 'darkgreen']
        
        for _, row in self.df.iterrows():
            folium.CircleMarker(
                location=[row['latitude'], row['longitude']],
                radius=3,
                color=colors[row['kmeans'] % len(colors)],
                fill=True,
                fill_color=colors[row['kmeans'] % len(colors)]
            ).add_to(m)
        
        m.save('kmeans_map.html')
        print("K-Means Clustering Map saved as 'kmeans_map.html'. Open in a browser to view.")


# Usage Example
if __name__ == "__main__":
    file_path = "/Users/pavandeekshith/B-Tech/Btech_3rd_Year/6th_Sem/Geospatial_clustering/ML Assignment Dataset.csv"  # Replace with actual file path
    clustering = GeospatialClustering(file_path)
    clustering.preprocess_data()
    clustering.plot_elbow_method()
    clustering.cluster_data(best_k=3)  # Set optimal k from elbow method
    clustering.evaluate_clusters()
    clustering.visualize_clusters()
    clustering.generate_folium_map()
    clustering.generate_folium_map_after_clustering()
