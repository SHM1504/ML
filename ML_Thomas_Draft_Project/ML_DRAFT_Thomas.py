#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  6 11:52:14 2025

@author: thomasreuser
"""

#%%

# --------------
# 1. CORRECTED IMPORTS
# --------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
from sklearn.decomposition import PCA  # Added PCA import
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.stats import chi2_contingency

# --------------
# 2. CORRECTED CUSTOM MODEL CLASS
# --------------
class CustomModel:
    def __init__(self):
        """Initialize your pre-trained models here"""
        pass
    
    def extract_features(self, image_path):
        """IMPLEMENT YOUR FEATURE EXTRACTION"""
        # Return: np.array of features
        # Example: return np.random.rand(512)
        raise NotImplementedError("Implement feature extraction")
    
    def predict_sentiment(self, image_path):
        """IMPLEMENT YOUR SENTIMENT ANALYSIS"""
        # Return: 1 (positive) or 0 (negative)
        # Example: return np.random.randint(0, 2)
        raise NotImplementedError("Implement sentiment prediction")
        
#%%

# --------------
# 3. PATH USAGE EXPLAINED
# --------------

def load_and_validate_data(excel_path):  # CORRECTED FUNCTION NAME
    """Uses Path for robust path handling"""
    df = pd.read_excel(excel_path)
    valid_paths, valid_sentiments = [], []
    
    for idx, row in df.iterrows():
        try:
            # ACTUAL Path USAGE HERE
            path = Path(row['Image_Path']).resolve()  # Convert to absolute path
            if not path.exists():
                print(f"Missing: {path}")
                continue
                
            valid_paths.append(str(path))
            # Sentiment conversion remains the same
            valid_sentiments.append(1 if row['Sentiment'].lower() == 'positive' else 0)
        except Exception as e:
            print(f"Error in row {idx}: {e}")
    
    return valid_paths, np.array(valid_sentiments)

# --------------
# 4. CORRECTED CLUSTERING WORKFLOW
# --------------
def cluster_analysis(image_paths, sentiments, max_clusters=15):
    """Main analysis workflow"""
    # Initialize models
    model = CustomModel()
    
    # Feature extraction
    print("Extracting features...")
    features = np.array([model.extract_features(p) for p in image_paths])
    
    # Preprocessing
    features = StandardScaler().fit_transform(features)
    
    # PCA dimensionality reduction  # Fixed PCA implementation
    pca = PCA(n_components=0.95)
    features_reduced = pca.fit_transform(features)
    
    # Clustering
    print("Clustering...")
    distance_matrix = pdist(features_reduced, metric='cosine')
    Z = linkage(distance_matrix, method='ward')
    clusters = fcluster(Z, t=max_clusters, criterion='maxclust')
    
    return features_reduced, clusters  # Return reduced features

# --------------
# (Keep all other functions the same as previous version,
# but remove any references to os module)

# --------------
# 5. VISUALIZATION & ANALYSIS
# --------------
def visualize_results(features, clusters, sentiments, image_paths):
    """Generate all visual outputs"""
    # First get unique clusters
    unique_clusters = np.unique(clusters)  # THIS WAS MISSING
    
    # t-SNE visualization
    tsne = TSNE(n_components=2, random_state=42)
    vis_features = tsne.fit_transform(features)
    
    plt.figure(figsize=(15, 10))
    scatter = plt.scatter(vis_features[:, 0], vis_features[:, 1], 
                         c=clusters, cmap='tab20', alpha=0.7)
    plt.colorbar(scatter, label='Cluster')
    plt.title('Cluster Visualization (t-SNE)')
    plt.show()
    
    # Sentiment-cluster relationship
    cluster_stats = []
    for c in unique_clusters:  # Now using defined variable
        mask = clusters == c
        pos_ratio = np.mean(sentiments[mask])
        cluster_stats.append(pos_ratio)
    
    plt.figure(figsize=(20, 3*len(unique_clusters)))
    for i, c in enumerate(unique_clusters):
        cluster_samples = np.where(clusters == c)[0][:5]
        for j, idx in enumerate(cluster_samples):
            ax = plt.subplot(len(unique_clusters), 5, i*5 + j + 1)
            try:
                img = Image.open(image_paths[idx])
                ax.imshow(img)  # USE ax INSTEAD OF plt
                ax.set_title(f"Cluster {c}\nSent: {'+' if sentiments[idx] else '-'}")  # ax METHOD
            except:
                ax.set_title("Image load error")
            ax.axis('off')  # ax METHOD
    plt.tight_layout()
    plt.show()
    
    # Sample images per cluster
    unique_clusters = np.unique(clusters)
    plt.figure(figsize=(20, 3*len(unique_clusters)))
    for i, c in enumerate(unique_clusters):
        cluster_samples = np.where(clusters == c)[0][:5]  # First 5 samples
        for j, idx in enumerate(cluster_samples):
            ax = plt.subplot(len(unique_clusters), 5, i*5 + j + 1)
            try:
                img = Image.open(image_paths[idx])
                plt.imshow(img)
                plt.title(f"Cluster {c}\nSent: {'+' if sentiments[idx] else '-'}")
            except:
                plt.title("Image load error")
            plt.axis('off')
    plt.tight_layout()
    plt.show()
    
    # Statistical analysis
    contingency = np.zeros((len(unique_clusters), 2))
    for i, c in enumerate(unique_clusters):
        mask = clusters == c
        contingency[i] = [sum(sentiments[mask] == 0), sum(sentiments[mask] == 1)]
    
    chi2, p, _, _ = chi2_contingency(contingency)
    print(f"\nStatistical Significance:\nχ² = {chi2:.1f}, p = {p:.4f}")
    print("Significant association!" if p < 0.05 else "No significant relationship")

# --------------
# 6. MAIN EXECUTION
# --------------
if __name__ == "__main__":
    # 1. Load data
    excel_path = "ML_InputDoc_ML__Project_AdSentiment_Clustering_052025.xlsx"  # UPDATE THIS
    image_paths, sentiments = load_and_validate_data(excel_path)
    
    # 2. Process and cluster
    features, clusters = cluster_analysis(image_paths, sentiments)
    
    # 3. Analyze and visualize
    visualize_results(features, clusters, sentiments, image_paths)
    
    # 4. Save cluster assignments
    output_df = pd.DataFrame({
        'Image_Path': image_paths,
        'Cluster': clusters,
        'Sentiment': sentiments
    })
    output_df.to_excel("cluster_results.xlsx", index=False)
    print("\nResults saved to cluster_results.xlsx")