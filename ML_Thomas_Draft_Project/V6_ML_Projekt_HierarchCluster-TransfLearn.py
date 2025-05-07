#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Image Clustering and Sentiment Analysis
Created on Tue May 6 11:52:14 2025
@author: thomasreuser

This script performs image clustering and sentiment analysis by:
1. Loading URLs from an Excel file
2. Downloading and processing images
3. Extracting features using a pre-trained ResNet50 model
4. Clustering images based on visual features
5. Analyzing sentiment distribution across clusters
6. Visualizing results and saving outputs
"""    

#%% 1. IMPORTS

# Data handling and analysis
import pandas as pd          # Data manipulation and analysis (Excel reading, DataFrame operations)
import numpy as np           # Numerical computing (arrays, mathematical operations)

# Visualization
import matplotlib.pyplot as plt  # Plotting library for creating visualizations
import seaborn as sns            # Statistical data visualization, enhances matplotlib (heatmaps)
from scipy.cluster.hierarchy import dendrogram  # For visualizing hierarchical clustering results

# HTTP and URL handling
import requests              # HTTP requests to download images from URLs
from urllib.parse import urlparse, urlunparse, parse_qs, urlencode  # URL parsing and manipulation

# Image processing
from io import BytesIO       # In-memory binary stream for image data
from PIL import Image        # Python Imaging Library for image operations

# Deep learning+
import torch                 # PyTorch deep learning framework
import torchvision.models as models  # Pre-trained models (ResNet50)
from torchvision import transforms   # Image transformations for neural networks

# pre trained sentiment analysis
from textblob import TextBlob
from textblob.sentiments import NaiveBayesAnalyzer,PatternAnalyzer

# Machine learning and statistics
from sklearn.decomposition import PCA  # Principal Component Analysis for dimensionality reduction
from sklearn.preprocessing import StandardScaler  # Feature normalization
from sklearn.manifold import TSNE      # t-SNE for visualization of high-dimensional data
from scipy.spatial.distance import pdist  # Pairwise distances computation
from scipy.cluster.hierarchy import linkage, fcluster  # Hierarchical clustering algorithms
from scipy.stats import chi2_contingency  # Statistical significance testing


#%% 2. CUSTOM MODEL CLASS

class CustomModel:
    """
    Custom model wrapper that uses a pre-trained feature extractor
    and adds sentiment prediction capability.
    """
    def __init__(self):
        """Initialize with a pre-trained feature extractor"""
        self.feature_extractor = PretrainedFeatureExtractor()
    
    def extract_features(self, image):
        """
        Extract features from an image using the pre-trained model
        
        Args:
            image (PIL.Image): Input image
            
        Returns:
            numpy.ndarray: Feature vector
        """
        # Convert to RGB if needed
        if image.mode in ('RGBA', 'LA'):
            image = image.convert('RGB')
            
        # Extract features
        features = self.feature_extractor.extract_features(image)
        
        if features is not None:
            return features
        else:
            # Fallback to random features (remove in production)
            print("Warning: Using random features as fallback")
            return np.random.rand(2048)  # ResNet-50 features are 2048-dim
    
    def predict_sentiment(self, image):
        """
        Predict sentiment from image
        
        Args:
            image (PIL.Image): Input image
            
        Returns:
            int: 0 for negative, 1 for positive sentiment
            
        Note: Currently returns random values. Implement your sentiment model here.
        """
        # TODO: Implement your sentiment analysis model
        return np.random.randint(0, 2)  # Placeholder


#%% 3. PRETRAINED FEATURE EXTRACTOR

class PretrainedFeatureExtractor:
    """
    Extracts image features using a pre-trained ResNet50 model
    with the final fully connected layer removed.
    """
    def __init__(self):
        """Initialize the feature extractor with a pre-trained ResNet50"""
        # Set device (GPU if available, otherwise CPU)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load pre-trained model
        self.model = models.resnet50(pretrained=True)
        
        # Remove final fully connected layer to get feature vectors
        self.model = torch.nn.Sequential(*list(self.model.children())[:-1])
        
        # Set model to evaluation mode
        self.model.eval()
        
        # Define image preprocessing pipeline
        self.preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])
    
    def extract_features(self, pil_image):
        """
        Extract features from a PIL Image
        
        Args:
            pil_image (PIL.Image): Input image
            
        Returns:
            numpy.ndarray: Feature vector or None if extraction fails
        """
        try:
            with torch.no_grad():
                # Preprocess and move to device
                img_tensor = self.preprocess(pil_image).unsqueeze(0).to(self.device)
                
                # Extract features
                features = self.model(img_tensor)
                
                return features.cpu().numpy().flatten()
        except Exception as e:
            print(f"Feature extraction error: {str(e)}")
            return None


#%% 4. IMAGE LOADING AND PROCESSING

def load_and_process_urls(excel_path, max_samples=100):
    """
    Load URLs from Excel file, download images, and process them
    
    Args:
        excel_path (str): Path to Excel file containing image URLs and sentiment labels
        max_samples (int): Maximum number of samples to process
        
    Returns:
        list: List of dictionaries containing processed images and metadata
    """
    # Read Excel file
    df = pd.read_excel(excel_path)
    
    valid_data = []
    
    # Set up request headers to avoid blocking
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Referer': 'https://www.xiaohongshu.com/'
    }
    
    for idx, row in df.iterrows():
        # Check if we've reached the sample limit
        if len(valid_data) >= max_samples:
            print(f"Reached max sample size of {max_samples}")
            break
            
        try:
            # Get original URL with parameters
            raw_url = row['Image_Path']
            sentiment = row['Sentiment']
            
            # Parse and clean URL
            parsed = urlparse(raw_url)
            
            # Define base URL without parameters
            clean_url = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
            
            # Preserve only safe parameters
            safe_params = ['format', 'w', 'h', 'q']
            query_params = parse_qs(parsed.query)
            filtered_params = {k: v for k, v in query_params.items() if k in safe_params}
            
            # Rebuild final URL
            final_url = urlunparse((
                parsed.scheme,
                parsed.netloc,
                parsed.path,
                parsed.params,
                urlencode(filtered_params, doseq=True),
                parsed.fragment
            ))
            
            # Download with headers and timeout
            response = requests.get(
                final_url,
                headers=headers,
                timeout=(3.05, 15),  # Connect timeout, read timeout
                stream=True  # Better for large images
            )
            
            # Validate response
            if response.status_code != 200:
                raise ValueError(f"HTTP {response.status_code}")
                
            if 'image' not in response.headers['Content-Type']:
                raise ValueError("Not an image response")
                
            # Process image
            img = Image.open(BytesIO(response.content))
            img.verify()  # Verify it's a valid image file
            
            valid_data.append({
                'original_url': raw_url,
                'processed_url': final_url,
                'image': Image.open(BytesIO(response.content)),  # Reopen after verify
                'sentiment': 1 if str(sentiment).lower() == 'positive' else 0
            })
            
            print(f"Processed: {final_url}")
            
        except Exception as e:
            error_msg = f"Row {idx} failed - {type(e).__name__}: {str(e)}"
            if 'raw_url' in locals():
                error_msg += f" | URL: {raw_url}"
            print(error_msg)
    
    return valid_data


#%% 5. CLUSTERING WORKFLOW

def cluster_analysis(valid_data, max_clusters=15, max_samples=None):
    """
    Perform feature extraction and clustering on image data
    
    Args:
        valid_data (list): List of dictionaries containing images and metadata
        max_clusters (int): Maximum number of clusters to create
        max_samples (int): Maximum number of samples to process
        
    Returns:
        tuple: (features_reduced, clusters, urls, sentiments, linkage_matrix)
            - features_reduced: PCA-reduced feature vectors
            - clusters: Cluster assignments
            - urls: Image URLs
            - sentiments: Sentiment labels
            - linkage_matrix: Hierarchical clustering linkage matrix for dendrogram
    """
    # Initialize model
    model = CustomModel()
    
    # Initialize lists to store data
    features = []
    sentiments = []
    urls = []
    
    # Process each item in the dataset
    for item in valid_data:
        # Check if we've reached the sample limit
        if max_samples and len(features) >= max_samples:
            break
            
        try:
            # Extract features using := assignment expression (Python 3.8+)
            if (feats := model.extract_features(item['image'])) is not None:
                features.append(feats)
                sentiments.append(item['sentiment'])
                urls.append(item['processed_url'])
        except Exception as e:
            print(f"Error processing {item['processed_url']}: {str(e)}")
            continue
    
    # Convert lists to numpy arrays
    features = np.array(features)
    sentiments = np.array(sentiments)
    
    # Check for empty features
    if len(features) == 0:
        raise ValueError("No features extracted - check image processing")
    
    # Check feature dimensions
    if features.ndim == 1:
        features = features.reshape(-1, 1)
    
    # Standardize features
    features = StandardScaler().fit_transform(features)
    
    # Apply PCA to reduce dimensionality while preserving 95% of variance
    pca = PCA(n_components=0.95)
    features_reduced = pca.fit_transform(features)
    
    # Perform hierarchical clustering
    print("Performing hierarchical clustering...")
    distance_matrix = pdist(features_reduced, metric='cosine')
    Z = linkage(distance_matrix, method='ward')
    clusters = fcluster(Z, t=max_clusters, criterion='maxclust')
    
    return features_reduced, clusters, urls, sentiments, Z


#%% 6. VISUALIZATION & ANALYSIS

def plot_sentiment_cluster_heatmap(clusters, sentiments):
    """
    Create a heatmap showing sentiment distribution across clusters
    
    Args:
        clusters (numpy.ndarray): Cluster assignments
        sentiments (numpy.ndarray): Sentiment labels
    """
    plt.figure(figsize=(12, 6))
    
    # Create cross-tabulation
    cluster_sentiment = pd.crosstab(
        index=clusters,
        columns=sentiments,
        normalize='index'
    )
    
    # Plot heatmap
    sns.heatmap(cluster_sentiment, annot=True, fmt=".1%", cmap="Blues")
    plt.title("Sentiment Distribution per Cluster")
    plt.xlabel("Sentiment (0=Negative, 1=Positive)")
    plt.ylabel("Cluster ID")
    plt.show()


def calculate_cluster_purity(clusters, true_labels):
    """
    Quantify how well clusters match true sentiment labels
    
    Args:
        clusters (numpy.ndarray): Cluster assignments
        true_labels (numpy.ndarray): True sentiment labels
        
    Returns:
        float: Purity score between 0 and 1
    """
    purity = 0
    
    for cluster_id in np.unique(clusters):
        cluster_mask = clusters == cluster_id
        
        if sum(cluster_mask) == 0:
            continue
            
        # Find majority class in this cluster
        majority = np.bincount(true_labels[cluster_mask]).argmax()
        
        # Add count of correctly assigned items
        purity += sum(true_labels[cluster_mask] == majority)
    
    return purity / len(clusters)

def plot_dendrogram(Z, max_d=None, truncate_mode='lastp', p=25):
    """
    Plot a dendrogram of the hierarchical clustering
    
    Args:
        Z (numpy.ndarray): Linkage matrix from hierarchical clustering
        max_d (float, optional): Max distance for horizontal cut line
        truncate_mode (str, optional): Truncation mode ('lastp' for last p clusters)
        p (int, optional): Number of clusters to show when using 'lastp' mode
    """
    plt.figure(figsize=(16, 8))
    
    # Create dendrogram with colors
    dendrogram(
        Z,
        truncate_mode=truncate_mode,
        p=p,
        leaf_rotation=90.,
        leaf_font_size=10.,
        show_contracted=True,
        color_threshold=0.7*max(Z[:,2])  # Color threshold at 70% of max distance
    )
    
    # Add a horizontal cut line if max_d is specified
    if max_d:
        plt.axhline(y=max_d, c='k', linestyle='--', label=f'Cut at distance {max_d:.2f}')
        plt.legend()
    
    plt.title('Hierarchical Clustering Dendrogram (truncated)')
    plt.xlabel('Cluster or Sample Index')
    plt.ylabel('Distance')
    plt.tight_layout()
    plt.show()#!/usr/bin/env python3

def visualize_results(features, clusters, urls, sentiments, valid_data, linkage_matrix):
    """
    Generate visualizations and analysis of clustering results
    
    Args:
        features (numpy.ndarray): Feature vectors
        clusters (numpy.ndarray): Cluster assignments
        urls (list): Image URLs
        sentiments (numpy.ndarray): Sentiment labels
        valid_data (list): List of dictionaries containing images and metadata
        linkage_matrix (numpy.ndarray): Hierarchical clustering linkage matrix
    """
    # 1. Plot dendrogram to visualize cluster hierarchy
    print("Generating dendrogram visualization...")
    plot_dendrogram(linkage_matrix, truncate_mode='lastp', p=25)
    
    # 2. t-SNE visualization
    print("Generating t-SNE visualization...")
    tsne = TSNE(n_components=2, random_state=42)
    vis_features = tsne.fit_transform(features)
    
    plt.figure(figsize=(15, 10))
    plt.scatter(vis_features[:, 0], vis_features[:, 1], c=clusters, cmap='tab20', alpha=0.7)
    plt.colorbar(label='Cluster')
    plt.title('Cluster Visualization (t-SNE)')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()
    
    # Sentiment-cluster relationship
    unique_clusters = np.unique(clusters)
    cluster_stats = []
    
    for c in unique_clusters:
        mask = clusters == c
        pos_ratio = np.mean(sentiments[mask])
        cluster_stats.append(pos_ratio)
    
    plt.figure(figsize=(10, 5))
    plt.bar(range(len(cluster_stats)), cluster_stats)
    plt.axhline(0.5, color='red', linestyle='--')
    plt.title('Positive Sentiment Ratio per Cluster')
    plt.xlabel('Cluster ID')
    plt.ylabel('Positive Ratio')
    plt.show()
    
    # Sentiment-cluster heatmap
    plot_sentiment_cluster_heatmap(clusters, sentiments)
    
    # Sample images per cluster
    plt.figure(figsize=(20, 3*len(unique_clusters)))
    samples_per_cluster = min(5, len(valid_data)//len(unique_clusters))
    
    for i, c in enumerate(unique_clusters):
        cluster_samples = np.where(clusters == c)[0][:samples_per_cluster]
        
        for j, idx in enumerate(cluster_samples):
            ax = plt.subplot(len(unique_clusters), samples_per_cluster, i*samples_per_cluster + j + 1)
            
            try:
                ax.imshow(valid_data[idx]['image'])
                true_sent = 'Pos' if valid_data[idx]['sentiment'] else 'Neg'
                ax.set_title(f"Cluster {c}\nPred: {'+' if sentiments[idx] else '-'}\nTrue: {true_sent}")
            except Exception as e:
                ax.set_title("Image error")
                
            ax.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # Calculate and display metrics
    purity = calculate_cluster_purity(clusters, sentiments)
    print(f"\nCluster Purity: {purity:.1%} of images match cluster majority sentiment")
    
    # Statistical significance testing
    contingency = pd.crosstab(clusters, sentiments)
    chi2, p, _, _ = chi2_contingency(contingency)
    
    print("\nStatistical Significance:")
    print(f"χ² = {chi2:.1f}, p = {p:.4f}")
    print("Significant association!" if p < 0.05 else "No significant relationship")


#%% 7. MAIN EXECUTION

if __name__ == "__main__":
    # Configuration
    TEST_RUN = True  # Set to False for full run
    SAMPLE_SIZE = 100
    MAX_CLUSTERS = 25  # Set to 25 clusters for dendrogram visualization
    
    # Define file path
    excel_path = "ML_Thomas_Draft_Project/ML_InputDoc_ML__Project_AdSentiment_Clustering_052025.xlsx"
    
    print(f"Starting {'test run' if TEST_RUN else 'full analysis'} with {SAMPLE_SIZE if TEST_RUN else 'all'} samples")
    
    # Load and process data
    valid_data = load_and_process_urls(excel_path, max_samples=SAMPLE_SIZE if TEST_RUN else None)
    print(f"Successfully processed {len(valid_data)} images")
    
    # Extract features and perform clustering
    features, clusters, urls, sentiments, linkage_matrix = cluster_analysis(valid_data, max_clusters=MAX_CLUSTERS)
    print(f"Performed clustering into {len(np.unique(clusters))} clusters")
    
    # Generate visualizations and analysis
    visualize_results(features, clusters, urls, sentiments, valid_data, linkage_matrix)
    
    # Save results to Excel
    output_df = pd.DataFrame({
        'URL': urls,
        'Cluster': clusters,
        'Predicted_Sentiment': sentiments,
        'Actual_Sentiment': [d['sentiment'] for d in valid_data]
    })
    
    output_name = "cluster_results_test.xlsx" if TEST_RUN else "cluster_results_full.xlsx"
    output_df.to_excel(output_name, index=False)
    
    print(f"\nResults saved to {output_name}")