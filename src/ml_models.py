"""
Machine Learning Models for AML Detection
Implements Isolation Forest and K-Means Clustering for anomaly detection
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pickle
import os


class AMLMLModels:
    """
    Machine Learning models for transaction anomaly detection
    """
    
    def __init__(self):
        self.isolation_forest = None
        self.kmeans = None
        self.scaler = StandardScaler()
        self.pca = None
        self.feature_names = None
        self.is_fitted = False
    
    def preprocess_features(self, X, fit=True):
        """
        Preprocess features: scaling and normalization
        
        Args:
            X: Feature matrix (numpy array or DataFrame)
            fit: If True, fit the scaler; if False, use existing scaler
            
        Returns:
            Scaled feature matrix
        """
        if fit:
            X_scaled = self.scaler.fit_transform(X)
            print("✓ Features scaled (fit_transform)")
        else:
            X_scaled = self.scaler.transform(X)
            print("✓ Features scaled (transform)")
        
        return X_scaled
    
    def train_isolation_forest(self, X, contamination=0.05, random_state=42):
        """
        Train Isolation Forest for anomaly detection
        
        Isolation Forest works by:
        - Building random decision trees
        - Anomalies are easier to isolate (fewer splits needed)
        - Assigns anomaly score to each sample
        
        Args:
            X: Feature matrix (already scaled)
            contamination: Expected proportion of anomalies (default: 5%)
            random_state: Random seed for reproducibility
            
        Returns:
            Trained Isolation Forest model
        """
        print("\n" + "=" * 70)
        print("TRAINING ISOLATION FOREST MODEL")
        print("=" * 70 + "\n")
        
        print(f"Training on {X.shape[0]:,} samples with {X.shape[1]} features")
        print(f"Expected contamination: {contamination*100}%")
        
        # Initialize model
        self.isolation_forest = IsolationForest(
            contamination=contamination,
            random_state=random_state,
            n_estimators=100,  # Number of trees
            max_samples='auto',
            max_features=1.0,
            bootstrap=False,
            n_jobs=-1,  # Use all CPU cores
            verbose=0
        )
        
        # Train model
        print("\nTraining Isolation Forest...")
        self.isolation_forest.fit(X)
        
        # Get predictions on training data
        predictions = self.isolation_forest.predict(X)
        anomaly_scores = self.isolation_forest.score_samples(X)
        
        # Count anomalies
        n_anomalies = (predictions == -1).sum()
        anomaly_rate = n_anomalies / len(predictions) * 100
        
        print(f"✅ Training complete!")
        print(f"\nTraining Set Results:")
        print(f"   Detected Anomalies: {n_anomalies:,} ({anomaly_rate:.2f}%)")
        print(f"   Normal Transactions: {(predictions == 1).sum():,}")
        print(f"   Anomaly Score Range: [{anomaly_scores.min():.3f}, {anomaly_scores.max():.3f}]")
        
        self.is_fitted = True
        return self.isolation_forest
    
    def predict_isolation_forest(self, X):
        """
        Predict anomalies using trained Isolation Forest
        
        Args:
            X: Feature matrix (scaled)
            
        Returns:
            predictions (-1 for anomaly, 1 for normal), anomaly scores
        """
        if self.isolation_forest is None:
            raise ValueError("Model not trained! Call train_isolation_forest first.")
        
        predictions = self.isolation_forest.predict(X)
        anomaly_scores = self.isolation_forest.score_samples(X)
        
        # Convert to more intuitive format
        # Higher score = more anomalous
        anomaly_scores_normalized = -anomaly_scores  # Flip so higher = more anomalous
        
        return predictions, anomaly_scores_normalized
    
    def train_kmeans_clustering(self, X, n_clusters=5, random_state=42):
        """
        Train K-Means clustering for anomaly detection
        
        K-Means works by:
        - Grouping similar transactions into clusters
        - Transactions far from cluster centers are anomalous
        - Distance from centroid = anomaly score
        
        Args:
            X: Feature matrix (already scaled)
            n_clusters: Number of clusters (default: 5)
            random_state: Random seed
            
        Returns:
            Trained K-Means model
        """
        print("\n" + "=" * 70)
        print("TRAINING K-MEANS CLUSTERING MODEL")
        print("=" * 70 + "\n")
        
        print(f"Training on {X.shape[0]:,} samples with {X.shape[1]} features")
        print(f"Number of clusters: {n_clusters}")
        
        # Initialize model
        self.kmeans = KMeans(
            n_clusters=n_clusters,
            random_state=random_state,
            n_init=10,
            max_iter=300,
            algorithm='lloyd'
        )
        
        # Train model
        print("\nTraining K-Means...")
        self.kmeans.fit(X)
        
        # Get cluster assignments
        cluster_labels = self.kmeans.labels_
        
        # Calculate distances from centroids
        distances = self.kmeans.transform(X)
        min_distances = distances.min(axis=1)
        
        # Find outliers (top 5% furthest from centroids)
        threshold = np.percentile(min_distances, 95)
        outliers = min_distances > threshold
        
        print(f"✅ Training complete!")
        print(f"\nCluster Distribution:")
        for i in range(n_clusters):
            count = (cluster_labels == i).sum()
            print(f"   Cluster {i}: {count:,} transactions ({count/len(cluster_labels)*100:.1f}%)")
        
        print(f"\nOutlier Detection (top 5% furthest from centers):")
        print(f"   Outliers: {outliers.sum():,} ({outliers.sum()/len(outliers)*100:.1f}%)")
        print(f"   Distance threshold: {threshold:.3f}")
        
        return self.kmeans
    
    def predict_kmeans_outliers(self, X, percentile=95):
        """
        Predict outliers using K-Means clustering
        
        Args:
            X: Feature matrix (scaled)
            percentile: Percentile threshold for outliers (default: 95)
            
        Returns:
            cluster labels, distances from centroids, outlier flags
        """
        if self.kmeans is None:
            raise ValueError("Model not trained! Call train_kmeans_clustering first.")
        
        # Get cluster assignments
        cluster_labels = self.kmeans.predict(X)
        
        # Calculate distances from centroids
        distances = self.kmeans.transform(X)
        min_distances = distances.min(axis=1)
        
        # Determine outliers
        threshold = np.percentile(min_distances, percentile)
        is_outlier = min_distances > threshold
        
        return cluster_labels, min_distances, is_outlier
    
    def apply_pca(self, X, n_components=10):
        """
        Apply PCA for dimensionality reduction (optional, for visualization)
        
        Args:
            X: Feature matrix
            n_components: Number of components to keep
            
        Returns:
            Reduced feature matrix
        """
        print(f"\nApplying PCA: {X.shape[1]} → {n_components} dimensions")
        
        self.pca = PCA(n_components=n_components, random_state=42)
        X_reduced = self.pca.fit_transform(X)
        
        variance_explained = self.pca.explained_variance_ratio_.sum() * 100
        print(f"✓ Variance explained: {variance_explained:.1f}%")
        
        return X_reduced
    
    def save_models(self, filepath='models/'):
        """
        Save trained models to disk
        
        Args:
            filepath: Directory to save models
        """
        os.makedirs(filepath, exist_ok=True)
        
        if self.isolation_forest is not None:
            with open(f'{filepath}isolation_forest.pkl', 'wb') as f:
                pickle.dump(self.isolation_forest, f)
            print(f"✓ Saved Isolation Forest to {filepath}isolation_forest.pkl")
        
        if self.kmeans is not None:
            with open(f'{filepath}kmeans.pkl', 'wb') as f:
                pickle.dump(self.kmeans, f)
            print(f"✓ Saved K-Means to {filepath}kmeans.pkl")
        
        if self.scaler is not None:
            with open(f'{filepath}scaler.pkl', 'wb') as f:
                pickle.dump(self.scaler, f)
            print(f"✓ Saved Scaler to {filepath}scaler.pkl")
    
    def load_models(self, filepath='models/'):
        """
        Load trained models from disk
        
        Args:
            filepath: Directory containing saved models
        """
        try:
            with open(f'{filepath}isolation_forest.pkl', 'rb') as f:
                self.isolation_forest = pickle.load(f)
            print(f"✓ Loaded Isolation Forest")
        except FileNotFoundError:
            print(f"⚠️ Isolation Forest model not found")
        
        try:
            with open(f'{filepath}kmeans.pkl', 'rb') as f:
                self.kmeans = pickle.load(f)
            print(f"✓ Loaded K-Means")
        except FileNotFoundError:
            print(f"⚠️ K-Means model not found")
        
        try:
            with open(f'{filepath}scaler.pkl', 'rb') as f:
                self.scaler = pickle.load(f)
            print(f"✓ Loaded Scaler")
        except FileNotFoundError:
            print(f"⚠️ Scaler not found")
        
        self.is_fitted = True
    
    def get_feature_importance(self, feature_names, X, n_top=10):
        """
        Get feature importance for Isolation Forest
        (Based on how often features are used for splitting)
        
        Args:
            feature_names: List of feature names
            X: Feature matrix
            n_top: Number of top features to return
            
        Returns:
            DataFrame with feature importance
        """
        if self.isolation_forest is None:
            print("⚠️ No Isolation Forest model trained")
            return None
        
        # Get anomaly scores
        scores = self.isolation_forest.score_samples(X)
        
        # Calculate correlation between each feature and anomaly score
        importances = []
        for i, feature in enumerate(feature_names):
            corr = np.corrcoef(X[:, i], scores)[0, 1]
            importances.append(abs(corr))
        
        # Create DataFrame
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        print(f"\nTop {n_top} Most Important Features:")
        print("=" * 50)
        for idx, row in importance_df.head(n_top).iterrows():
            print(f"   {row['feature']}: {row['importance']:.4f}")
        
        return importance_df


def test_ml_models():
    """Test ML models on sample data"""
    print("Testing ML Models...")
    
    # Create sample data
    np.random.seed(42)
    n_samples = 1000
    n_features = 10
    
    # Normal data
    X_normal = np.random.randn(n_samples, n_features)
    
    # Add some anomalies
    X_anomalies = np.random.randn(50, n_features) * 3 + 5
    X = np.vstack([X_normal, X_anomalies])
    
    # Initialize ML models
    ml_models = AMLMLModels()
    
    # Preprocess
    X_scaled = ml_models.preprocess_features(X, fit=True)
    
    # Train Isolation Forest
    ml_models.train_isolation_forest(X_scaled, contamination=0.05)
    
    # Predict
    predictions, scores = ml_models.predict_isolation_forest(X_scaled)
    print(f"\nIsolation Forest detected {(predictions == -1).sum()} anomalies")
    
    # Train K-Means
    ml_models.train_kmeans_clustering(X_scaled, n_clusters=3)
    
    # Predict
    labels, distances, outliers = ml_models.predict_kmeans_outliers(X_scaled)
    print(f"\nK-Means detected {outliers.sum()} outliers")
    
    print("\n✅ ML Models test complete!")


if __name__ == "__main__":
    test_ml_models()
    
    print("\n" + "=" * 70)
    print("✅ ML Models Module Ready!")
    print("=" * 70)
    print("\nTo use in your project:")
    print("  from ml_models import AMLMLModels")
    print("  ml = AMLMLModels()")
    print("  X_scaled = ml.preprocess_features(X)")
    print("  ml.train_isolation_forest(X_scaled)")
