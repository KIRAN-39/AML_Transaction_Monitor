"""
Model Evaluation and Comparison
Compares rule-based detection, Isolation Forest, K-Means, and hybrid approaches
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Import our modules
from feature_engineering import FeatureEngineer
from ml_models import AMLMLModels
from detection_rules import AMLDetectionRules
from risk_scorer import RiskScorer


class ModelEvaluator:
    """
    Evaluates and compares different detection approaches
    """
    
    def __init__(self):
        self.results = {}
        self.comparisons = None
    
    def evaluate_rules_based(self, df):
        """
        Evaluate rule-based detection
        
        Args:
            df: Transaction DataFrame
            
        Returns:
            Set of flagged transaction IDs
        """
        print("\n" + "=" * 70)
        print("EVALUATING RULE-BASED DETECTION")
        print("=" * 70 + "\n")
        
        detector = AMLDetectionRules()
        rule_results = detector.run_all_rules(df)
        
        # Combine all flagged transactions
        all_flagged_ids = set()
        for rule_name, flagged_df in rule_results.items():
            if len(flagged_df) > 0:
                all_flagged_ids.update(flagged_df['transaction_id'].tolist())
        
        print(f"\n✅ Rule-based detection flagged {len(all_flagged_ids)} unique transactions")
        
        self.results['rules_based'] = {
            'flagged_ids': all_flagged_ids,
            'count': len(all_flagged_ids),
            'method': 'Rules-Based'
        }
        
        return all_flagged_ids
    
    def evaluate_isolation_forest(self, df, contamination=0.05):
        """
        Evaluate Isolation Forest model
        
        Args:
            df: Transaction DataFrame
            contamination: Expected anomaly rate
            
        Returns:
            Set of flagged transaction IDs
        """
        print("\n" + "=" * 70)
        print("EVALUATING ISOLATION FOREST MODEL")
        print("=" * 70 + "\n")
        
        # Feature engineering
        engineer = FeatureEngineer()
        X, feature_names, df_features = engineer.prepare_for_ml(df)
        
        # Train model
        ml_models = AMLMLModels()
        X_scaled = ml_models.preprocess_features(X, fit=True)
        ml_models.train_isolation_forest(X_scaled, contamination=contamination)
        
        # Predict
        predictions, anomaly_scores = ml_models.predict_isolation_forest(X_scaled)
        
        # Get flagged transaction IDs
        df_features['ml_prediction'] = predictions
        df_features['ml_anomaly_score'] = anomaly_scores
        
        flagged_ids = set(df_features[df_features['ml_prediction'] == -1]['transaction_id'].tolist())
        
        print(f"\n✅ Isolation Forest flagged {len(flagged_ids)} transactions")
        
        self.results['isolation_forest'] = {
            'flagged_ids': flagged_ids,
            'count': len(flagged_ids),
            'method': 'Isolation Forest',
            'scores': anomaly_scores,
            'df': df_features,
            'model': ml_models
        }
        
        return flagged_ids
    
    def evaluate_kmeans(self, df, n_clusters=5, percentile=95):
        """
        Evaluate K-Means clustering model
        
        Args:
            df: Transaction DataFrame
            n_clusters: Number of clusters
            percentile: Outlier threshold percentile
            
        Returns:
            Set of flagged transaction IDs
        """
        print("\n" + "=" * 70)
        print("EVALUATING K-MEANS CLUSTERING MODEL")
        print("=" * 70 + "\n")
        
        # Feature engineering
        engineer = FeatureEngineer()
        X, feature_names, df_features = engineer.prepare_for_ml(df)
        
        # Train model
        ml_models = AMLMLModels()
        X_scaled = ml_models.preprocess_features(X, fit=True)
        ml_models.train_kmeans_clustering(X_scaled, n_clusters=n_clusters)
        
        # Predict
        cluster_labels, distances, is_outlier = ml_models.predict_kmeans_outliers(X_scaled, percentile=percentile)
        
        # Get flagged transaction IDs
        df_features['cluster'] = cluster_labels
        df_features['distance_to_centroid'] = distances
        df_features['is_outlier'] = is_outlier
        
        flagged_ids = set(df_features[df_features['is_outlier']]['transaction_id'].tolist())
        
        print(f"\n✅ K-Means flagged {len(flagged_ids)} transactions")
        
        self.results['kmeans'] = {
            'flagged_ids': flagged_ids,
            'count': len(flagged_ids),
            'method': 'K-Means Clustering',
            'distances': distances,
            'df': df_features
        }
        
        return flagged_ids
    
    def evaluate_hybrid(self, df):
        """
        Evaluate hybrid approach (Rules + ML)
        Flags transactions if EITHER rules OR ML detects them
        
        Args:
            df: Transaction DataFrame
            
        Returns:
            Set of flagged transaction IDs
        """
        print("\n" + "=" * 70)
        print("EVALUATING HYBRID APPROACH (Rules + ML)")
        print("=" * 70 + "\n")
        
        # Get results from both approaches
        if 'rules_based' not in self.results:
            self.evaluate_rules_based(df)
        
        if 'isolation_forest' not in self.results:
            self.evaluate_isolation_forest(df)
        
        # Combine results (union)
        rules_ids = self.results['rules_based']['flagged_ids']
        ml_ids = self.results['isolation_forest']['flagged_ids']
        
        hybrid_ids = rules_ids | ml_ids  # Union
        
        # Also track intersection (high confidence)
        intersection_ids = rules_ids & ml_ids
        
        print(f"\nHybrid Detection Results:")
        print(f"   Rules only: {len(rules_ids - ml_ids)}")
        print(f"   ML only: {len(ml_ids - rules_ids)}")
        print(f"   Both (high confidence): {len(intersection_ids)}")
        print(f"   Total flagged: {len(hybrid_ids)}")
        
        self.results['hybrid'] = {
            'flagged_ids': hybrid_ids,
            'count': len(hybrid_ids),
            'method': 'Hybrid (Rules + ML)',
            'intersection': intersection_ids
        }
        
        return hybrid_ids
    
    def compare_with_ground_truth(self, df):
        """
        Compare all methods with ground truth labels (if available)
        
        Args:
            df: Transaction DataFrame with ground truth labels
        """
        if 'is_suspicious' not in df.columns:
            print("\n⚠️ No ground truth labels available for comparison")
            return None
        
        print("\n" + "=" * 70)
        print("PERFORMANCE COMPARISON WITH GROUND TRUTH")
        print("=" * 70 + "\n")
        
        # Get ground truth
        suspicious_df = df[df['is_suspicious'] == True]
        true_positive_ids = set(suspicious_df['transaction_id'].tolist())
        
        print(f"Ground Truth: {len(true_positive_ids)} suspicious transactions\n")
        
        # Evaluate each method
        comparison_results = []
        
        for method_name, result in self.results.items():
            flagged_ids = result['flagged_ids']
            
            # Calculate metrics
            tp = len(flagged_ids & true_positive_ids)  # True Positives
            fp = len(flagged_ids - true_positive_ids)  # False Positives
            fn = len(true_positive_ids - flagged_ids)  # False Negatives
            tn = len(df) - tp - fp - fn  # True Negatives
            
            # Calculate performance metrics
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            accuracy = (tp + tn) / len(df)
            
            comparison_results.append({
                'Method': result['method'],
                'Flagged': len(flagged_ids),
                'True Positives': tp,
                'False Positives': fp,
                'False Negatives': fn,
                'Precision': precision,
                'Recall': recall,
                'F1-Score': f1_score,
                'Accuracy': accuracy
            })
        
        # Create comparison DataFrame
        self.comparisons = pd.DataFrame(comparison_results)
        
        # Print results
        print("Performance Metrics:")
        print("=" * 70)
        print(self.comparisons.to_string(index=False))
        
        return self.comparisons
    
    def plot_comparison(self, save_path='model_comparison.png'):
        """
        Create visualization comparing model performance
        
        Args:
            save_path: Path to save the plot
        """
        if self.comparisons is None or len(self.comparisons) == 0:
            print("⚠️ No comparison data available. Run compare_with_ground_truth first.")
            return
        
        print(f"\nCreating comparison visualizations...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Precision, Recall, F1-Score comparison
        metrics = ['Precision', 'Recall', 'F1-Score']
        x = np.arange(len(self.comparisons))
        width = 0.25
        
        for i, metric in enumerate(metrics):
            axes[0, 0].bar(x + i*width, self.comparisons[metric], 
                          width, label=metric, alpha=0.8)
        
        axes[0, 0].set_xlabel('Method')
        axes[0, 0].set_ylabel('Score')
        axes[0, 0].set_title('Performance Metrics Comparison', fontweight='bold')
        axes[0, 0].set_xticks(x + width)
        axes[0, 0].set_xticklabels(self.comparisons['Method'], rotation=15, ha='right')
        axes[0, 0].legend()
        axes[0, 0].grid(axis='y', alpha=0.3)
        axes[0, 0].set_ylim(0, 1.1)
        
        # 2. True vs False Positives
        width = 0.35
        axes[0, 1].bar(x - width/2, self.comparisons['True Positives'], 
                      width, label='True Positives', color='green', alpha=0.7)
        axes[0, 1].bar(x + width/2, self.comparisons['False Positives'], 
                      width, label='False Positives', color='red', alpha=0.7)
        axes[0, 1].set_xlabel('Method')
        axes[0, 1].set_ylabel('Count')
        axes[0, 1].set_title('True vs False Positives', fontweight='bold')
        axes[0, 1].set_xticks(x)
        axes[0, 1].set_xticklabels(self.comparisons['Method'], rotation=15, ha='right')
        axes[0, 1].legend()
        axes[0, 1].grid(axis='y', alpha=0.3)
        
        # 3. Detection Rate (Recall) comparison
        axes[1, 0].barh(self.comparisons['Method'], self.comparisons['Recall'], 
                       color='steelblue', alpha=0.7)
        axes[1, 0].set_xlabel('Recall (Detection Rate)')
        axes[1, 0].set_title('Detection Rate by Method', fontweight='bold')
        axes[1, 0].grid(axis='x', alpha=0.3)
        axes[1, 0].set_xlim(0, 1)
        
        # Add value labels
        for i, (method, recall) in enumerate(zip(self.comparisons['Method'], 
                                                  self.comparisons['Recall'])):
            axes[1, 0].text(recall + 0.02, i, f'{recall:.2%}', 
                          va='center', fontsize=9)
        
        # 4. F1-Score (Overall Performance)
        colors = ['#2ecc71' if f1 > 0.7 else '#f39c12' if f1 > 0.5 else '#e74c3c' 
                 for f1 in self.comparisons['F1-Score']]
        
        axes[1, 1].barh(self.comparisons['Method'], self.comparisons['F1-Score'], 
                       color=colors, alpha=0.7)
        axes[1, 1].set_xlabel('F1-Score')
        axes[1, 1].set_title('Overall Performance (F1-Score)', fontweight='bold')
        axes[1, 1].grid(axis='x', alpha=0.3)
        axes[1, 1].set_xlim(0, 1)
        
        # Add value labels
        for i, (method, f1) in enumerate(zip(self.comparisons['Method'], 
                                             self.comparisons['F1-Score'])):
            axes[1, 1].text(f1 + 0.02, i, f'{f1:.2%}', 
                          va='center', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Comparison plot saved to {save_path}")
        plt.show()
    
    def generate_summary_report(self):
        """Generate text summary report"""
        print("\n" + "=" * 70)
        print("MODEL COMPARISON SUMMARY REPORT")
        print("=" * 70 + "\n")
        
        if self.comparisons is None:
            print("⚠️ Run comparison first")
            return
        
        # Find best performing method for each metric
        best_precision = self.comparisons.loc[self.comparisons['Precision'].idxmax()]
        best_recall = self.comparisons.loc[self.comparisons['Recall'].idxmax()]
        best_f1 = self.comparisons.loc[self.comparisons['F1-Score'].idxmax()]
        
        print("🏆 Best Performance by Metric:")
        print(f"   Precision: {best_precision['Method']} ({best_precision['Precision']:.2%})")
        print(f"   Recall: {best_recall['Method']} ({best_recall['Recall']:.2%})")
        print(f"   F1-Score: {best_f1['Method']} ({best_f1['F1-Score']:.2%})")
        
        print("\n📊 Detection Efficiency:")
        for _, row in self.comparisons.iterrows():
            efficiency = row['True Positives'] / row['Flagged'] if row['Flagged'] > 0 else 0
            print(f"   {row['Method']}: {efficiency:.1%} of flags are actual threats")
        
        print("\n💡 Recommendations:")
        
        # Determine best approach based on F1-Score
        if best_f1['Method'] == 'Hybrid (Rules + ML)':
            print("   ✓ Use HYBRID approach for best overall performance")
            print("   ✓ Combines strengths of both rule-based and ML methods")
        elif best_f1['F1-Score'] > 0.7:
            print(f"   ✓ {best_f1['Method']} shows excellent performance (F1 > 70%)")
        else:
            print("   ⚠️ Consider tuning thresholds or adding more features")
        
        print("\n" + "=" * 70)


def main():
    """
    Main execution: Load data, train models, compare performance
    """
    print("\n" + "=" * 70)
    print("AML MODEL EVALUATION & COMPARISON")
    print("=" * 70)
    print(f"\nEvaluation Run: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Load transaction data
    print("Loading transaction data...")
    try:
        df = pd.read_csv('transactions.csv')
        print(f"✅ Loaded {len(df):,} transactions\n")
    except FileNotFoundError:
        print("❌ Error: transactions.csv not found!")
        print("   Run data_generator.py first.")
        return
    
    # Initialize evaluator
    evaluator = ModelEvaluator()
    
    # Evaluate all methods
    evaluator.evaluate_rules_based(df)
    evaluator.evaluate_isolation_forest(df, contamination=0.05)
    evaluator.evaluate_kmeans(df, n_clusters=5)
    evaluator.evaluate_hybrid(df)
    
    # Compare with ground truth (if available)
    evaluator.compare_with_ground_truth(df)
    
    # Generate visualizations
    if evaluator.comparisons is not None:
        evaluator.plot_comparison('model_comparison.png')
    
    # Generate summary
    evaluator.generate_summary_report()
    
    print("\n✅ Model evaluation complete!")


if __name__ == "__main__":
    main()
