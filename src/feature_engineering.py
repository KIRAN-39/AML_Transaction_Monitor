"""
Feature Engineering for AML Machine Learning
Transforms transaction data into ML-ready features
"""

import pandas as pd
import numpy as np
from datetime import timedelta


class FeatureEngineer:
    """
    Creates features from transaction data for ML models
    """
    
    def __init__(self):
        self.feature_names = []
        self.scaler_params = None
    
    def create_all_features(self, df):
        """
        Create complete feature set from transaction data
        
        Args:
            df: DataFrame with transaction data
            
        Returns:
            DataFrame with engineered features
        """
        print("\n" + "=" * 70)
        print("FEATURE ENGINEERING FOR MACHINE LEARNING")
        print("=" * 70 + "\n")
        
        # Make a copy to avoid modifying original
        df_features = df.copy()
        
        # Ensure timestamp is datetime
        df_features['timestamp'] = pd.to_datetime(df_features['timestamp'])
        
        # Sort by user and time
        df_features = df_features.sort_values(['user_id', 'timestamp']).reset_index(drop=True)
        
        print("Creating features...")
        
        # 1. Basic transaction features
        df_features = self._add_basic_features(df_features)
        
        # 2. Time-based features
        df_features = self._add_time_features(df_features)
        
        # 3. User behavior features
        df_features = self._add_user_features(df_features)
        
        # 4. Rolling window features
        df_features = self._add_rolling_features(df_features)
        
        # 5. Amount-based features
        df_features = self._add_amount_features(df_features)
        
        # 6. Frequency features
        df_features = self._add_frequency_features(df_features)
        
        # Store feature names
        self.feature_names = [col for col in df_features.columns 
                             if col not in ['transaction_id', 'user_id', 'timestamp', 
                                          'merchant', 'counterparty', 'description', 
                                          'location', 'status', 'is_suspicious', 
                                          'suspicious_pattern', 'date']]
        
        print(f"\n✅ Created {len(self.feature_names)} features")
        print(f"   Feature list: {', '.join(self.feature_names[:10])}...")
        
        return df_features
    
    def _add_basic_features(self, df):
        """Add basic transaction features"""
        print("   → Basic features...")
        
        # Transaction amount (already exists, but ensure it's float)
        df['amount'] = df['amount'].astype(float)
        
        # Log-transformed amount (reduces impact of extreme values)
        df['amount_log'] = np.log1p(df['amount'])
        
        # Transaction type encoding
        if 'transaction_type' in df.columns:
            df['trans_type_encoded'] = pd.Categorical(df['transaction_type']).codes
        
        # Is round number
        df['is_round_1000'] = (df['amount'] % 1000 == 0).astype(int)
        df['is_round_5000'] = (df['amount'] % 5000 == 0).astype(int)
        
        # Amount bins
        df['amount_bin'] = pd.cut(df['amount'], 
                                  bins=[0, 100, 500, 1000, 5000, 10000, 50000, np.inf],
                                  labels=[0, 1, 2, 3, 4, 5, 6]).astype(int)
        
        return df
    
    def _add_time_features(self, df):
        """Add time-based features"""
        print("   → Time-based features...")
        
        # Hour of day
        df['hour'] = df['timestamp'].dt.hour
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        
        # Day of week (0=Monday, 6=Sunday)
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        
        # Day of month
        df['day_of_month'] = df['timestamp'].dt.day
        
        # Is business hours (9 AM - 5 PM, weekday)
        df['is_business_hours'] = (
            (df['hour'] >= 9) & 
            (df['hour'] <= 17) & 
            (df['day_of_week'] < 5)
        ).astype(int)
        
        # Is late night (11 PM - 5 AM)
        df['is_late_night'] = (
            (df['hour'] >= 23) | (df['hour'] <= 5)
        ).astype(int)
        
        return df
    
    def _add_user_features(self, df):
        """Add user-level aggregated features"""
        print("   → User behavior features...")
        
        # Group by user to calculate statistics
        user_stats = df.groupby('user_id').agg({
            'amount': ['mean', 'std', 'median', 'min', 'max', 'count'],
            'transaction_id': 'count'
        })
        
        # Flatten column names
        user_stats.columns = [
            'user_avg_amount', 'user_std_amount', 'user_median_amount',
            'user_min_amount', 'user_max_amount', 'user_amount_count',
            'user_total_transactions'
        ]
        
        # Fill NaN std with 0 (users with only 1 transaction)
        user_stats['user_std_amount'] = user_stats['user_std_amount'].fillna(0)
        
        # Merge back to main dataframe
        df = df.merge(user_stats, on='user_id', how='left')
        
        # Deviation from user's typical behavior
        df['amount_vs_user_avg'] = df['amount'] / (df['user_avg_amount'] + 1)
        df['amount_diff_from_avg'] = df['amount'] - df['user_avg_amount']
        
        # Z-score (how many standard deviations from user's mean)
        df['amount_zscore'] = np.where(
            df['user_std_amount'] > 0,
            (df['amount'] - df['user_avg_amount']) / df['user_std_amount'],
            0
        )
        
        return df
    
    def _add_rolling_features(self, df):
        """Add rolling window statistics"""
        print("   → Rolling window features...")
        
        # For each user, calculate rolling statistics
        rolling_features = []
        
        for user_id, user_df in df.groupby('user_id'):
            user_df = user_df.sort_values('timestamp').copy()
            
            # 7-day rolling average amount
            user_df['rolling_7d_avg'] = user_df['amount'].rolling(
                window=7, min_periods=1
            ).mean()
            
            # 7-day rolling std
            user_df['rolling_7d_std'] = user_df['amount'].rolling(
                window=7, min_periods=1
            ).std().fillna(0)
            
            # 30-day rolling average
            user_df['rolling_30d_avg'] = user_df['amount'].rolling(
                window=30, min_periods=1
            ).mean()
            
            # Transaction count in last 7 transactions
            user_df['recent_trans_count'] = range(1, len(user_df) + 1)
            user_df['recent_trans_count'] = user_df['recent_trans_count'].rolling(
                window=7, min_periods=1
            ).count()
            
            rolling_features.append(user_df)
        
        df = pd.concat(rolling_features, ignore_index=True)
        
        # Current transaction vs rolling average
        df['amount_vs_rolling_7d'] = df['amount'] / (df['rolling_7d_avg'] + 1)
        df['amount_vs_rolling_30d'] = df['amount'] / (df['rolling_30d_avg'] + 1)
        
        return df
    
    def _add_amount_features(self, df):
        """Add amount-related features"""
        print("   → Amount pattern features...")
        
        # Distance from common thresholds
        df['dist_from_10k'] = abs(df['amount'] - 10000)
        df['near_10k_threshold'] = (df['dist_from_10k'] < 1000).astype(int)
        
        # Is in structuring range (9000-9999)
        df['in_structuring_range'] = (
            (df['amount'] >= 9000) & (df['amount'] <= 9999)
        ).astype(int)
        
        # Decimal precision (suspicious if always whole numbers)
        df['has_decimals'] = (df['amount'] % 1 != 0).astype(int)
        
        return df
    
    def _add_frequency_features(self, df):
        """Add transaction frequency features"""
        print("   → Frequency features...")
        
        # Time since last transaction (for each user)
        df['time_since_last'] = df.groupby('user_id')['timestamp'].diff()
        df['hours_since_last'] = df['time_since_last'].dt.total_seconds() / 3600
        df['hours_since_last'] = df['hours_since_last'].fillna(0)
        
        # Days since last transaction
        df['days_since_last'] = df['hours_since_last'] / 24
        
        # Is rapid succession (< 1 hour since last)
        df['is_rapid_succession'] = (df['hours_since_last'] < 1).astype(int)
        
        # Is dormant reactivation (> 90 days since last)
        df['is_dormant_reactivation'] = (df['days_since_last'] > 90).astype(int)
        
        return df
    
    def get_feature_matrix(self, df):
        """
        Extract feature matrix for ML models
        
        Args:
            df: DataFrame with engineered features
            
        Returns:
            numpy array with features only, feature names
        """
        # Select only numeric features
        feature_cols = [col for col in self.feature_names 
                       if col in df.columns and df[col].dtype in ['int64', 'float64', 'int32', 'float32']]
        
        # Handle any remaining NaN values
        X = df[feature_cols].fillna(0).values
        
        return X, feature_cols
    
    def prepare_for_ml(self, df):
        """
        Complete pipeline: engineer features and prepare for ML
        
        Args:
            df: Raw transaction DataFrame
            
        Returns:
            Feature matrix (X), feature names, original DataFrame with features
        """
        # Create all features
        df_with_features = self.create_all_features(df)
        
        # Extract feature matrix
        X, feature_names = self.get_feature_matrix(df_with_features)
        
        print(f"\n✅ ML-Ready Dataset:")
        print(f"   Samples: {X.shape[0]:,}")
        print(f"   Features: {X.shape[1]}")
        print(f"   Feature names: {feature_names[:5]}...")
        
        return X, feature_names, df_with_features


def test_feature_engineering():
    """Test feature engineering on sample data"""
    print("Testing Feature Engineering...")
    
    # Create sample data
    sample_data = pd.DataFrame({
        'transaction_id': [f'T{i}' for i in range(1, 101)],
        'user_id': ['U1'] * 50 + ['U2'] * 50,
        'amount': np.random.uniform(100, 5000, 100),
        'timestamp': pd.date_range('2024-01-01', periods=100, freq='D'),
        'transaction_type': np.random.choice(['purchase', 'transfer', 'withdrawal'], 100)
    })
    
    # Add some suspicious patterns
    sample_data.loc[10:15, 'amount'] = 9500  # Structuring
    sample_data.loc[20:25, 'amount'] = 15000  # High value
    
    # Engineer features
    engineer = FeatureEngineer()
    X, feature_names, df_features = engineer.prepare_for_ml(sample_data)
    
    print("\n" + "=" * 70)
    print("Sample of Engineered Features:")
    print("=" * 70)
    print(df_features[['transaction_id', 'amount', 'amount_log', 
                       'hour', 'is_business_hours', 'user_avg_amount', 
                       'amount_vs_user_avg']].head(10))
    
    print("\n" + "=" * 70)
    print("Feature Matrix Shape:", X.shape)
    print("=" * 70)


if __name__ == "__main__":
    test_feature_engineering()
    
    print("\n" + "=" * 70)
    print("✅ Feature Engineering Module Ready!")
    print("=" * 70)
    print("\nTo use in your project:")
    print("  from feature_engineering import FeatureEngineer")
    print("  engineer = FeatureEngineer()")
    print("  X, feature_names, df = engineer.prepare_for_ml(transactions_df)")
