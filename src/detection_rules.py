"""
AML Detection Rules
Implements rule-based detection for suspicious transaction patterns
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta


class AMLDetectionRules:
    """
    Collection of rule-based detection methods for AML monitoring
    """
    
    def __init__(self):
        self.rules_triggered = []
    
    def detect_high_value_transactions(self, df, threshold=10000):
        """
        Rule 1: High-Value Transaction Detection
        
        Flags transactions above a specified threshold.
        Regulatory requirement: Transactions over $10,000 must be reported.
        
        Args:
            df: DataFrame with transaction data
            threshold: Dollar amount threshold (default: $10,000)
            
        Returns:
            DataFrame with flagged transactions
        """
        print(f"Running Rule 1: High-Value Detection (threshold: ${threshold:,})")
        
        flagged = df[df['amount'] > threshold].copy()
        
        if len(flagged) > 0:
            flagged['rule_triggered'] = 'HIGH_VALUE'
            flagged['risk_score'] = 30  # Base score
            
            # Increase score for extremely high values
            flagged.loc[flagged['amount'] > 50000, 'risk_score'] = 50
            flagged.loc[flagged['amount'] > 100000, 'risk_score'] = 70
            
            print(f"   ✓ Flagged {len(flagged)} high-value transactions")
        else:
            print(f"   ✓ No high-value transactions found")
            
        return flagged
    
    def detect_velocity_anomalies(self, df, time_window_hours=24, transaction_threshold=10):
        """
        Rule 2: Velocity Detection
        
        Flags users with unusual number of transactions in short time period.
        Indicates possible account takeover or rapid fund movement.
        
        Args:
            df: DataFrame with transaction data
            time_window_hours: Time window to check (default: 24 hours)
            transaction_threshold: Number of transactions to trigger alert (default: 10)
            
        Returns:
            DataFrame with flagged transactions
        """
        print(f"Running Rule 2: Velocity Detection ({transaction_threshold}+ transactions in {time_window_hours}h)")
        
        df_sorted = df.sort_values(['user_id', 'timestamp']).copy()
        df_sorted['timestamp'] = pd.to_datetime(df_sorted['timestamp'])
        
        flagged_list = []
        
        # Group by user
        for user_id, user_trans in df_sorted.groupby('user_id'):
            user_trans = user_trans.reset_index(drop=True)
            
            # Check each transaction
            for idx, row in user_trans.iterrows():
                current_time = row['timestamp']
                
                # Count transactions within time window
                time_window_start = current_time - timedelta(hours=time_window_hours)
                recent_trans = user_trans[
                    (user_trans['timestamp'] >= time_window_start) & 
                    (user_trans['timestamp'] <= current_time)
                ]
                
                # Flag if threshold exceeded
                if len(recent_trans) >= transaction_threshold:
                    flagged_trans = recent_trans.copy()
                    flagged_trans['rule_triggered'] = 'VELOCITY'
                    
                    # Score based on severity
                    trans_count = len(recent_trans)
                    if trans_count >= 20:
                        flagged_trans['risk_score'] = 50
                    elif trans_count >= 15:
                        flagged_trans['risk_score'] = 40
                    else:
                        flagged_trans['risk_score'] = 30
                    
                    flagged_list.append(flagged_trans)
                    break  # Avoid duplicate flagging for same user
        
        if flagged_list:
            flagged = pd.concat(flagged_list, ignore_index=True)
            flagged = flagged.drop_duplicates(subset=['transaction_id'])
            print(f"   ✓ Flagged {len(flagged)} velocity anomalies")
        else:
            flagged = pd.DataFrame()
            print(f"   ✓ No velocity anomalies found")
            
        return flagged
    
    def detect_structuring(self, df, min_amount=9000, max_amount=9999, 
                          time_window_days=7, min_transactions=3):
        """
        Rule 3: Structuring (Smurfing) Detection
        
        Flags multiple transactions just below reporting threshold.
        Classic money laundering technique to avoid $10,000 reporting requirement.
        
        Args:
            df: DataFrame with transaction data
            min_amount: Minimum suspicious amount (default: $9,000)
            max_amount: Maximum suspicious amount (default: $9,999)
            time_window_days: Days to look for pattern (default: 7)
            min_transactions: Minimum transactions to flag (default: 3)
            
        Returns:
            DataFrame with flagged transactions
        """
        print(f"Running Rule 3: Structuring Detection (${min_amount}-${max_amount}, {min_transactions}+ in {time_window_days} days)")
        
        df_sorted = df.sort_values(['user_id', 'timestamp']).copy()
        df_sorted['timestamp'] = pd.to_datetime(df_sorted['timestamp'])
        
        # Filter to suspicious amount range
        suspicious_amounts = df_sorted[
            (df_sorted['amount'] >= min_amount) & 
            (df_sorted['amount'] <= max_amount)
        ].copy()
        
        flagged_list = []
        
        # Group by user
        for user_id, user_trans in suspicious_amounts.groupby('user_id'):
            user_trans = user_trans.reset_index(drop=True)
            
            if len(user_trans) < min_transactions:
                continue
            
            # Check for multiple transactions in time window
            for idx, row in user_trans.iterrows():
                current_time = row['timestamp']
                window_start = current_time - timedelta(days=time_window_days)
                
                recent_suspicious = user_trans[
                    (user_trans['timestamp'] >= window_start) & 
                    (user_trans['timestamp'] <= current_time)
                ]
                
                if len(recent_suspicious) >= min_transactions:
                    flagged_trans = recent_suspicious.copy()
                    flagged_trans['rule_triggered'] = 'STRUCTURING'
                    
                    # Higher score for more transactions
                    trans_count = len(recent_suspicious)
                    if trans_count >= 10:
                        flagged_trans['risk_score'] = 70
                    elif trans_count >= 5:
                        flagged_trans['risk_score'] = 50
                    else:
                        flagged_trans['risk_score'] = 40
                    
                    flagged_list.append(flagged_trans)
                    break
        
        if flagged_list:
            flagged = pd.concat(flagged_list, ignore_index=True)
            flagged = flagged.drop_duplicates(subset=['transaction_id'])
            print(f"   ✓ Flagged {len(flagged)} structuring patterns")
        else:
            flagged = pd.DataFrame()
            print(f"   ✓ No structuring patterns found")
            
        return flagged
    
    def detect_dormant_account_activity(self, df, dormant_period_days=90, 
                                       reactivation_threshold=5):
        """
        Rule 4: Dormant Account Reactivation Detection
        
        Flags accounts inactive for extended period that suddenly become active.
        May indicate account takeover or money laundering through old accounts.
        
        Args:
            df: DataFrame with transaction data
            dormant_period_days: Days of inactivity to consider dormant (default: 90)
            reactivation_threshold: Number of transactions after reactivation to flag (default: 5)
            
        Returns:
            DataFrame with flagged transactions
        """
        print(f"Running Rule 4: Dormant Account Detection ({dormant_period_days}+ days inactive)")
        
        df_sorted = df.sort_values(['user_id', 'timestamp']).copy()
        df_sorted['timestamp'] = pd.to_datetime(df_sorted['timestamp'])
        
        flagged_list = []
        
        # Analyze each user's transaction history
        for user_id, user_trans in df_sorted.groupby('user_id'):
            user_trans = user_trans.sort_values('timestamp').reset_index(drop=True)
            
            if len(user_trans) < 2:
                continue
            
            # Calculate time gaps between transactions
            user_trans['time_gap'] = user_trans['timestamp'].diff()
            
            # Find dormant periods
            dormant_mask = user_trans['time_gap'] > timedelta(days=dormant_period_days)
            
            if dormant_mask.any():
                # Get transactions after dormant period
                dormant_indices = user_trans[dormant_mask].index
                
                for dormant_idx in dormant_indices:
                    # Get transactions after reactivation
                    reactivation_trans = user_trans.loc[dormant_idx:dormant_idx + reactivation_threshold - 1]
                    
                    if len(reactivation_trans) >= reactivation_threshold:
                        flagged_trans = reactivation_trans.copy()
                        flagged_trans['rule_triggered'] = 'DORMANT_REACTIVATION'
                        
                        # Score based on dormancy period
                        gap_days = user_trans.loc[dormant_idx, 'time_gap'].days
                        if gap_days > 180:
                            flagged_trans['risk_score'] = 50
                        else:
                            flagged_trans['risk_score'] = 35
                        
                        flagged_list.append(flagged_trans)
        
        if flagged_list:
            flagged = pd.concat(flagged_list, ignore_index=True)
            flagged = flagged.drop_duplicates(subset=['transaction_id'])
            print(f"   ✓ Flagged {len(flagged)} dormant account reactivations")
        else:
            flagged = pd.DataFrame()
            print(f"   ✓ No dormant account patterns found")
            
        return flagged
    
    def detect_round_number_pattern(self, df, round_amounts=[5000, 10000, 15000, 20000, 25000],
                                   time_window_days=30, min_occurrences=3):
        """
        Rule 5: Round Number Transaction Pattern Detection
        
        Flags frequent exact round amount transactions.
        Legitimate transactions rarely use exact round numbers repeatedly.
        
        Args:
            df: DataFrame with transaction data
            round_amounts: List of suspicious round amounts
            time_window_days: Days to check for pattern (default: 30)
            min_occurrences: Minimum round transactions to flag (default: 3)
            
        Returns:
            DataFrame with flagged transactions
        """
        print(f"Running Rule 5: Round Number Pattern Detection ({min_occurrences}+ round amounts in {time_window_days} days)")
        
        df_sorted = df.sort_values(['user_id', 'timestamp']).copy()
        df_sorted['timestamp'] = pd.to_datetime(df_sorted['timestamp'])
        
        # Identify round number transactions
        df_sorted['is_round'] = df_sorted['amount'].apply(
            lambda x: x in round_amounts or x % 1000 == 0
        )
        
        flagged_list = []
        
        # Check each user
        for user_id, user_trans in df_sorted.groupby('user_id'):
            round_trans = user_trans[user_trans['is_round']].copy()
            
            if len(round_trans) < min_occurrences:
                continue
            
            # Check within time window
            round_trans = round_trans.sort_values('timestamp')
            
            for idx, row in round_trans.iterrows():
                current_time = row['timestamp']
                window_start = current_time - timedelta(days=time_window_days)
                
                recent_round = round_trans[
                    (round_trans['timestamp'] >= window_start) &
                    (round_trans['timestamp'] <= current_time)
                ]
                
                if len(recent_round) >= min_occurrences:
                    flagged_trans = recent_round.copy()
                    flagged_trans['rule_triggered'] = 'ROUND_NUMBERS'
                    
                    # Score based on frequency
                    count = len(recent_round)
                    if count >= 6:
                        flagged_trans['risk_score'] = 45
                    else:
                        flagged_trans['risk_score'] = 30
                    
                    flagged_list.append(flagged_trans)
                    break
        
        if flagged_list:
            flagged = pd.concat(flagged_list, ignore_index=True)
            flagged = flagged.drop_duplicates(subset=['transaction_id'])
            print(f"   ✓ Flagged {len(flagged)} round number patterns")
        else:
            flagged = pd.DataFrame()
            print(f"   ✓ No round number patterns found")
            
        return flagged
    
    def run_all_rules(self, df):
        """
        Run all detection rules on the dataset
        
        Args:
            df: DataFrame with transaction data
            
        Returns:
            Dictionary with results from each rule
        """
        print("\n" + "=" * 70)
        print("RUNNING ALL AML DETECTION RULES")
        print("=" * 70 + "\n")
        
        results = {
            'high_value': self.detect_high_value_transactions(df),
            'velocity': self.detect_velocity_anomalies(df),
            'structuring': self.detect_structuring(df),
            'dormant_account': self.detect_dormant_account_activity(df),
            'round_numbers': self.detect_round_number_pattern(df)
        }
        
        # Summary
        print("\n" + "=" * 70)
        print("DETECTION SUMMARY")
        print("=" * 70)
        
        total_flags = 0
        for rule_name, flagged_df in results.items():
            count = len(flagged_df) if len(flagged_df) > 0 else 0
            total_flags += count
            print(f"{rule_name.upper()}: {count} transactions flagged")
        
        print(f"\nTOTAL FLAGGED: {total_flags} (may have overlaps)")
        print("=" * 70 + "\n")
        
        return results


# Standalone test functions
def test_single_rule():
    """Test a single rule with sample data"""
    print("Testing High-Value Detection Rule...")
    
    # Create sample data
    sample_data = pd.DataFrame({
        'transaction_id': ['T1', 'T2', 'T3', 'T4'],
        'user_id': ['U1', 'U1', 'U2', 'U3'],
        'amount': [5000, 15000, 25000, 8000],
        'timestamp': pd.date_range('2024-01-01', periods=4, freq='D'),
        'transaction_type': ['purchase', 'wire_transfer', 'wire_transfer', 'purchase']
    })
    
    detector = AMLDetectionRules()
    flagged = detector.detect_high_value_transactions(sample_data, threshold=10000)
    
    print("\nFlagged Transactions:")
    print(flagged[['transaction_id', 'amount', 'rule_triggered', 'risk_score']])


if __name__ == "__main__":
    test_single_rule()
    
    print("\n" + "=" * 70)
    print("✅ Detection Rules Module Ready!")
    print("=" * 70)
    print("\nTo use in your project:")
    print("  from detection_rules import AMLDetectionRules")
    print("  detector = AMLDetectionRules()")
    print("  results = detector.run_all_rules(transactions_df)")