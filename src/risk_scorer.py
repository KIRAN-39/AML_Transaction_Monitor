"""
Risk Scoring System
Combines multiple detection rules into a unified risk score
"""

import pandas as pd
import numpy as np


class RiskScorer:
    """
    Combines results from multiple detection rules and assigns overall risk scores
    """
    
    def __init__(self):
        # Define risk levels
        self.risk_levels = {
            'LOW': (0, 30),
            'MEDIUM': (31, 60),
            'HIGH': (61, 90),
            'CRITICAL': (91, float('inf'))
        }
        
        # Rule weights (if you want to weight certain rules more heavily)
        self.rule_weights = {
            'HIGH_VALUE': 1.0,
            'VELOCITY': 1.2,
            'STRUCTURING': 1.5,  # Most serious
            'DORMANT_REACTIVATION': 1.1,
            'ROUND_NUMBERS': 0.9
        }
    
    def combine_rule_results(self, rule_results_dict):
        """
        Combine results from all detection rules
        
        Args:
            rule_results_dict: Dictionary with rule names as keys and DataFrames as values
            
        Returns:
            DataFrame with combined flagged transactions and cumulative scores
        """
        print("\n" + "=" * 70)
        print("COMBINING RULE RESULTS & CALCULATING RISK SCORES")
        print("=" * 70 + "\n")
        
        all_flagged = []
        
        # Collect all flagged transactions
        for rule_name, flagged_df in rule_results_dict.items():
            if len(flagged_df) > 0:
                # Apply rule weight to score
                rule_key = flagged_df['rule_triggered'].iloc[0]
                weight = self.rule_weights.get(rule_key, 1.0)
                
                flagged_df = flagged_df.copy()
                flagged_df['weighted_score'] = flagged_df['risk_score'] * weight
                
                all_flagged.append(flagged_df)
        
        if not all_flagged:
            print("⚠️ No flagged transactions found")
            return pd.DataFrame()
        
        # Concatenate all flagged transactions
        combined_df = pd.concat(all_flagged, ignore_index=True)
        
        print(f"Total flagged transaction records: {len(combined_df)}")
        print(f"Unique transactions: {combined_df['transaction_id'].nunique()}")
        
        return combined_df
    
    def calculate_cumulative_scores(self, combined_df):
        """
        Calculate cumulative risk scores for transactions flagged by multiple rules
        
        Args:
            combined_df: DataFrame with all flagged transactions
            
        Returns:
            DataFrame with one row per transaction and cumulative risk score
        """
        if len(combined_df) == 0:
            return pd.DataFrame()
        
        print("\nCalculating cumulative risk scores...")
        
        # Group by transaction_id and aggregate
        agg_dict = {
            'user_id': 'first',
            'amount': 'first',
            'timestamp': 'first',
            'transaction_type': 'first',
            'rule_triggered': lambda x: ', '.join(sorted(set(x))),  # Combine rules
            'risk_score': 'sum',  # Sum base scores
            'weighted_score': 'sum'  # Sum weighted scores
        }
        
        # Include other useful columns if they exist
        optional_cols = ['merchant', 'counterparty', 'location', 'description']
        for col in optional_cols:
            if col in combined_df.columns:
                agg_dict[col] = 'first'
        
        scored_df = combined_df.groupby('transaction_id', as_index=False).agg(agg_dict)
        
        # Use weighted score as final score
        scored_df['final_risk_score'] = scored_df['weighted_score'].round(0).astype(int)
        
        # Cap at reasonable maximum
        scored_df['final_risk_score'] = scored_df['final_risk_score'].clip(upper=150)
        
        # Count how many rules triggered
        scored_df['rules_triggered_count'] = scored_df['rule_triggered'].apply(
            lambda x: len(x.split(', '))
        )
        
        # Assign risk level
        scored_df['risk_level'] = scored_df['final_risk_score'].apply(self._assign_risk_level)
        
        print(f"✓ Calculated scores for {len(scored_df)} unique transactions")
        
        return scored_df
    
    def _assign_risk_level(self, score):
        """
        Assign risk level based on score
        
        Args:
            score: Numerical risk score
            
        Returns:
            Risk level string (LOW, MEDIUM, HIGH, CRITICAL)
        """
        for level, (min_score, max_score) in self.risk_levels.items():
            if min_score <= score <= max_score:
                return level
        return 'CRITICAL'  # Fallback for very high scores
    
    def get_risk_distribution(self, scored_df):
        """
        Get distribution of transactions by risk level
        
        Args:
            scored_df: DataFrame with risk scores
            
        Returns:
            Dictionary with counts per risk level
        """
        if len(scored_df) == 0:
            return {}
        
        distribution = scored_df['risk_level'].value_counts().to_dict()
        
        # Ensure all levels are represented
        for level in ['LOW', 'MEDIUM', 'HIGH', 'CRITICAL']:
            if level not in distribution:
                distribution[level] = 0
        
        return distribution
    
    def get_top_risky_transactions(self, scored_df, n=10):
        """
        Get top N riskiest transactions
        
        Args:
            scored_df: DataFrame with risk scores
            n: Number of top transactions to return
            
        Returns:
            DataFrame with top risky transactions
        """
        if len(scored_df) == 0:
            return pd.DataFrame()
        
        top_risky = scored_df.nlargest(n, 'final_risk_score')
        return top_risky
    
    def get_high_risk_users(self, scored_df, threshold='HIGH'):
        """
        Get users with multiple high-risk transactions
        
        Args:
            scored_df: DataFrame with risk scores
            threshold: Minimum risk level to consider (default: HIGH)
            
        Returns:
            DataFrame with user risk profiles
        """
        if len(scored_df) == 0:
            return pd.DataFrame()
        
        # Filter by risk level
        if threshold == 'HIGH':
            filtered = scored_df[scored_df['risk_level'].isin(['HIGH', 'CRITICAL'])]
        elif threshold == 'CRITICAL':
            filtered = scored_df[scored_df['risk_level'] == 'CRITICAL']
        else:
            filtered = scored_df
        
        # Aggregate by user
        user_risk = filtered.groupby('user_id').agg({
            'transaction_id': 'count',
            'final_risk_score': ['mean', 'max', 'sum'],
            'amount': 'sum',
            'risk_level': lambda x: x.value_counts().index[0]  # Most common risk level
        }).reset_index()
        
        # Flatten column names
        user_risk.columns = [
            'user_id', 'flagged_transaction_count', 'avg_risk_score', 
            'max_risk_score', 'total_risk_score', 'total_amount', 'primary_risk_level'
        ]
        
        # Sort by total risk
        user_risk = user_risk.sort_values('total_risk_score', ascending=False)
        
        return user_risk
    
    def generate_risk_report(self, scored_df):
        """
        Generate comprehensive risk report
        
        Args:
            scored_df: DataFrame with risk scores
            
        Returns:
            Dictionary with report statistics
        """
        if len(scored_df) == 0:
            return {
                'total_flagged': 0,
                'risk_distribution': {},
                'average_score': 0,
                'users_flagged': 0
            }
        
        report = {
            'total_flagged': len(scored_df),
            'risk_distribution': self.get_risk_distribution(scored_df),
            'average_score': scored_df['final_risk_score'].mean(),
            'median_score': scored_df['final_risk_score'].median(),
            'max_score': scored_df['final_risk_score'].max(),
            'users_flagged': scored_df['user_id'].nunique(),
            'total_flagged_amount': scored_df['amount'].sum(),
            'rules_triggered_breakdown': scored_df['rule_triggered'].value_counts().to_dict()
        }
        
        return report
    
    def print_risk_summary(self, scored_df):
        """
        Print human-readable risk summary
        
        Args:
            scored_df: DataFrame with risk scores
        """
        if len(scored_df) == 0:
            print("⚠️ No risky transactions to report")
            return
        
        report = self.generate_risk_report(scored_df)
        distribution = report['risk_distribution']
        
        print("\n" + "=" * 70)
        print("RISK ASSESSMENT SUMMARY")
        print("=" * 70)
        print(f"\n📊 Overall Statistics:")
        print(f"   Total Flagged Transactions: {report['total_flagged']:,}")
        print(f"   Unique Users Flagged: {report['users_flagged']:,}")
        print(f"   Total Flagged Amount: ${report['total_flagged_amount']:,.2f}")
        print(f"   Average Risk Score: {report['average_score']:.1f}")
        print(f"   Median Risk Score: {report['median_score']:.1f}")
        print(f"   Maximum Risk Score: {report['max_score']:.0f}")
        
        print(f"\n🚦 Risk Level Distribution:")
        total = sum(distribution.values())
        for level in ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW']:
            count = distribution.get(level, 0)
            percentage = (count / total * 100) if total > 0 else 0
            
            # Color coding for terminal
            if level == 'CRITICAL':
                icon = "🔴"
            elif level == 'HIGH':
                icon = "🟠"
            elif level == 'MEDIUM':
                icon = "🟡"
            else:
                icon = "🟢"
            
            print(f"   {icon} {level}: {count:,} ({percentage:.1f}%)")
        
        print(f"\n🎯 Top Risk Patterns:")
        rules_breakdown = report['rules_triggered_breakdown']
        for i, (rule, count) in enumerate(list(rules_breakdown.items())[:5], 1):
            print(f"   {i}. {rule}: {count} transactions")
        
        print("\n" + "=" * 70 + "\n")


def test_risk_scorer():
    """Test the risk scoring system"""
    print("Testing Risk Scorer...")
    
    # Create sample flagged transactions
    sample_data = pd.DataFrame({
        'transaction_id': ['T1', 'T1', 'T2', 'T3'],  # T1 flagged by 2 rules
        'user_id': ['U1', 'U1', 'U2', 'U3'],
        'amount': [15000, 15000, 25000, 9500],
        'timestamp': pd.date_range('2024-01-01', periods=4, freq='D'),
        'transaction_type': ['wire_transfer', 'wire_transfer', 'wire_transfer', 'transfer'],
        'rule_triggered': ['HIGH_VALUE', 'VELOCITY', 'HIGH_VALUE', 'STRUCTURING'],
        'risk_score': [30, 40, 50, 50],
        'weighted_score': [30, 48, 50, 75]
    })
    
    scorer = RiskScorer()
    scored_df = scorer.calculate_cumulative_scores(sample_data)
    
    print("\nScored Transactions:")
    print(scored_df[['transaction_id', 'amount', 'rule_triggered', 'final_risk_score', 'risk_level']])
    
    scorer.print_risk_summary(scored_df)


if __name__ == "__main__":
    test_risk_scorer()
    
    print("\n" + "=" * 70)
    print("✅ Risk Scoring Module Ready!")
    print("=" * 70)