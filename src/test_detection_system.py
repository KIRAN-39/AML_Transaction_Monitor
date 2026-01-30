"""
End-to-End AML Detection System Test
Runs complete pipeline: Load Data → Detect → Score → Store → Report
"""

import pandas as pd
import sys
from datetime import datetime

# Import our modules
from detection_rules import AMLDetectionRules
from risk_scorer import RiskScorer
from alert_manager import AlertManager


def print_header(title):
    """Print formatted section header"""
    print("\n" + "=" * 70)
    print(title.center(70))
    print("=" * 70 + "\n")


def load_transaction_data(filepath='transactions.csv'):
    """
    Load transaction data from CSV
    
    Args:
        filepath: Path to transactions CSV file
        
    Returns:
        DataFrame with transaction data
    """
    print_header("STEP 1: LOADING TRANSACTION DATA")
    
    try:
        df = pd.read_csv(filepath)
        print(f"✅ Loaded {len(df):,} transactions from {filepath}")
        print(f"   Date Range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        print(f"   Unique Users: {df['user_id'].nunique():,}")
        print(f"   Total Volume: ${df['amount'].sum():,.2f}")
        
        # Show suspicious pattern breakdown if available
        if 'is_suspicious' in df.columns:
            suspicious_count = df['is_suspicious'].sum()
            print(f"   Pre-labeled Suspicious: {suspicious_count} ({suspicious_count/len(df)*100:.2f}%)")
        
        return df
        
    except FileNotFoundError:
        print(f"❌ Error: {filepath} not found!")
        print("   Please run 'python data_generator.py' first to generate data.")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        sys.exit(1)


def run_detection_rules(df):
    """
    Run all detection rules on transaction data
    
    Args:
        df: DataFrame with transactions
        
    Returns:
        Dictionary with detection results
    """
    print_header("STEP 2: RUNNING DETECTION RULES")
    
    detector = AMLDetectionRules()
    results = detector.run_all_rules(df)
    
    return results


def calculate_risk_scores(rule_results):
    """
    Calculate risk scores from rule results
    
    Args:
        rule_results: Dictionary with detection results
        
    Returns:
        DataFrame with scored transactions
    """
    print_header("STEP 3: CALCULATING RISK SCORES")
    
    scorer = RiskScorer()
    
    # Combine results
    combined_df = scorer.combine_rule_results(rule_results)
    
    # Calculate scores
    scored_df = scorer.calculate_cumulative_scores(combined_df)
    
    # Print summary
    if len(scored_df) > 0:
        scorer.print_risk_summary(scored_df)
    
    return scored_df


def store_alerts(scored_df, db_path='aml_database.db'):
    """
    Store alerts in database
    
    Args:
        scored_df: DataFrame with scored transactions
        db_path: Path to database
        
    Returns:
        AlertManager instance
    """
    print_header("STEP 4: STORING ALERTS IN DATABASE")
    
    manager = AlertManager(db_path)
    
    if len(scored_df) > 0:
        manager.store_alerts(scored_df, overwrite=True)
        manager.print_alert_summary()
    else:
        print("⚠️ No alerts to store")
    
    return manager


def analyze_results(scored_df, alert_manager):
    """
    Analyze and report results
    
    Args:
        scored_df: DataFrame with scored transactions
        alert_manager: AlertManager instance
    """
    print_header("STEP 5: DETAILED ANALYSIS")
    
    if len(scored_df) == 0:
        print("⚠️ No flagged transactions to analyze")
        return
    
    # Top 10 riskiest transactions
    print("🔴 TOP 10 RISKIEST TRANSACTIONS:")
    print("-" * 70)
    
    top_10 = scored_df.nlargest(10, 'final_risk_score')
    
    for idx, row in top_10.iterrows():
        print(f"\n{idx+1}. Transaction: {row['transaction_id']}")
        print(f"   User: {row['user_id']}")
        print(f"   Amount: ${row['amount']:,.2f}")
        print(f"   Type: {row['transaction_type']}")
        print(f"   Rules: {row['rule_triggered']}")
        print(f"   Risk Score: {row['final_risk_score']} ({row['risk_level']})")
    
    # High-risk users
    print("\n" + "-" * 70)
    print("\n👥 HIGH-RISK USERS (Multiple Alerts):")
    print("-" * 70)
    
    scorer = RiskScorer()
    high_risk_users = scorer.get_high_risk_users(scored_df, threshold='HIGH')
    
    if len(high_risk_users) > 0:
        print(high_risk_users.head(10).to_string(index=False))
    else:
        print("   No users with multiple high-risk transactions")
    
    # Rule effectiveness
    print("\n" + "-" * 70)
    print("\n📊 RULE EFFECTIVENESS:")
    print("-" * 70)
    
    rule_counts = {}
    for rules in scored_df['rule_triggered']:
        for rule in rules.split(', '):
            rule_counts[rule] = rule_counts.get(rule, 0) + 1
    
    for rule, count in sorted(rule_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"   {rule}: {count} detections")


def compare_with_ground_truth(df, scored_df):
    """
    Compare detection results with pre-labeled suspicious transactions
    
    Args:
        df: Original DataFrame with ground truth labels
        scored_df: DataFrame with detection results
    """
    print_header("STEP 6: PERFORMANCE EVALUATION")
    
    if 'is_suspicious' not in df.columns or 'suspicious_pattern' not in df.columns:
        print("⚠️ Ground truth labels not available (generated data doesn't have them)")
        print("   This comparison requires the 'is_suspicious' column in your data")
        return
    
    # Get flagged transaction IDs
    flagged_ids = set(scored_df['transaction_id'].tolist())
    
    # Get ground truth suspicious IDs
    suspicious_df = df[df['is_suspicious'] == True]
    suspicious_ids = set(suspicious_df['transaction_id'].tolist())
    
    # Calculate metrics
    true_positives = len(flagged_ids & suspicious_ids)
    false_positives = len(flagged_ids - suspicious_ids)
    false_negatives = len(suspicious_ids - flagged_ids)
    true_negatives = len(df) - true_positives - false_positives - false_negatives
    
    # Calculate performance metrics
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    print("📈 Detection Performance Metrics:")
    print("-" * 70)
    print(f"\nConfusion Matrix:")
    print(f"   True Positives:  {true_positives:,} (correctly flagged suspicious)")
    print(f"   False Positives: {false_positives:,} (incorrectly flagged normal)")
    print(f"   True Negatives:  {true_negatives:,} (correctly identified as normal)")
    print(f"   False Negatives: {false_negatives:,} (missed suspicious)")
    
    print(f"\nPerformance Scores:")
    print(f"   Precision: {precision:.2%} (of flagged transactions, how many were actually suspicious)")
    print(f"   Recall:    {recall:.2%} (of suspicious transactions, how many did we catch)")
    print(f"   F1-Score:  {f1_score:.2%} (overall detection quality)")
    
    # Pattern-by-pattern analysis
    print(f"\n🎯 Detection by Pattern Type:")
    print("-" * 70)
    
    for pattern in suspicious_df['suspicious_pattern'].unique():
        pattern_ids = set(suspicious_df[suspicious_df['suspicious_pattern'] == pattern]['transaction_id'].tolist())
        detected = len(pattern_ids & flagged_ids)
        total = len(pattern_ids)
        detection_rate = detected / total if total > 0 else 0
        
        print(f"   {pattern}: {detected}/{total} detected ({detection_rate:.1%})")


def generate_summary_report(df, scored_df):
    """
    Generate final summary report
    
    Args:
        df: Original transaction DataFrame
        scored_df: Scored transactions DataFrame
    """
    print_header("FINAL SUMMARY REPORT")
    
    total_transactions = len(df)
    flagged_transactions = len(scored_df) if len(scored_df) > 0 else 0
    flag_rate = flagged_transactions / total_transactions * 100 if total_transactions > 0 else 0
    
    print(f"📊 Overall Statistics:")
    print(f"   Total Transactions Processed: {total_transactions:,}")
    print(f"   Transactions Flagged: {flagged_transactions:,} ({flag_rate:.2f}%)")
    print(f"   Transactions Clear: {total_transactions - flagged_transactions:,}")
    
    if len(scored_df) > 0:
        print(f"\n   Total Flagged Amount: ${scored_df['amount'].sum():,.2f}")
        print(f"   Average Risk Score: {scored_df['final_risk_score'].mean():.1f}")
        print(f"   Highest Risk Score: {scored_df['final_risk_score'].max():.0f}")
        print(f"   Users Requiring Review: {scored_df['user_id'].nunique():,}")
    
    print("\n✅ AML Detection System Test Complete!")
    print("\nNext Steps:")
    print("   1. Review high-risk alerts in database")
    print("   2. Investigate flagged users")
    print("   3. Tune rule thresholds if needed")
    print("   4. Build dashboard for visualization")
    print("   5. Implement ML models for enhanced detection")


def main():
    """
    Main execution function - runs complete detection pipeline
    """
    print("\n" + "=" * 70)
    print("AML TRANSACTION MONITORING SYSTEM - END-TO-END TEST".center(70))
    print("=" * 70)
    print(f"\nTest Run: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        # Step 1: Load data
        df = load_transaction_data('transactions.csv')
        
        # Step 2: Run detection rules
        rule_results = run_detection_rules(df)
        
        # Step 3: Calculate risk scores
        scored_df = calculate_risk_scores(rule_results)
        
        # Step 4: Store alerts
        alert_manager = store_alerts(scored_df, 'aml_database.db')
        
        # Step 5: Analyze results
        analyze_results(scored_df, alert_manager)
        
        # Step 6: Compare with ground truth (if available)
        compare_with_ground_truth(df, scored_df)
        
        # Step 7: Generate summary
        generate_summary_report(df, scored_df)
        
    except KeyboardInterrupt:
        print("\n\n⚠️ Test interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error during test: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()