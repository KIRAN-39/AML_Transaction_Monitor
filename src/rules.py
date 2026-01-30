def detect_high_value_transactions(df, threshold=10000):
    """
    Flag transactions above a certain threshold
    """
    suspicious = df[df['amount'] > threshold].copy()
    suspicious['rule_triggered'] = 'HIGH_VALUE'
    suspicious['risk_score'] = 30  # Assign points
    
    return suspicious

# Test it
if __name__ == "__main__":
    import pandas as pd
    df = pd.read_csv('../data/transactions.csv')
    
    flagged = detect_high_value_transactions(df)
    print(f"Flagged {len(flagged)} high-value transactions")
    print(flagged.head())