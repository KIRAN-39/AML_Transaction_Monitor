import sqlite3
import pandas as pd

# Create database
conn = sqlite3.connect('../data/aml_database.db')

# Load your CSV
df = pd.read_csv('../data/transactions.csv')

# Write to database
df.to_sql('transactions', conn, if_exists='replace', index=False)

# Create alerts table (for later)
conn.execute('''
    CREATE TABLE IF NOT EXISTS alerts (
        alert_id INTEGER PRIMARY KEY AUTOINCREMENT,
        transaction_id TEXT,
        rule_triggered TEXT,
        risk_score INTEGER,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
    )
''')

conn.close()
print("Database created successfully!")

# Query it
conn = sqlite3.connect('../data/aml_database.db')
query = "SELECT * FROM transactions WHERE amount > 10000"
result = pd.read_sql(query, conn)
print(result)
conn.close()