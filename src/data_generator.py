"""
AML Transaction Data Generator
Generates synthetic transaction data with suspicious patterns for testing
"""

import pandas as pd
import numpy as np
from faker import Faker
from datetime import datetime, timedelta
import random
import uuid

# Initialize Faker
fake = Faker()
Faker.seed(42)
np.random.seed(42)
random.seed(42)

class TransactionGenerator:
    def __init__(self, num_users=1000, num_transactions=10):
        self.num_users = num_users
        self.num_transactions = num_transactions
        self.users = []
        self.transactions = []
        
        # Define transaction types and their typical amounts
        self.transaction_types = {
            'purchase': (10, 500),      # min, max typical amounts
            'transfer': (50, 2000),
            'withdrawal': (20, 1000),
            'deposit': (100, 5000),
            'wire_transfer': (1000, 50000)
        }
        
        # Date range: 6 months of data
        self.end_date = datetime.now()
        self.start_date = self.end_date - timedelta(days=180)
        
    def generate_users(self):
        """Generate user profiles"""
        print("Generating users...")
        
        for i in range(self.num_users):
            user = {
                'user_id': f'USER_{i+1:06d}',
                'name': fake.name(),
                'email': fake.email(),
                'phone': fake.phone_number(),
                'address': fake.address().replace('\n', ', '),
                'city': fake.city(),
                'state': fake.state_abbr(),
                'country': 'USA',
                'account_created_date': fake.date_between(
                    start_date='-3y', 
                    end_date='-6m'
                ),
                'account_type': random.choice(['checking', 'savings', 'business']),
                'risk_category': 'normal'  # Will update for suspicious users
            }
            self.users.append(user)
        
        return pd.DataFrame(self.users)
    
    def generate_normal_transactions(self, num_transactions):
        """Generate normal transaction patterns"""
        print(f"Generating {num_transactions} normal transactions...")
        
        transactions = []
        
        for _ in range(num_transactions):
            user = random.choice(self.users)
            trans_type = random.choice(list(self.transaction_types.keys()))
            min_amt, max_amt = self.transaction_types[trans_type]
            
            # Generate amount with some randomness
            if trans_type == 'wire_transfer':
                amount = round(np.random.exponential(scale=2000) + 1000, 2)
                amount = min(amount, max_amt)
            else:
                amount = round(random.uniform(min_amt, max_amt), 2)
            
            # Random timestamp within 6 months
            days_offset = random.randint(0, 180)
            hours_offset = random.randint(6, 22)  # Business hours mostly
            timestamp = self.start_date + timedelta(
                days=days_offset,
                hours=hours_offset,
                minutes=random.randint(0, 59)
            )
            
            transaction = {
                'transaction_id': str(uuid.uuid4()),
                'user_id': user['user_id'],
                'timestamp': timestamp,
                'amount': amount,
                'transaction_type': trans_type,
                'merchant': fake.company() if trans_type == 'purchase' else None,
                'counterparty': fake.name() if trans_type in ['transfer', 'wire_transfer'] else None,
                'location': f"{user['city']}, {user['state']}",
                'description': fake.sentence(nb_words=6),
                'status': 'completed',
                'is_suspicious': False,
                'suspicious_pattern': None
            }
            
            transactions.append(transaction)
        
        return transactions
    
    def inject_structuring_pattern(self):
        """
        Inject structuring pattern (smurfing)
        Multiple transactions just below $10,000 reporting threshold
        """
        print("Injecting structuring patterns...")
        
        num_structuring_users = random.randint(5, 10)
        structuring_transactions = []
        
        for _ in range(num_structuring_users):
            user = random.choice(self.users)
            user['risk_category'] = 'structuring'
            
            # Generate 5-15 transactions between $9,000 - $9,900
            num_trans = random.randint(5, 15)
            base_date = self.start_date + timedelta(days=random.randint(30, 150))
            
            for i in range(num_trans):
                # Spread over 2-7 days
                timestamp = base_date + timedelta(
                    days=random.randint(0, 7),
                    hours=random.randint(8, 20)
                )
                
                transaction = {
                    'transaction_id': str(uuid.uuid4()),
                    'user_id': user['user_id'],
                    'timestamp': timestamp,
                    'amount': round(random.uniform(9000, 9900), 2),
                    'transaction_type': 'wire_transfer',
                    'merchant': None,
                    'counterparty': fake.name(),
                    'location': f"{user['city']}, {user['state']}",
                    'description': 'Wire transfer',
                    'status': 'completed',
                    'is_suspicious': True,
                    'suspicious_pattern': 'structuring'
                }
                
                structuring_transactions.append(transaction)
        
        return structuring_transactions
    
    def inject_velocity_pattern(self):
        """
        Inject velocity pattern
        Unusual number of transactions in short time period
        """
        print("Injecting velocity patterns...")
        
        num_velocity_users = random.randint(3, 5)
        velocity_transactions = []
        
        for _ in range(num_velocity_users):
            user = random.choice(self.users)
            user['risk_category'] = 'velocity'
            
            # Generate 15-25 transactions in one day
            num_trans = random.randint(15, 25)
            spike_date = self.start_date + timedelta(days=random.randint(30, 150))
            
            for i in range(num_trans):
                timestamp = spike_date + timedelta(
                    hours=random.randint(0, 23),
                    minutes=random.randint(0, 59)
                )
                
                transaction = {
                    'transaction_id': str(uuid.uuid4()),
                    'user_id': user['user_id'],
                    'timestamp': timestamp,
                    'amount': round(random.uniform(100, 2000), 2),
                    'transaction_type': random.choice(['purchase', 'transfer', 'withdrawal']),
                    'merchant': fake.company(),
                    'counterparty': fake.name() if random.random() > 0.5 else None,
                    'location': f"{user['city']}, {user['state']}",
                    'description': fake.sentence(nb_words=4),
                    'status': 'completed',
                    'is_suspicious': True,
                    'suspicious_pattern': 'velocity'
                }
                
                velocity_transactions.append(transaction)
        
        return velocity_transactions
    
    def inject_dormant_account_pattern(self):
        """
        Inject dormant account suddenly active pattern
        Account inactive for months, then sudden activity
        """
        print("Injecting dormant account patterns...")
        
        num_dormant_users = random.randint(3, 5)
        dormant_transactions = []
        
        for _ in range(num_dormant_users):
            user = random.choice(self.users)
            user['risk_category'] = 'dormant_reactivation'
            
            # Last transaction was 4-5 months ago, then sudden activity
            dormant_end = self.end_date - timedelta(days=random.randint(10, 30))
            
            # Generate 8-15 recent transactions
            num_trans = random.randint(8, 15)
            
            for i in range(num_trans):
                timestamp = dormant_end + timedelta(
                    days=random.randint(0, 20),
                    hours=random.randint(0, 23)
                )
                
                transaction = {
                    'transaction_id': str(uuid.uuid4()),
                    'user_id': user['user_id'],
                    'timestamp': timestamp,
                    'amount': round(random.uniform(500, 5000), 2),
                    'transaction_type': random.choice(['wire_transfer', 'transfer', 'withdrawal']),
                    'merchant': None,
                    'counterparty': fake.name(),
                    'location': f"{user['city']}, {user['state']}",
                    'description': fake.sentence(nb_words=5),
                    'status': 'completed',
                    'is_suspicious': True,
                    'suspicious_pattern': 'dormant_reactivation'
                }
                
                dormant_transactions.append(transaction)
        
        return dormant_transactions
    
    def inject_high_value_pattern(self):
        """
        Inject unusually high-value transactions
        """
        print("Injecting high-value transaction patterns...")
        
        num_high_value = random.randint(20, 40)
        high_value_transactions = []
        
        for _ in range(num_high_value):
            user = random.choice(self.users)
            
            timestamp = self.start_date + timedelta(
                days=random.randint(0, 180),
                hours=random.randint(9, 17)
            )
            
            # High-value: $15,000 - $100,000
            amount = round(random.uniform(15000, 100000), 2)
            
            transaction = {
                'transaction_id': str(uuid.uuid4()),
                'user_id': user['user_id'],
                'timestamp': timestamp,
                'amount': amount,
                'transaction_type': 'wire_transfer',
                'merchant': None,
                'counterparty': fake.company(),
                'location': f"{user['city']}, {user['state']}",
                'description': 'Large wire transfer',
                'status': 'completed',
                'is_suspicious': True,
                'suspicious_pattern': 'high_value'
            }
            
            high_value_transactions.append(transaction)
        
        return high_value_transactions
    
    def inject_round_number_pattern(self):
        """
        Inject suspicious round number transactions
        Unusually frequent exact round amounts (often sign of money laundering)
        """
        print("Injecting round number patterns...")
        
        num_round_users = random.randint(2, 4)
        round_transactions = []
        
        round_amounts = [5000, 10000, 15000, 20000, 25000]
        
        for _ in range(num_round_users):
            user = random.choice(self.users)
            user['risk_category'] = 'round_numbers'
            
            # Generate 6-12 round number transactions
            num_trans = random.randint(6, 12)
            
            for i in range(num_trans):
                timestamp = self.start_date + timedelta(
                    days=random.randint(0, 180),
                    hours=random.randint(9, 18)
                )
                
                transaction = {
                    'transaction_id': str(uuid.uuid4()),
                    'user_id': user['user_id'],
                    'timestamp': timestamp,
                    'amount': float(random.choice(round_amounts)),
                    'transaction_type': 'transfer',
                    'merchant': None,
                    'counterparty': fake.name(),
                    'location': f"{user['city']}, {user['state']}",
                    'description': 'Round amount transfer',
                    'status': 'completed',
                    'is_suspicious': True,
                    'suspicious_pattern': 'round_numbers'
                }
                
                round_transactions.append(transaction)
        
        return round_transactions
    
    def generate_all_transactions(self):
        """Generate all transactions including suspicious patterns"""
        
        # Calculate normal transactions (majority)
        suspicious_trans_estimate = 200  # Approximate
        num_normal = self.num_transactions - suspicious_trans_estimate
        
        # Generate normal transactions
        self.transactions = self.generate_normal_transactions(num_normal)
        
        # Inject suspicious patterns
        self.transactions.extend(self.inject_structuring_pattern())
        self.transactions.extend(self.inject_velocity_pattern())
        self.transactions.extend(self.inject_dormant_account_pattern())
        self.transactions.extend(self.inject_high_value_pattern())
        self.transactions.extend(self.inject_round_number_pattern())
        
        # Shuffle transactions
        random.shuffle(self.transactions)
        
        # Trim to exact number if needed
        self.transactions = self.transactions[:self.num_transactions]
        
        return pd.DataFrame(self.transactions)
    
    def generate_dataset(self):
        """Main method to generate complete dataset"""
        print("=" * 60)
        print("AML Transaction Data Generator")
        print("=" * 60)
        
        # Generate users
        users_df = self.generate_users()
        
        # Generate transactions
        transactions_df = self.generate_all_transactions()
        
        # Sort by timestamp
        transactions_df = transactions_df.sort_values('timestamp').reset_index(drop=True)
        
        # Add some metadata columns
        transactions_df['date'] = pd.to_datetime(transactions_df['timestamp']).dt.date
        transactions_df['hour'] = pd.to_datetime(transactions_df['timestamp']).dt.hour
        transactions_df['day_of_week'] = pd.to_datetime(transactions_df['timestamp']).dt.dayofweek
        
        print("\n" + "=" * 60)
        print("Dataset Generation Complete!")
        print("=" * 60)
        print(f"\nTotal Users: {len(users_df)}")
        print(f"Total Transactions: {len(transactions_df)}")
        print(f"\nSuspicious Patterns Breakdown:")
        print(transactions_df['suspicious_pattern'].value_counts())
        print(f"\nSuspicious Transactions: {transactions_df['is_suspicious'].sum()}")
        print(f"Normal Transactions: {(~transactions_df['is_suspicious']).sum()}")
        print(f"Suspicious Rate: {transactions_df['is_suspicious'].sum() / len(transactions_df) * 100:.2f}%")
        
        return users_df, transactions_df


def main():
    """Main execution function"""
    
    # Initialize generator
    # Adjust these numbers based on your needs
    generator = TransactionGenerator(
        num_users=1000,        # Number of unique users
        num_transactions=50000  # Total transactions to generate
    )
    
    # Generate data
    users_df, transactions_df = generator.generate_dataset()
    
    # Save to CSV files
    print("\nSaving data to files...")
    users_df.to_csv('users.csv', index=False)
    transactions_df.to_csv('transactions.csv', index=False)
    
    print("\n✅ Files saved:")
    print("   - users.csv")
    print("   - transactions.csv")
    
    # Display sample data
    print("\n" + "=" * 60)
    print("Sample Transactions:")
    print("=" * 60)
    print(transactions_df.head(10))
    
    print("\n" + "=" * 60)
    print("Sample Suspicious Transactions:")
    print("=" * 60)
    print(transactions_df[transactions_df['is_suspicious'] == True].head(10))
    
    print("\n" + "=" * 60)
    print("Transaction Amount Statistics:")
    print("=" * 60)
    print(transactions_df['amount'].describe())
    
    print("\n✅ Data generation complete! You can now use these files for your AML system.")


if __name__ == "__main__":
    main()