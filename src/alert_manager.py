"""
Alert Management System
Stores and manages alerts in database
"""

import sqlite3
import pandas as pd
from datetime import datetime


class AlertManager:
    """
    Manages storage and retrieval of AML alerts in database
    """
    
    def __init__(self, db_path='aml_database.db'):
        """
        Initialize alert manager
        
        Args:
            db_path: Path to SQLite database
        """
        self.db_path = db_path
        self._ensure_tables_exist()
    
    def _ensure_tables_exist(self):
        """Create alerts table if it doesn't exist"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create alerts table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS alerts (
                alert_id INTEGER PRIMARY KEY AUTOINCREMENT,
                transaction_id TEXT NOT NULL,
                user_id TEXT NOT NULL,
                amount REAL NOT NULL,
                transaction_type TEXT,
                timestamp DATETIME NOT NULL,
                rules_triggered TEXT NOT NULL,
                risk_score INTEGER NOT NULL,
                risk_level TEXT NOT NULL,
                status TEXT DEFAULT 'NEW',
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                reviewed_by TEXT,
                reviewed_at DATETIME,
                notes TEXT,
                UNIQUE(transaction_id)
            )
        ''')
        
        # Create index for faster queries
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_alerts_risk_level 
            ON alerts(risk_level)
        ''')
        
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_alerts_user_id 
            ON alerts(user_id)
        ''')
        
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_alerts_status 
            ON alerts(status)
        ''')
        
        conn.commit()
        conn.close()
        
        print(f"✓ Alert tables initialized in {self.db_path}")
    
    def store_alerts(self, scored_df, overwrite=False):
        """
        Store flagged transactions as alerts in database
        
        Args:
            scored_df: DataFrame with scored transactions
            overwrite: If True, replace existing alerts; if False, skip duplicates
            
        Returns:
            Number of alerts stored
        """
        if len(scored_df) == 0:
            print("⚠️ No alerts to store")
            return 0
        
        conn = sqlite3.connect(self.db_path)
        
        # Prepare data for insertion
        alert_data = scored_df[[
            'transaction_id', 'user_id', 'amount', 'transaction_type', 
            'timestamp', 'rule_triggered', 'final_risk_score', 'risk_level'
        ]].copy()
        
        alert_data.columns = [
            'transaction_id', 'user_id', 'amount', 'transaction_type',
            'timestamp', 'rules_triggered', 'risk_score', 'risk_level'
        ]
        
        # Add status column
        alert_data['status'] = 'NEW'
        
        stored_count = 0
        skipped_count = 0
        
        for _, row in alert_data.iterrows():
            try:
                if overwrite:
                    # Delete existing alert first
                    conn.execute(
                        "DELETE FROM alerts WHERE transaction_id = ?",
                        (row['transaction_id'],)
                    )
                
                # Insert new alert
                conn.execute('''
                    INSERT INTO alerts 
                    (transaction_id, user_id, amount, transaction_type, timestamp, 
                     rules_triggered, risk_score, risk_level, status)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    row['transaction_id'],
                    row['user_id'],
                    row['amount'],
                    row['transaction_type'],
                    str(row['timestamp']),
                    row['rules_triggered'],
                    row['risk_score'],
                    row['risk_level'],
                    row['status']
                ))
                
                stored_count += 1
                
            except sqlite3.IntegrityError:
                # Transaction already exists and overwrite=False
                skipped_count += 1
                continue
        
        conn.commit()
        conn.close()
        
        print(f"\n✅ Alert Storage Complete:")
        print(f"   Stored: {stored_count} alerts")
        if skipped_count > 0:
            print(f"   Skipped (duplicates): {skipped_count} alerts")
        
        return stored_count
    
    def get_alerts(self, risk_level=None, status=None, user_id=None, limit=None):
        """
        Query alerts from database
        
        Args:
            risk_level: Filter by risk level (LOW, MEDIUM, HIGH, CRITICAL)
            status: Filter by status (NEW, UNDER_REVIEW, CLOSED, FALSE_POSITIVE)
            user_id: Filter by specific user
            limit: Maximum number of alerts to return
            
        Returns:
            DataFrame with alerts
        """
        conn = sqlite3.connect(self.db_path)
        
        # Build query
        query = "SELECT * FROM alerts WHERE 1=1"
        params = []
        
        if risk_level:
            query += " AND risk_level = ?"
            params.append(risk_level)
        
        if status:
            query += " AND status = ?"
            params.append(status)
        
        if user_id:
            query += " AND user_id = ?"
            params.append(user_id)
        
        query += " ORDER BY risk_score DESC, created_at DESC"
        
        if limit:
            query += f" LIMIT {limit}"
        
        # Execute query
        alerts_df = pd.read_sql_query(query, conn, params=params)
        conn.close()
        
        return alerts_df
    
    def get_alert_statistics(self):
        """
        Get overall alert statistics
        
        Returns:
            Dictionary with statistics
        """
        conn = sqlite3.connect(self.db_path)
        
        # Total alerts
        total_query = "SELECT COUNT(*) as total FROM alerts"
        total = pd.read_sql_query(total_query, conn).iloc[0]['total']
        
        # By risk level
        risk_query = "SELECT risk_level, COUNT(*) as count FROM alerts GROUP BY risk_level"
        risk_dist = pd.read_sql_query(risk_query, conn)
        risk_distribution = dict(zip(risk_dist['risk_level'], risk_dist['count']))
        
        # By status
        status_query = "SELECT status, COUNT(*) as count FROM alerts GROUP BY status"
        status_dist = pd.read_sql_query(status_query, conn)
        status_distribution = dict(zip(status_dist['status'], status_dist['count']))
        
        # Top users
        user_query = """
            SELECT user_id, COUNT(*) as alert_count, SUM(amount) as total_amount
            FROM alerts
            GROUP BY user_id
            ORDER BY alert_count DESC
            LIMIT 10
        """
        top_users = pd.read_sql_query(user_query, conn)
        
        # Recent alerts
        recent_query = """
            SELECT COUNT(*) as count
            FROM alerts
            WHERE created_at >= datetime('now', '-7 days')
        """
        recent_count = pd.read_sql_query(recent_query, conn).iloc[0]['count']
        
        conn.close()
        
        stats = {
            'total_alerts': total,
            'risk_distribution': risk_distribution,
            'status_distribution': status_distribution,
            'top_users': top_users,
            'recent_alerts_7d': recent_count
        }
        
        return stats
    
    def update_alert_status(self, alert_id, new_status, reviewer=None, notes=None):
        """
        Update alert status (for manual review process)
        
        Args:
            alert_id: Alert ID to update
            new_status: New status (UNDER_REVIEW, CLOSED, FALSE_POSITIVE, etc.)
            reviewer: Name of person reviewing
            notes: Review notes
            
        Returns:
            Boolean indicating success
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                UPDATE alerts
                SET status = ?, reviewed_by = ?, reviewed_at = ?, notes = ?
                WHERE alert_id = ?
            ''', (new_status, reviewer, datetime.now(), notes, alert_id))
            
            conn.commit()
            success = cursor.rowcount > 0
            
            if success:
                print(f"✅ Alert {alert_id} updated to {new_status}")
            else:
                print(f"⚠️ Alert {alert_id} not found")
            
        except Exception as e:
            print(f"❌ Error updating alert: {e}")
            success = False
        finally:
            conn.close()
        
        return success
    
    def delete_alert(self, alert_id):
        """
        Delete an alert (use with caution)
        
        Args:
            alert_id: Alert ID to delete
            
        Returns:
            Boolean indicating success
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("DELETE FROM alerts WHERE alert_id = ?", (alert_id,))
        conn.commit()
        
        success = cursor.rowcount > 0
        conn.close()
        
        if success:
            print(f"✅ Alert {alert_id} deleted")
        else:
            print(f"⚠️ Alert {alert_id} not found")
        
        return success
    
    def clear_all_alerts(self, confirm=False):
        """
        Clear all alerts from database (destructive operation)
        
        Args:
            confirm: Must be True to execute
            
        Returns:
            Number of alerts deleted
        """
        if not confirm:
            print("⚠️ Must set confirm=True to clear all alerts")
            return 0
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT COUNT(*) FROM alerts")
        count = cursor.fetchone()[0]
        
        cursor.execute("DELETE FROM alerts")
        conn.commit()
        conn.close()
        
        print(f"✅ Cleared {count} alerts from database")
        return count
    
    def print_alert_summary(self):
        """Print human-readable alert summary"""
        stats = self.get_alert_statistics()
        
        print("\n" + "=" * 70)
        print("ALERT DATABASE SUMMARY")
        print("=" * 70)
        
        print(f"\n📊 Overall Statistics:")
        print(f"   Total Alerts: {stats['total_alerts']:,}")
        print(f"   Recent (7 days): {stats['recent_alerts_7d']:,}")
        
        print(f"\n🚦 Risk Level Distribution:")
        risk_dist = stats['risk_distribution']
        for level in ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW']:
            count = risk_dist.get(level, 0)
            if level == 'CRITICAL':
                icon = "🔴"
            elif level == 'HIGH':
                icon = "🟠"
            elif level == 'MEDIUM':
                icon = "🟡"
            else:
                icon = "🟢"
            print(f"   {icon} {level}: {count:,}")
        
        print(f"\n📋 Status Distribution:")
        status_dist = stats['status_distribution']
        for status, count in status_dist.items():
            print(f"   • {status}: {count:,}")
        
        print(f"\n👥 Top 5 Users by Alert Count:")
        top_users = stats['top_users'].head(5)
        for idx, row in top_users.iterrows():
            print(f"   {idx+1}. {row['user_id']}: {row['alert_count']} alerts (${row['total_amount']:,.2f})")
        
        print("\n" + "=" * 70 + "\n")


def test_alert_manager():
    """Test the alert management system"""
    print("Testing Alert Manager...")
    
    # Create sample scored transactions
    sample_data = pd.DataFrame({
        'transaction_id': ['T1', 'T2', 'T3'],
        'user_id': ['U1', 'U2', 'U3'],
        'amount': [15000, 25000, 9500],
        'transaction_type': ['wire_transfer', 'wire_transfer', 'transfer'],
        'timestamp': pd.date_range('2024-01-01', periods=3, freq='D'),
        'rule_triggered': ['HIGH_VALUE, VELOCITY', 'HIGH_VALUE', 'STRUCTURING'],
        'final_risk_score': [78, 50, 75],
        'risk_level': ['HIGH', 'MEDIUM', 'HIGH']
    })
    
    # Initialize manager (will create test database)
    manager = AlertManager('test_aml.db')
    
    # Store alerts
    manager.store_alerts(sample_data)
    
    # Query alerts
    print("\nQuerying HIGH risk alerts:")
    high_risk = manager.get_alerts(risk_level='HIGH')
    print(high_risk[['alert_id', 'transaction_id', 'amount', 'risk_level']])
    
    # Print summary
    manager.print_alert_summary()


if __name__ == "__main__":
    test_alert_manager()
    
    print("\n" + "=" * 70)
    print("✅ Alert Management Module Ready!")
    print("=" * 70)