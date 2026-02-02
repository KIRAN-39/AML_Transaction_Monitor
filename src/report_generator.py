"""
Report Generator for AML System
Generates PDF and Excel reports for compliance and analysis
"""

import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO


class ReportGenerator:
    """
    Generates various reports for AML compliance and analysis
    """
    
    def __init__(self):
        self.report_date = datetime.now()
    
    def generate_summary_report(self, alerts_df, transactions_df):
        """
        Generate summary statistics report
        
        Args:
            alerts_df: DataFrame with flagged transactions
            transactions_df: DataFrame with all transactions
            
        Returns:
            Dictionary with summary statistics
        """
        summary = {
            'report_date': self.report_date.strftime('%Y-%m-%d %H:%M:%S'),
            'total_transactions': len(transactions_df),
            'flagged_transactions': len(alerts_df),
            'flag_rate': len(alerts_df) / len(transactions_df) * 100,
            'total_volume': transactions_df['amount'].sum(),
            'flagged_volume': alerts_df['amount'].sum(),
            'unique_users': transactions_df['user_id'].nunique(),
            'flagged_users': alerts_df['user_id'].nunique(),
        }
        
        # Risk level breakdown
        if 'risk_level' in alerts_df.columns:
            risk_dist = alerts_df['risk_level'].value_counts().to_dict()
            summary['risk_distribution'] = risk_dist
        
        # Average scores
        if 'final_risk_score' in alerts_df.columns:
            summary['avg_risk_score'] = alerts_df['final_risk_score'].mean()
            summary['max_risk_score'] = alerts_df['final_risk_score'].max()
        
        return summary
    
    def export_to_excel(self, alerts_df, transactions_df, filename='aml_report.xlsx'):
        """
        Export data to Excel with multiple sheets
        
        Args:
            alerts_df: DataFrame with alerts
            transactions_df: DataFrame with all transactions
            filename: Output filename
            
        Returns:
            Path to saved file
        """
        print(f"Generating Excel report: {filename}")
        
        with pd.ExcelWriter(filename, engine='openpyxl') as writer:
            # Summary sheet
            summary = self.generate_summary_report(alerts_df, transactions_df)
            summary_df = pd.DataFrame([summary])
            summary_df.to_excel(writer, sheet_name='Summary', index=False)
            
            # All alerts
            alerts_df.to_excel(writer, sheet_name='All Alerts', index=False)
            
            # High-risk alerts only
            if 'risk_level' in alerts_df.columns:
                high_risk = alerts_df[alerts_df['risk_level'].isin(['CRITICAL', 'HIGH'])]
                high_risk.to_excel(writer, sheet_name='High Risk Alerts', index=False)
            
            # Top risky users
            top_users = alerts_df.groupby('user_id').agg({
                'transaction_id': 'count',
                'final_risk_score': ['mean', 'max'],
                'amount': 'sum'
            }).reset_index()
            
            top_users.columns = ['user_id', 'alert_count', 'avg_risk_score', 
                                'max_risk_score', 'total_amount']
            top_users = top_users.sort_values('alert_count', ascending=False)
            top_users.to_excel(writer, sheet_name='Top Risky Users', index=False)
            
            # Rule effectiveness
            if 'rule_triggered' in alerts_df.columns:
                rule_counts = {}
                for rules in alerts_df['rule_triggered']:
                    for rule in str(rules).split(', '):
                        rule_counts[rule] = rule_counts.get(rule, 0) + 1
                
                rule_df = pd.DataFrame(list(rule_counts.items()), 
                                      columns=['Rule', 'Count'])
                rule_df = rule_df.sort_values('Count', ascending=False)
                rule_df.to_excel(writer, sheet_name='Rule Effectiveness', index=False)
        
        print(f"✅ Excel report saved: {filename}")
        return filename
    
    def create_alert_csv(self, alerts_df, filename='alerts_export.csv'):
        """
        Export alerts to CSV for external systems
        
        Args:
            alerts_df: DataFrame with alerts
            filename: Output filename
            
        Returns:
            Path to saved file
        """
        print(f"Exporting alerts to CSV: {filename}")
        
        # Select key columns
        export_cols = ['transaction_id', 'user_id', 'amount', 'timestamp', 
                      'transaction_type', 'rule_triggered', 'final_risk_score', 
                      'risk_level']
        
        export_cols = [col for col in export_cols if col in alerts_df.columns]
        
        alerts_df[export_cols].to_csv(filename, index=False)
        
        print(f"✅ CSV export saved: {filename}")
        return filename
    
    def generate_compliance_report(self, alerts_df, transactions_df):
        """
        Generate compliance-ready text report
        
        Args:
            alerts_df: DataFrame with alerts
            transactions_df: DataFrame with all transactions
            
        Returns:
            String with formatted report
        """
        report_lines = []
        
        # Header
        report_lines.append("=" * 80)
        report_lines.append("AML TRANSACTION MONITORING REPORT")
        report_lines.append("=" * 80)
        report_lines.append(f"\nReport Generated: {self.report_date.strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")
        
        # Executive Summary
        report_lines.append("EXECUTIVE SUMMARY")
        report_lines.append("-" * 80)
        
        summary = self.generate_summary_report(alerts_df, transactions_df)
        
        report_lines.append(f"Total Transactions Analyzed: {summary['total_transactions']:,}")
        report_lines.append(f"Flagged Transactions: {summary['flagged_transactions']:,} ({summary['flag_rate']:.2f}%)")
        report_lines.append(f"Total Transaction Volume: ${summary['total_volume']:,.2f}")
        report_lines.append(f"Flagged Transaction Volume: ${summary['flagged_volume']:,.2f}")
        report_lines.append(f"Unique Users Monitored: {summary['unique_users']:,}")
        report_lines.append(f"Users with Alerts: {summary['flagged_users']:,}")
        report_lines.append("")
        
        # Risk Distribution
        report_lines.append("RISK LEVEL DISTRIBUTION")
        report_lines.append("-" * 80)
        
        if 'risk_distribution' in summary:
            for level in ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW']:
                count = summary['risk_distribution'].get(level, 0)
                percentage = count / summary['flagged_transactions'] * 100 if summary['flagged_transactions'] > 0 else 0
                report_lines.append(f"{level:12s}: {count:6,} ({percentage:5.1f}%)")
        
        report_lines.append("")
        
        # Top 10 Highest Risk Transactions
        report_lines.append("TOP 10 HIGHEST RISK TRANSACTIONS")
        report_lines.append("-" * 80)
        
        top_10 = alerts_df.nlargest(10, 'final_risk_score')
        
        for idx, row in top_10.iterrows():
            report_lines.append(f"\n{idx+1}. Transaction ID: {row['transaction_id']}")
            report_lines.append(f"   User: {row['user_id']}")
            report_lines.append(f"   Amount: ${row['amount']:,.2f}")
            report_lines.append(f"   Risk Score: {row['final_risk_score']}")
            report_lines.append(f"   Risk Level: {row['risk_level']}")
            report_lines.append(f"   Rules Triggered: {row['rule_triggered']}")
        
        report_lines.append("")
        report_lines.append("=" * 80)
        report_lines.append("END OF REPORT")
        report_lines.append("=" * 80)
        
        return "\n".join(report_lines)
    
    def save_compliance_report(self, alerts_df, transactions_df, filename='compliance_report.txt'):
        """
        Save compliance report to text file
        
        Args:
            alerts_df: DataFrame with alerts
            transactions_df: DataFrame with all transactions
            filename: Output filename
            
        Returns:
            Path to saved file
        """
        print(f"Generating compliance report: {filename}")
        
        report = self.generate_compliance_report(alerts_df, transactions_df)
        
        with open(filename, 'w') as f:
            f.write(report)
        
        print(f"✅ Compliance report saved: {filename}")
        return filename
    
    def create_visualization_report(self, alerts_df, filename='alert_charts.png'):
        """
        Create comprehensive visualization report
        
        Args:
            alerts_df: DataFrame with alerts
            filename: Output filename
            
        Returns:
            Path to saved file
        """
        print(f"Creating visualization report: {filename}")
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        plt.style.use('seaborn-v0_8-darkgrid')
        
        # 1. Risk Level Distribution
        risk_counts = alerts_df['risk_level'].value_counts()
        colors = {'CRITICAL': '#f44336', 'HIGH': '#ff9800', 
                 'MEDIUM': '#ffc107', 'LOW': '#4caf50'}
        
        risk_colors = [colors.get(level, '#999999') for level in risk_counts.index]
        
        axes[0, 0].bar(risk_counts.index, risk_counts.values, color=risk_colors, alpha=0.7)
        axes[0, 0].set_title('Alert Distribution by Risk Level', fontsize=14, fontweight='bold')
        axes[0, 0].set_ylabel('Number of Alerts')
        axes[0, 0].grid(axis='y', alpha=0.3)
        
        # Add value labels
        for i, (level, count) in enumerate(risk_counts.items()):
            axes[0, 0].text(i, count + max(risk_counts.values)*0.02, str(count), 
                          ha='center', fontweight='bold')
        
        # 2. Risk Score Distribution
        axes[0, 1].hist(alerts_df['final_risk_score'], bins=30, 
                       color='steelblue', alpha=0.7, edgecolor='black')
        axes[0, 1].set_title('Risk Score Distribution', fontsize=14, fontweight='bold')
        axes[0, 1].set_xlabel('Risk Score')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].axvline(alerts_df['final_risk_score'].mean(), 
                          color='red', linestyle='--', label=f'Mean: {alerts_df["final_risk_score"].mean():.1f}')
        axes[0, 1].legend()
        axes[0, 1].grid(axis='y', alpha=0.3)
        
        # 3. Amount by Risk Level (Box Plot)
        risk_order = ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW']
        risk_data = [alerts_df[alerts_df['risk_level'] == level]['amount'].values 
                    for level in risk_order if level in alerts_df['risk_level'].unique()]
        risk_labels = [level for level in risk_order 
                      if level in alerts_df['risk_level'].unique()]
        
        bp = axes[1, 0].boxplot(risk_data, labels=risk_labels, patch_artist=True)
        
        # Color the boxes
        box_colors = [colors.get(level, '#999999') for level in risk_labels]
        for patch, color in zip(bp['boxes'], box_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        
        axes[1, 0].set_title('Transaction Amount by Risk Level', fontsize=14, fontweight='bold')
        axes[1, 0].set_ylabel('Amount ($)')
        axes[1, 0].grid(axis='y', alpha=0.3)
        
        # 4. Top Rules Triggered
        rule_counts = {}
        for rules in alerts_df['rule_triggered']:
            for rule in str(rules).split(', '):
                rule_counts[rule] = rule_counts.get(rule, 0) + 1
        
        rule_df = pd.DataFrame(list(rule_counts.items()), 
                              columns=['Rule', 'Count']).sort_values('Count', ascending=True)
        
        axes[1, 1].barh(rule_df['Rule'], rule_df['Count'], color='coral', alpha=0.7)
        axes[1, 1].set_title('Detection Rules Triggered', fontsize=14, fontweight='bold')
        axes[1, 1].set_xlabel('Number of Alerts')
        axes[1, 1].grid(axis='x', alpha=0.3)
        
        # Add value labels
        for i, (rule, count) in enumerate(zip(rule_df['Rule'], rule_df['Count'])):
            axes[1, 1].text(count + max(rule_df['Count'])*0.02, i, str(count), 
                          va='center', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Visualization report saved: {filename}")
        return filename
    
    def generate_all_reports(self, alerts_df, transactions_df, output_dir='reports/'):
        """
        Generate all types of reports
        
        Args:
            alerts_df: DataFrame with alerts
            transactions_df: DataFrame with all transactions
            output_dir: Directory to save reports
            
        Returns:
            Dictionary with paths to all generated reports
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        print("\n" + "=" * 70)
        print("GENERATING COMPREHENSIVE REPORT SUITE")
        print("=" * 70 + "\n")
        
        reports = {}
        
        # Excel report
        excel_file = f'{output_dir}aml_report_{timestamp}.xlsx'
        reports['excel'] = self.export_to_excel(alerts_df, transactions_df, excel_file)
        
        # CSV export
        csv_file = f'{output_dir}alerts_{timestamp}.csv'
        reports['csv'] = self.create_alert_csv(alerts_df, csv_file)
        
        # Compliance report
        txt_file = f'{output_dir}compliance_report_{timestamp}.txt'
        reports['compliance'] = self.save_compliance_report(alerts_df, transactions_df, txt_file)
        
        # Visualizations
        chart_file = f'{output_dir}charts_{timestamp}.png'
        reports['charts'] = self.create_visualization_report(alerts_df, chart_file)
        
        print("\n" + "=" * 70)
        print("REPORT GENERATION COMPLETE")
        print("=" * 70)
        print(f"\nGenerated {len(reports)} report files:")
        for report_type, filepath in reports.items():
            print(f"   {report_type}: {filepath}")
        
        return reports


def test_report_generator():
    """Test the report generator with sample data"""
    print("Testing Report Generator...")
    
    # Create sample data
    sample_alerts = pd.DataFrame({
        'transaction_id': [f'T{i}' for i in range(1, 51)],
        'user_id': [f'U{i%20}' for i in range(1, 51)],
        'amount': np.random.uniform(1000, 50000, 50),
        'timestamp': pd.date_range('2024-01-01', periods=50, freq='D'),
        'transaction_type': np.random.choice(['wire_transfer', 'purchase', 'transfer'], 50),
        'rule_triggered': np.random.choice(['HIGH_VALUE', 'VELOCITY', 'STRUCTURING', 
                                           'HIGH_VALUE, VELOCITY'], 50),
        'final_risk_score': np.random.randint(30, 120, 50),
        'risk_level': np.random.choice(['CRITICAL', 'HIGH', 'MEDIUM', 'LOW'], 50)
    })
    
    sample_transactions = pd.DataFrame({
        'transaction_id': [f'T{i}' for i in range(1, 1001)],
        'user_id': [f'U{i%100}' for i in range(1, 1001)],
        'amount': np.random.uniform(100, 10000, 1000),
        'timestamp': pd.date_range('2024-01-01', periods=1000, freq='H')
    })
    
    # Generate reports
    generator = ReportGenerator()
    reports = generator.generate_all_reports(sample_alerts, sample_transactions, 'test_reports/')
    
    print("\n✅ Report generator test complete!")


if __name__ == "__main__":
    test_report_generator()
    
    print("\n" + "=" * 70)
    print("✅ Report Generator Module Ready!")
    print("=" * 70)
