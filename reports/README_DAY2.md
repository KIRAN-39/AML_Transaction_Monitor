# Day 2: Detection Rules & Scoring System - Setup Guide

## What You Got Today 🎉

4 new Python modules that work together:

1. **detection_rules.py** - 5 AML detection rules
2. **risk_scorer.py** - Risk scoring and assessment system
3. **alert_manager.py** - Database storage and alert management
4. **test_detection_system.py** - End-to-end testing script

---

## Quick Start

### Prerequisites
Make sure you have:
- ✅ Generated transaction data (from Day 1)
- ✅ Python environment with pandas, numpy installed

### Step 1: Organize Your Files

Your project folder should look like this:
```
AML_Project/
├── data_generator.py          (Day 1)
├── transactions.csv           (Generated Day 1)
├── users.csv                  (Generated Day 1)
├── detection_rules.py         (NEW - Day 2)
├── risk_scorer.py             (NEW - Day 2)
├── alert_manager.py           (NEW - Day 2)
└── test_detection_system.py   (NEW - Day 2)
```

### Step 2: Run the Complete Detection System

```bash
python test_detection_system.py
```

This will:
1. ✅ Load your transaction data
2. ✅ Run all 5 detection rules
3. ✅ Calculate risk scores
4. ✅ Store alerts in database (creates `aml_database.db`)
5. ✅ Generate comprehensive reports

**Expected Runtime**: 30 seconds - 2 minutes (depending on data size)

---

## What Each Module Does

### 1. detection_rules.py

**5 Detection Rules Implemented:**

| Rule | What It Detects | Risk Score |
|------|----------------|------------|
| **High-Value** | Transactions > $10,000 | 30-70 |
| **Velocity** | 10+ transactions in 24 hours | 30-50 |
| **Structuring** | Multiple $9,000-$9,999 transactions | 40-70 |
| **Dormant Account** | Inactive account suddenly active | 35-50 |
| **Round Numbers** | 3+ exact round amount transactions | 30-45 |

**Usage Example:**
```python
from detection_rules import AMLDetectionRules

detector = AMLDetectionRules()
results = detector.run_all_rules(transactions_df)

# Access specific rule results
high_value_flags = results['high_value']
velocity_flags = results['velocity']
```

### 2. risk_scorer.py

**Combines multiple rule detections into single risk score**

**Risk Levels:**
- 🟢 **LOW**: 0-30 points
- 🟡 **MEDIUM**: 31-60 points
- 🟠 **HIGH**: 61-90 points
- 🔴 **CRITICAL**: 91+ points

**Features:**
- Weighted scoring (structuring gets 1.5x weight)
- Cumulative scores for transactions flagged by multiple rules
- User risk profiling
- Performance reporting

**Usage Example:**
```python
from risk_scorer import RiskScorer

scorer = RiskScorer()
combined = scorer.combine_rule_results(rule_results)
scored_df = scorer.calculate_cumulative_scores(combined)

# Get top 10 riskiest
top_10 = scorer.get_top_risky_transactions(scored_df, n=10)
```

### 3. alert_manager.py

**Database storage and query system**

**Features:**
- SQLite database with indexed tables
- Store/retrieve alerts
- Update alert status (NEW, UNDER_REVIEW, CLOSED, FALSE_POSITIVE)
- Query by risk level, status, user
- Alert statistics and reporting

**Usage Example:**
```python
from alert_manager import AlertManager

manager = AlertManager('aml_database.db')

# Store alerts
manager.store_alerts(scored_df)

# Query high-risk alerts
high_risk = manager.get_alerts(risk_level='HIGH')

# Update alert status
manager.update_alert_status(
    alert_id=1, 
    new_status='UNDER_REVIEW',
    reviewer='John Doe'
)
```

### 4. test_detection_system.py

**Complete end-to-end pipeline**

Runs everything automatically and generates:
- Detection summary
- Risk distribution
- Top 10 riskiest transactions
- High-risk users list
- Rule effectiveness analysis
- Performance metrics (if ground truth available)

---

## Understanding the Output

### When You Run test_detection_system.py:

**1. Detection Summary**
```
HIGH_VALUE: 127 transactions flagged
VELOCITY: 45 transactions flagged
STRUCTURING: 89 transactions flagged
...
```

**2. Risk Assessment**
```
🚦 Risk Level Distribution:
   🔴 CRITICAL: 12 (2.3%)
   🟠 HIGH: 45 (8.7%)
   🟡 MEDIUM: 98 (19.0%)
   🟢 LOW: 67 (13.0%)
```

**3. Top Risky Transactions**
```
1. Transaction: T12345
   User: USER_000456
   Amount: $25,000.00
   Rules: HIGH_VALUE, STRUCTURING
   Risk Score: 120 (CRITICAL)
```

**4. High-Risk Users**
```
user_id          | alerts | avg_score | max_score | total_amount
USER_000456      | 8      | 85.3      | 120       | $187,500.00
USER_000789      | 6      | 72.1      | 95        | $145,000.00
```

---

## Testing Individual Modules

### Test Detection Rules Only:
```bash
python detection_rules.py
```

### Test Risk Scorer Only:
```bash
python risk_scorer.py
```

### Test Alert Manager Only:
```bash
python alert_manager.py
```

Each module has built-in test functions that run when executed directly.

---

## Customizing Rules

### Adjust Thresholds:

**In detection_rules.py:**

```python
# Change high-value threshold
detector.detect_high_value_transactions(df, threshold=15000)

# Change velocity threshold
detector.detect_velocity_anomalies(df, transaction_threshold=15)

# Change structuring amounts
detector.detect_structuring(df, min_amount=8500, max_amount=9999)
```

### Adjust Risk Scoring:

**In risk_scorer.py:**

```python
# Modify rule weights
scorer.rule_weights = {
    'HIGH_VALUE': 1.0,
    'VELOCITY': 1.5,        # Increase velocity importance
    'STRUCTURING': 2.0,     # Highest priority
    'DORMANT_REACTIVATION': 1.0,
    'ROUND_NUMBERS': 0.5    # Lower priority
}

# Modify risk level ranges
scorer.risk_levels = {
    'LOW': (0, 40),         # Wider low range
    'MEDIUM': (41, 70),
    'HIGH': (71, 100),
    'CRITICAL': (101, float('inf'))
}
```

---

## Database Schema

**alerts table:**
```sql
alert_id          INTEGER PRIMARY KEY
transaction_id    TEXT (unique)
user_id           TEXT
amount            REAL
transaction_type  TEXT
timestamp         DATETIME
rules_triggered   TEXT (comma-separated)
risk_score        INTEGER
risk_level        TEXT (LOW/MEDIUM/HIGH/CRITICAL)
status            TEXT (NEW/UNDER_REVIEW/CLOSED/FALSE_POSITIVE)
created_at        DATETIME
reviewed_by       TEXT
reviewed_at       DATETIME
notes             TEXT
```

**Query Examples:**
```python
# Get all critical alerts
critical = manager.get_alerts(risk_level='CRITICAL')

# Get new alerts only
new_alerts = manager.get_alerts(status='NEW')

# Get alerts for specific user
user_alerts = manager.get_alerts(user_id='USER_000123')
```

---

## Troubleshooting

### "transactions.csv not found"
**Solution:** Run `python data_generator.py` first

### "No module named 'detection_rules'"
**Solution:** Make sure all .py files are in the same folder

### Very few alerts generated
**Solution:** Your data might not have suspicious patterns. Check if you used the data_generator.py from Day 1 that injects patterns.

### Database locked error
**Solution:** Close any programs that might have the database open

---

## Performance Metrics

**What to expect with 50,000 transactions:**

- Detection time: 10-30 seconds
- Typical flag rate: 3-8% of transactions
- Alert count: 1,500-4,000 alerts
- High-risk alerts: 100-500
- Critical alerts: 10-100

**If your numbers are very different:**
- Too many alerts? Increase thresholds
- Too few alerts? Decrease thresholds or check data quality

---

## Next Steps (Day 3 Preview)

Tomorrow you'll add:
1. ✅ Feature engineering for ML
2. ✅ Isolation Forest model
3. ✅ K-Means clustering
4. ✅ Model comparison

---

## Success Checklist ✅

By end of Day 2, you should have:

- [ ] All 4 new Python files in your project folder
- [ ] Successfully run `test_detection_system.py`
- [ ] Generated `aml_database.db` with alerts
- [ ] Seen detection summary with flagged transactions
- [ ] Understood the output reports
- [ ] Verified alerts are stored in database

---

## Getting Help

**Common Questions:**

**Q: How do I know if my rules are working well?**
A: Run the test script. If you see flags in each rule category and the performance metrics look reasonable, you're good!

**Q: Should I tune the thresholds?**
A: Not yet! First see what the default settings give you. You can tune in Week 3.

**Q: Can I use this with real data?**
A: Yes! Just replace transactions.csv with your data (must have same columns).

---

**🎉 Congratulations! You now have a working AML detection system!**

**Day 2 Complete**: ✅ Detection Rules | ✅ Risk Scoring | ✅ Alert Storage

**Tomorrow**: Machine Learning Models 🤖
