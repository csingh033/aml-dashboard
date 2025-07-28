# 🎯 AML Dashboard Customer Filtering Improvements

## Problem Solved

**Issue**: The customer-level AML summary was showing **all customers** with any flagged transactions instead of focusing on the **highest-risk customers** based on Isolation Forest, Z-score, and percentile criteria.

**Root Cause**: The original implementation displayed every customer who had at least one transaction flagged by the Isolation Forest algorithm, without applying additional risk-based filtering thresholds.

## ✅ Solution Implemented

### 1. **Smart Customer Filtering Controls**
Added interactive filtering controls that allow you to focus on truly high-risk customers:

- **Minimum Flagged Transactions**: Show only customers with multiple suspicious transactions (default: 2+)
- **Z-Score Threshold**: Filter by statistical significance of transaction amounts (default: 1.5+) 
- **Percentile Threshold**: Show customers in the highest risk percentiles (default: 80%+)

### 2. **Enhanced Risk Scoring**
- **Composite Risk Score**: Combines Z-score and percentile into a single risk ranking
- **Automatic Sorting**: Customers are sorted by highest risk first
- **Risk Score Column**: New column showing calculated risk level for easy identification

### 3. **Improved Isolation Forest Configuration**
- **Conservative Default**: Reduced contamination rate from "auto" to 5% (was flagging too many transactions)
- **Configurable Contamination**: Sidebar control to adjust detection sensitivity (1-20%)
- **Real-time Statistics**: Shows how many transactions are flagged by the algorithm

### 4. **Better User Experience**
- **Clear Status Messages**: Shows exactly how many high-risk vs. total flagged customers
- **Summary Statistics**: When no customers meet criteria, shows distribution to help adjust filters
- **CSV Download**: Export filtered high-risk customer list
- **Help Text**: Tooltips explain what each filter does

## 🚀 How to Use the Improved System

### Step 1: Upload Your Data
1. Go to the **📊 EDA** tab
2. Upload your transaction CSV file
3. Review the data quality and patterns

### Step 2: Configure Detection Settings
1. In the **💼 AML Dashboard** tab, use the sidebar controls:
   - **Contamination Rate**: Start with 5% for conservative detection
   - Adjust based on your data characteristics

### Step 3: Apply Customer Filters
1. Set **Minimum Flagged Transactions** (recommended: 2-3)
2. Set **Z-Score Threshold** (recommended: 1.5-2.0)
3. Set **Percentile Threshold** (recommended: 80-90%)
4. Review the filtered results

### Step 4: Analyze High-Risk Customers
1. Review customers sorted by **risk_score** (highest first)
2. Examine **reasons** for flagging
3. Check **transfer_types** and **beneficiaries**
4. Use the **story** column for quick summaries

## 📊 Expected Results

### Before (Issue):
```
Showing: 250 customers with any flagged transactions
Problem: Too many customers, difficult to prioritize
```

### After (Fixed):
```
✅ Found 12 high-risk customers out of 250 total flagged customers
Result: Focused list of truly suspicious customers
```

## 🎛️ Recommended Settings by Use Case

### **Conservative Detection** (Low False Positives)
- Contamination Rate: 2-3%
- Min Flagged Transactions: 3+
- Z-Score Threshold: 2.0+
- Percentile Threshold: 90%+

### **Balanced Detection** (Default)
- Contamination Rate: 5%
- Min Flagged Transactions: 2+
- Z-Score Threshold: 1.5+
- Percentile Threshold: 80%+

### **Sensitive Detection** (Catch More Cases)
- Contamination Rate: 8-10%
- Min Flagged Transactions: 1+
- Z-Score Threshold: 1.0+
- Percentile Threshold: 70%+

## 🔍 Algorithm Details

### Isolation Forest
- **Purpose**: Detects transactions that are statistical outliers
- **Features Used**: Amount, timing, transfer type, beneficiary patterns, frequency
- **Contamination Rate**: Now configurable (default 5% vs. previous "auto")

### Z-Score Analysis
- **Purpose**: Measures how many standard deviations a transaction is from the mean
- **Threshold**: Values > 1.5 indicate statistical significance
- **Calculation**: Based on standardized transaction amounts

### Percentile Analysis
- **Purpose**: Shows where a customer ranks relative to all others
- **Threshold**: 80%+ means customer is in the top 20% of risk
- **Calculation**: Based on transaction amounts and patterns

## 🎯 Key Benefits

1. **Focused Results**: Only see customers who meet multiple risk criteria
2. **Adjustable Sensitivity**: Configure detection based on your risk tolerance  
3. **Clear Prioritization**: Customers sorted by composite risk score
4. **Actionable Insights**: Each customer has detailed reason codes
5. **Exportable Data**: Download filtered results for further analysis

## 🛠️ Technical Improvements

- **Performance**: Faster processing with smarter filtering
- **Accuracy**: Reduced false positives through multi-criteria filtering  
- **Usability**: Interactive controls for real-time filtering adjustment
- **Transparency**: Clear metrics showing detection effectiveness
- **Scalability**: Handles large datasets with efficient pandas operations

Your AML Dashboard now provides a **focused, risk-based view** of customers who warrant investigation, rather than an overwhelming list of all flagged transactions. 