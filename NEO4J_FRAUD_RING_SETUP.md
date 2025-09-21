# Neo4j Fraud Ring Detection Setup Guide

## Overview

The new **Neo4j Fraud Ring Detection** tab provides advanced fraud detection capabilities by creating a comprehensive graph database that connects senders and beneficiaries, utilizing consolidated beneficiary groups from intelligent name matching.

## Features

### 🔗 Core Functionality
- **Sender-Beneficiary Connections**: Creates relationships between transaction senders and recipients
- **Consolidated Beneficiaries**: Uses advanced fuzzy matching to group similar beneficiary names
- **Fraud Ring Detection**: Identifies 3 types of suspicious patterns:
  - 🔄 **Circular Flows**: Money flowing in cycles (A→B→C→A)
  - 🎯 **Hub Patterns**: Single beneficiary receiving from many senders
  - ⚡ **Rapid Fire**: High-frequency transactions between same parties

### 📊 Risk Analysis
- **Explainable Risk Scores**: Each node and relationship gets risk scores based on multiple factors
- **Interactive Visualization**: Network graphs showing fraud rings with color-coded risk levels
- **Custom Cypher Queries**: Run advanced Neo4j queries for specialized analysis
- **Export Capabilities**: Download results in JSON format with full analysis details

## Prerequisites

### Option 1: Neo4j Desktop (Recommended)
1. **Download Neo4j Desktop**: Visit https://neo4j.com/download/
2. **Install and Create Database**: 
   - Create a new project
   - Add a local database
   - Set a password (remember this!)
   - Start the database
3. **Connection Details**:
   - URI: `neo4j://localhost:7687`
   - Username: `neo4j`
   - Password: Your chosen password

### Option 2: Docker Setup (Quick Start)
```bash
# Start Neo4j with Docker
docker run -d \
    --name neo4j-fraud \
    -p 7474:7474 -p 7687:7687 \
    -e NEO4J_AUTH=neo4j/password \
    neo4j:latest

# Access Neo4j Browser at http://localhost:7474
# Use username: neo4j, password: password
```

### Option 3: Neo4j AuraDB (Cloud)
1. Create account at https://neo4j.com/cloud/aura/
2. Create a free instance
3. Note the connection URI and credentials

## Installation

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Start Your AML Dashboard**:
   ```bash
   streamlit run app.py
   ```

3. **Navigate to Neo4j Tab**: Click on the "🕸️ Neo4j Fraud Rings" tab

## Usage Guide

### Step 1: Connect to Neo4j
1. Enter your Neo4j connection details (URI, username, password)
2. Click "🔌 Connect to Neo4j"
3. Wait for the success message: "✅ Successfully connected to Neo4j!"

### Step 2: Configure Beneficiary Consolidation
1. **Enable Consolidation**: Check "Use Beneficiary Name Consolidation" (recommended)
2. **Set Similarity Threshold**: Adjust the slider (70% is usually optimal)
   - Higher values = stricter matching
   - Lower values = more name variations grouped together

### Step 3: Build the Fraud Graph
1. Click "🚀 Build Neo4j Fraud Graph"
2. Monitor the progress bar as the system:
   - Creates beneficiary consolidation mapping
   - Builds sender and beneficiary nodes
   - Creates transaction relationships
   - Calculates risk scores
3. Review the **Network Statistics** showing nodes, relationships, and high-risk entities

### Step 4: Detect Fraud Rings
1. **Configure Detection Parameters**:
   - **Minimum Ring Size**: Number of entities in a ring (default: 3)
   - **Maximum Rings**: How many rings to show (default: 10)
2. Click "🔍 Detect Fraud Rings"
3. **Review Results**: Each fraud ring shows:
   - Risk level (🔴 High, 🟡 Medium, 🟢 Low)
   - Pattern type and description
   - Involved parties and amounts
   - Detailed connection information

### Step 5: Analyze and Export
1. **Interactive Visualization**: View network graphs of detected fraud rings
2. **Custom Analysis**: Use the Cypher query tool for advanced investigations
3. **Export Results**: Download comprehensive analysis in JSON format

## Fraud Detection Patterns Explained

### 🔄 Circular Flows
**What it detects**: Money flowing in cycles where funds eventually return to the original sender
**Why it's suspicious**: Classic money laundering pattern used in the "layering" phase
**Example**: Company A → Person B → Company C → Company A

### 🎯 Hub Patterns  
**What it detects**: Single beneficiaries receiving funds from many different senders
**Why it's suspicious**: Typical money mule behavior or collection point for illicit funds
**Risk factors**: Number of senders, total amounts, frequency of transactions

### ⚡ Rapid Fire
**What it detects**: High-frequency transactions between the same sender-beneficiary pair
**Why it's suspicious**: Unusual velocity may indicate automated systems or structured transactions
**Thresholds**: >10 transactions with high risk scores

## Risk Scoring Methodology

### Sender Risk Factors
- **High Transaction Volumes**: Large total amounts sent
- **Multiple Beneficiaries**: Many different recipients (potential distribution)
- **Transaction Frequency**: High number of transactions
- **International Payments**: Cross-border transactions

### Beneficiary Risk Factors
- **Multiple Senders**: Receiving from many different sources
- **Name Consolidation**: Multiple name variations (potential identity obfuscation)
- **Large Amounts**: High total received amounts
- **Transaction Patterns**: Unusual receiving patterns

### Relationship Risk Factors
- **Amount Thresholds**: Transactions over $50,000 get higher risk scores
- **Frequency**: More than 10 transactions between same parties
- **Transfer Types**: International payments increase risk
- **Consolidated Beneficiaries**: Relationships to consolidated entities

## Advanced Features

### Custom Cypher Queries
Run sophisticated Neo4j queries for specialized analysis:

```cypher
# Find high-risk relationships
MATCH (s:Sender)-[r:SENT_TO]->(b:Beneficiary)
WHERE r.risk_score > 0.7
RETURN s.name, b.name, r.total_amount, r.risk_score
ORDER BY r.risk_score DESC
LIMIT 10

# Find beneficiaries with most senders (potential mules)
MATCH (b:Beneficiary)<-[r:SENT_TO]-(s:Sender)
WITH b, count(s) as sender_count, sum(r.total_amount) as total_received
WHERE sender_count > 5
RETURN b.name, sender_count, total_received
ORDER BY sender_count DESC

# Find circular patterns
MATCH path = (s1:Sender)-[:SENT_TO*2..4]->(s1)
RETURN path, length(path)
ORDER BY length(path)
```

### Export and Integration
- **JSON Export**: Complete analysis with metadata for reporting
- **Timestamp Tracking**: All exports include analysis timestamp
- **Parameter Recording**: Analysis parameters saved for reproducibility

## Troubleshooting

### Connection Issues
- **Check Neo4j Status**: Ensure Neo4j database is running
- **Verify Credentials**: Double-check username and password
- **Port Conflicts**: Ensure ports 7474 and 7687 are available
- **Firewall**: Check firewall settings for Neo4j ports

### Performance Optimization
- **Large Datasets**: Consider using smaller samples for initial testing
- **Memory Settings**: Increase Neo4j memory settings for large graphs
- **Indexes**: The system automatically creates indexes for better performance

### Common Errors
- **"Failed to connect"**: Check Neo4j server status and credentials
- **"No fraud rings detected"**: Try lowering detection thresholds or minimum ring size
- **Memory errors**: Reduce dataset size or increase system memory

## Integration with Existing Features

### Beneficiary Analysis Integration
The Neo4j tab seamlessly integrates with the existing **Beneficiary Analysis** tab by:
- Using the same advanced fuzzy matching algorithms
- Applying name consolidation rules consistently
- Leveraging similarity thresholds and pattern detection

### AML Dashboard Synergy
- **Anomaly Detection**: Use flagged customers from the AML Dashboard for targeted Neo4j analysis
- **Customer Investigation**: Cross-reference Neo4j findings with customer-level summaries
- **Risk Validation**: Validate ML-detected anomalies through graph pattern analysis

## Best Practices

### Analysis Workflow
1. **Start with EDA**: Upload data and understand transaction patterns
2. **Run AML Detection**: Identify high-risk customers using machine learning
3. **Build Neo4j Graph**: Create comprehensive relationship mapping
4. **Detect Fraud Rings**: Find network-based suspicious patterns
5. **Validate Findings**: Use LLM Investigator for detailed analysis
6. **Export Results**: Create comprehensive reports for stakeholders

### Parameter Tuning
- **Similarity Threshold**: Start with 70%, adjust based on data quality
- **Ring Size**: Begin with 3, increase for complex organizations
- **Risk Thresholds**: Adjust based on regulatory requirements and risk appetite

### Data Quality
- **Name Standardization**: Better beneficiary names improve consolidation
- **Complete Timestamps**: Ensure transaction dates are accurate
- **Amount Validation**: Verify transaction amounts are numeric

## Support and Further Development

This implementation provides a solid foundation for Neo4j-powered fraud detection. Future enhancements could include:
- **Time-series Analysis**: Temporal fraud pattern detection
- **Community Detection**: Advanced graph clustering algorithms
- **Machine Learning Integration**: Graph neural networks for enhanced detection
- **Real-time Monitoring**: Streaming fraud detection capabilities

For technical support or feature requests, refer to the main AML Dashboard documentation or create issues in the project repository.

