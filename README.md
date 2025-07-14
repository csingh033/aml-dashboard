# 🚀 AML Dashboard - Anti-Money Laundering Analysis Tool

A comprehensive Anti-Money Laundering (AML) dashboard built with Streamlit, featuring anomaly detection, network analysis, and AI-powered investigation capabilities.

## 🌟 Features

### 🔍 **Anomaly Detection**
- **Isolation Forest Algorithm**: Detects unusual transactions
- **Explainable AI**: Provides human-readable reasons for flagged transactions
- **Multi-dimensional Analysis**: Considers amount, time, location, and patterns

### 🌐 **Network Analysis**
- **Transaction Networks**: Visualize money flow between customers
- **Cycle Detection**: Identify potential laundering rings
- **Interactive Graphs**: Explore customer connections with compact visualization

### 🤖 **AI-Powered Investigation**
- **LLM Investigator**: GPT-4o powered analysis with color-coded risk assessment
- **Risk Scoring**: RED (HIGH), GREEN (LOW), BLUE (MEDIUM) risk levels
- **Actionable Insights**: 4-key bullet point recommendations

### 📊 **Exploratory Data Analysis**
- **Daily Transaction Trends**: Three smooth line charts
- **Pattern Recognition**: Temporal and volume analysis
- **Data Quality Assessment**: Comprehensive data validation

### 🔐 **Privacy & Security**
- **Data Hashing**: SHA-256 encryption of customer identifiers
- **Local Deployment**: All data stays on your machine
- **Audit Trail**: Complete transaction history preservation

## 🚀 Quick Start

### Option 1: Docker (Recommended)

```bash
# Clone the repository
git clone https://github.com/csingh033/aml-dashboard.git
cd aml-dashboard

# Start the dashboard
./run-docker.sh

# Access the dashboard
# Open http://localhost:8501 in your browser
```

### Option 2: Local Development

```bash
# Clone the repository
git clone https://github.com/csingh033/aml-dashboard.git
cd aml-dashboard

# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run app.py
```

## 📋 Prerequisites

- **Python 3.8+** or **Docker Desktop**
- **4GB RAM** minimum, 8GB recommended
- **Modern web browser** (Chrome, Firefox, Safari, Edge)

## 🏗️ Architecture

### **Core Components**

1. **📊 EDA Tab**: Exploratory data analysis with interactive charts
2. **💼 AML Dashboard**: Main anomaly detection and analysis
3. **🤖 LLM Investigator**: AI-powered transaction analysis
4. **🔍 Customer ID Lookup**: Reverse hashing for investigation
5. **ℹ️ Model Information**: Methodology and technical details

### **Technical Stack**

- **Frontend**: Streamlit
- **Backend**: Python 3.9+
- **ML**: Scikit-learn (Isolation Forest)
- **Graph Analysis**: NetworkX
- **Visualization**: Plotly, Matplotlib
- **AI**: OpenAI GPT-4o
- **Containerization**: Docker

## 📁 Project Structure

```
aml-dashboard/
├── app.py                          # Main Streamlit application
├── requirements.txt                 # Python dependencies
├── run-docker.sh                   # Start Docker container
├── stop-docker.sh                  # Stop Docker container
├── README.md                       # This file
├── CONTRIBUTING.md                 # Contribution guidelines
├── LICENSE                         # MIT license
├── .gitignore                      # Git ignore rules
├── .github/workflows/              # CI/CD pipelines
└── docker/                         # Docker configuration
    ├── Dockerfile                  # Container configuration
    ├── docker-compose.yml          # Docker orchestration
    ├── run-aml-dashboard.sh        # Docker start script
    ├── stop-aml-dashboard.sh       # Docker stop script
    ├── package-for-stakeholders.sh # Distribution script
    └── README-STAKESHOLDER.md     # Stakeholder guide
```

## 🔧 Configuration

### **Environment Variables**

Create a `.env` file for sensitive configuration:

```bash
# OpenAI API Key (optional, for LLM features)
OPENAI_API_KEY=your_openai_api_key_here
```

### **Data Format**

The dashboard expects CSV files with the following columns:

```csv
customer_no,CustomerName,transfer_type,amount,beneficiary_name,createdDateTime,reference_no
CUST001,John Doe,INTERNATIONAL_PAYMENT,5000.00,Beneficiary A,2024-01-15 10:30:00,REF001
CUST002,Jane Smith,TOP-UP,1000.00,,2024-01-15 11:15:00,REF002
```

## 🛠️ Development

### **Local Development Setup**

```bash
# Clone the repository
git clone https://github.com/csingh033/aml-dashboard.git
cd aml-dashboard

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run app.py
```

### **Docker Development**

```bash
# Run with Docker
./run-docker.sh

# Stop Docker container
./stop-docker.sh

# Or use docker-compose directly
cd docker
docker-compose up
```

## 📦 Deployment

### **Stakeholder Package**

Create a distribution package for stakeholders:

```bash
# Create stakeholder package
cd docker
./package-for-stakeholders.sh

# This creates: aml-dashboard-stakeholder-package.tar.gz
```

### **Docker Hub**

Push to Docker Hub for easy distribution:

```bash
# Build and tag the image
cd docker
docker build -t csingh033/aml-dashboard:latest -f Dockerfile ..

# Push to Docker Hub
docker push csingh033/aml-dashboard:latest
```

## 🔒 Security Features

- **Data Hashing**: All customer IDs are SHA-256 hashed
- **Local Processing**: No data leaves your machine
- **Privacy First**: No external data transmission
- **Audit Trail**: Complete transaction logging

## 📊 Usage Examples

### **1. Upload Transaction Data**
1. Navigate to the **📊 EDA** tab
2. Upload your CSV transaction file
3. View exploratory data analysis charts

### **2. Run Anomaly Detection**
1. Go to the **💼 AML Dashboard** tab
2. The system automatically detects suspicious transactions
3. Review customer-level summaries and explanations

### **3. Network Analysis**
1. In the AML Dashboard, enter a hashed customer ID
2. View their transaction network and cycles
3. Analyze connection patterns and suspicious flows

### **4. AI-Powered Investigation**
1. Navigate to **🤖 LLM Investigator**
2. Enter your OpenAI API key
3. Select a customer for AI analysis
4. Get color-coded risk assessment and insights

## 🛠️ Troubleshooting

### **Common Issues**

1. **Docker Not Running**
   ```bash
   # Check Docker status
   docker info
   
   # Start Docker (Linux)
   sudo systemctl start docker
   ```

2. **Port Already in Use**
   ```bash
   # Check port usage
   lsof -i :8501
   
   # Stop conflicting applications
   ./stop-docker.sh
   ```

3. **Data Not Loading**
   - Ensure CSV files are properly formatted
   - Check file permissions
   - Verify column names match expected format

### **Logs and Debugging**

```bash
# View application logs
cd docker
docker-compose logs -f

# Check container status
docker ps

# Restart the application
docker-compose restart
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Streamlit** for the web framework
- **Scikit-learn** for machine learning capabilities
- **NetworkX** for graph analysis
- **OpenAI** for LLM integration
- **Plotly** for interactive visualizations

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/csingh033/aml-dashboard/issues)
- **Documentation**: [Wiki](https://github.com/csingh033/aml-dashboard/wiki)
- **Email**: chandan.singh@beyond.one

---

**🎉 Ready to detect money laundering patterns with AI-powered insights!**

**Access the dashboard**: `http://localhost:8501` 
