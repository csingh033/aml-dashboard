# 🚀 AML Dashboard - Easy Stakeholder Deployment

This guide helps stakeholders easily run the AML Dashboard on their laptops using Docker.

## 📋 Prerequisites

### 1. Install Docker
- **Windows/Mac**: Download [Docker Desktop](https://www.docker.com/products/docker-desktop)
- **Linux**: Run `sudo apt-get install docker.io` (Ubuntu/Debian) or `sudo yum install docker` (CentOS/RHEL)

### 2. Start Docker
- **Windows/Mac**: Open Docker Desktop application
- **Linux**: Run `sudo systemctl start docker`

## 🚀 Quick Start

### Step 1: Download the Package
1. Download the AML Dashboard package to your laptop
2. Extract the files to a folder (e.g., `aml-dashboard`)

### Step 2: Run the Dashboard
```bash
# Navigate to the folder
cd aml-dashboard

# Make scripts executable (Linux/Mac only)
chmod +x run-aml-dashboard.sh stop-aml-dashboard.sh

# Start the dashboard
./run-aml-dashboard.sh
```

### Step 3: Access the Dashboard
1. Open your web browser
2. Go to: `http://localhost:8501`
3. The AML Dashboard will load automatically

## 📁 Using the Dashboard

### 1. Upload Data
- Place your CSV transaction files in the `./data` folder
- The dashboard will automatically detect them
- Upload files through the web interface

### 2. Navigate the Dashboard
- **📊 EDA**: Explore transaction data with charts
- **💼 AML Dashboard**: Run anomaly detection and analysis
- **🤖 LLM Investigator**: Get AI-powered insights (requires OpenAI API key)
- **🔍 Customer ID Lookup**: Reverse hash customer IDs
- **ℹ️ Model Information**: Learn about the methodology

### 3. LLM Investigator Setup
1. Get an OpenAI API key from [OpenAI Platform](https://platform.openai.com/api-keys)
2. Enter the API key in the LLM Investigator tab
3. Select a customer to analyze

## 🔧 Management Commands

### Start the Dashboard
```bash
./run-aml-dashboard.sh
```

### Stop the Dashboard
```bash
./stop-aml-dashboard.sh
```

### View Logs
```bash
docker-compose logs -f
```

### Restart the Dashboard
```bash
docker-compose restart
```

### Update the Application
```bash
# Stop the current version
./stop-aml-dashboard.sh

# Pull the latest version and restart
docker-compose pull
./run-aml-dashboard.sh
```

## 📊 Features

### 🔍 Anomaly Detection
- **Isolation Forest**: Detects unusual transactions
- **Explainable AI**: Provides reasons for flagged transactions
- **Customer Consolidation**: Aggregates suspicious activity per customer

### 🌐 Network Analysis
- **Transaction Networks**: Visualize money flow between customers
- **Cycle Detection**: Identify potential laundering rings
- **Interactive Graphs**: Explore customer connections

### 🤖 AI-Powered Analysis
- **LLM Investigator**: GPT-4o powered analysis
- **Risk Assessment**: Color-coded risk levels (RED/GREEN/BLUE)
- **Actionable Insights**: 4-key bullet point recommendations

### 🔐 Privacy Features
- **Data Hashing**: SHA-256 encryption of customer IDs
- **Secure Access**: No data leaves your laptop
- **Audit Trail**: Complete transaction history

## 🛠️ Troubleshooting

### Common Issues

1. **Docker Not Running**
   ```bash
   # Check Docker status
   docker info
   
   # Start Docker (Linux)
   sudo systemctl start docker
   ```

2. **Port Already in Use**
   ```bash
   # Check what's using port 8501
   lsof -i :8501
   
   # Stop the dashboard and restart
   ./stop-aml-dashboard.sh
   ./run-aml-dashboard.sh
   ```

3. **Application Not Loading**
   ```bash
   # Check container status
   docker ps
   
   # View logs
   docker-compose logs
   ```

4. **Data Not Loading**
   - Ensure CSV files are in the `./data` folder
   - Check file format (should be CSV)
   - Verify file permissions

### System Requirements

- **RAM**: Minimum 4GB, Recommended 8GB
- **Storage**: 2GB free space
- **OS**: Windows 10+, macOS 10.14+, or Linux
- **Browser**: Chrome, Firefox, Safari, or Edge

## 🔒 Security Notes

- **Local Deployment**: All data stays on your laptop
- **No Internet Required**: Works offline (except for LLM features)
- **No Data Sharing**: No information is sent to external servers
- **Privacy First**: Customer IDs are automatically hashed

## 📞 Support

If you encounter issues:

1. **Check the logs**: `docker-compose logs`
2. **Restart the application**: `./stop-aml-dashboard.sh && ./run-aml-dashboard.sh`
3. **Verify Docker**: `docker info`
4. **Contact IT Support**: For technical assistance

## 🎯 Quick Reference

| Action | Command |
|--------|---------|
| Start Dashboard | `./run-aml-dashboard.sh` |
| Stop Dashboard | `./stop-aml-dashboard.sh` |
| View Logs | `docker-compose logs -f` |
| Restart | `docker-compose restart` |
| Update | `docker-compose pull && ./run-aml-dashboard.sh` |

---

**🎉 Your AML Dashboard is ready to use!**

Access it at: `http://localhost:8501` 