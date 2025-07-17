# 🚀 AML Dashboard Update Instructions for Windows

## Overview
This guide will help you update your AML Dashboard to fix the OpenAI integration error and get everything running smoothly on your Windows machine.

---

## 📥 Step 1: Pull Latest Changes

Open **Git Bash** or **Command Prompt** in your AML Dashboard directory and run:

```bash
# Navigate to your AML Dashboard folder
cd path/to/your/aml-dashboard

# Pull the latest changes from the repository
git pull origin main
```

**Expected Output:** You should see something like:
```
Updating fed4f8e..519004b
requirements.txt | 2 +-
app.py          | 1 +-
2 files changed, 3 insertions(+), 2 deletions(-)
```

---

## 🔧 Step 2: Stop Current Docker Container

If you have the dashboard running, stop it first:

```bash
# Stop all running containers (if any)
docker stop $(docker ps -q)

# Or stop specific container if you know the name
docker stop aml-dashboard
```

---

## 🏗️ Step 3: Rebuild Docker Image with Updated Dependencies

```bash
# Build the new Docker image with fixed dependencies
docker build -f docker/Dockerfile -t aml-dashboard .
```

**Expected Output:** You should see a successful build with messages like:
```
[+] Building 60s (12/12) FINISHED
 => [internal] load build definition from Dockerfile
 => [internal] load metadata for docker.io/library/python:3.9-slim
 => [1/7] FROM docker.io/library/python:3.9-slim
 => [2/7] WORKDIR /app
 => [3/7] RUN apt-get update && apt-get install -y gcc g++
 => [4/7] COPY requirements.txt .
 => [5/7] RUN pip install --no-cache-dir -r requirements.txt
 => [6/7] COPY app.py .
 => [7/7] COPY *.py ./
 => exporting to image
 => => writing image sha256:...
 => naming to docker.io/library/aml-dashboard:latest
```

---

## 🚀 Step 4: Run the Updated Dashboard

```bash
# Run the dashboard on port 8501
docker run -p 8501:8501 aml-dashboard
```

**Expected Output:**
```
Collecting usage statistics. To deactivate, set browser.gatherUsageStats to False.
  You can now view your Streamlit app in your browser.
  URL: http://0.0.0.0:8501
```

---

## 🌐 Step 5: Access the Dashboard

1. Open your web browser
2. Navigate to: `http://localhost:8501`
3. The dashboard should load without any OpenAI errors

---

## ✅ Step 6: Test the LLM Investigator

1. Go to the **"🤖 LLM Investigator"** tab
2. Enter your OpenAI API key
3. Select a customer to analyze
4. The analysis should work without the previous error

---

## 🔧 Troubleshooting

### **If Port 8501 is Already in Use:**
```bash
# Find what's using port 8501
netstat -ano | findstr :8501

# Kill the process (replace XXXX with the PID from above command)
taskkill /PID XXXX /F
```

### **If Docker Build Fails:**
```bash
# Clean up Docker cache and try again
docker system prune -a
docker build -f docker/Dockerfile -t aml-dashboard .
```

### **If Docker Daemon is Not Running:**
1. Open Docker Desktop
2. Wait for it to start completely
3. Try the commands again

### **If Git Pull Fails:**
```bash
# Save any local changes first
git stash

# Then pull
git pull origin main

# Restore changes if needed
git stash pop
```

---

## 📋 What Was Fixed

✅ **OpenAI Integration Error:** Fixed the "unexpected keyword argument 'proxies'" error  
✅ **Library Compatibility:** Updated openai and httpx versions for better compatibility  
✅ **Docker Build:** Ensured all dependencies work correctly in the container  

---

## 📞 Support

If you encounter any issues:
1. Check the troubleshooting section above
2. Ensure Docker Desktop is running
3. Verify your internet connection for pulling updates
4. Contact the development team if problems persist

---

## 🎯 Success Indicators

✅ Repository updated successfully  
✅ Docker image built without errors  
✅ Dashboard loads at http://localhost:8501  
✅ LLM Investigator tab works without OpenAI errors  
✅ All tabs and features are functional  

**You're all set! 🎉** 