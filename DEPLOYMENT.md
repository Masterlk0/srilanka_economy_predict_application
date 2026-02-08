# Deployment Guide

## Local Testing
```bash
streamlit run app.py
```

## AWS EC2 Deployment

### Step 1: Launch EC2 Instance
- Choose Ubuntu Server 22.04 LTS
- Instance type: t2.micro (free tier) or t2.small
- Configure Security Group: Allow inbound traffic on port 8501

### Step 2: Connect to EC2 and Install Dependencies
```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Python and pip
sudo apt install python3-pip -y

# Install required packages
pip3 install streamlit pandas numpy scikit-learn joblib
```

### Step 3: Upload Files
```bash
# From your local machine, upload files using SCP
scp -i your-key.pem app.py ubuntu@your-ec2-ip:~/
scp -i your-key.pem requirements.txt ubuntu@your-ec2-ip:~/
scp -i your-key.pem *.pkl ubuntu@your-ec2-ip:~/
scp -i your-key.pem *.csv ubuntu@your-ec2-ip:~/
```

### Step 4: Run Streamlit
```bash
# Run in background with nohup
nohup streamlit run app.py --server.port 8501 --server.address 0.0.0.0 &

# Or use screen for persistent session
screen -S streamlit
streamlit run app.py --server.port 8501 --server.address 0.0.0.0
# Press Ctrl+A then D to detach
```

### Step 5: Access App
Open browser: `http://your-ec2-public-ip:8501`

## Streamlit Cloud Deployment (Easiest)

1. Push code to GitHub repository
2. Go to https://share.streamlit.io
3. Sign in with GitHub
4. Click "New app"
5. Select your repository, branch, and app.py
6. Click "Deploy"

Note: Upload all .pkl files to GitHub (ensure they're not in .gitignore)

## Alternative: Docker Deployment

Create Dockerfile:
```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

Build and run:
```bash
docker build -t gdp-predictor .
docker run -p 8501:8501 gdp-predictor
```
