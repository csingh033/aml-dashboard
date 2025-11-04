# Fix AWS Upload Size Limit (413 Error)

## Problem
Getting `AxiosError: Request failed with status code 413` when uploading CSV files on AWS EC2, even for files as small as 6MB.

## Root Cause
This is caused by NGINX (or other reverse proxy) limiting the upload size, **not** Streamlit itself.

---

## Solution 1: Using Docker with Built-in NGINX (Recommended)

### Step 1: Pull latest changes
```bash
cd /path/to/aml-dashboard
git pull origin main
```

### Step 2: Stop current containers
```bash
cd docker
docker-compose down
```

### Step 3: Use the new NGINX-enabled docker-compose
```bash
# Rename or backup old compose file
mv docker-compose.yml docker-compose-original.yml

# Use the new compose file with NGINX
cp docker-compose-nginx.yml docker-compose.yml

# Build and start with NGINX
docker-compose build --no-cache
docker-compose up -d
```

### Step 4: Update EC2 Security Group
Make sure your EC2 security group allows:
- **Port 80** (HTTP) - Inbound from 0.0.0.0/0
- **Port 443** (HTTPS) - Inbound from 0.0.0.0/0 (if using SSL)

### Step 5: Test
- Access your dashboard at: `http://your-ec2-public-ip`
- Try uploading your CSV file

---

## Solution 2: If You're Using NGINX Installed on EC2 Directly

### Step 1: SSH into your EC2 instance
```bash
ssh -i your-key.pem ubuntu@your-ec2-ip
```

### Step 2: Edit NGINX configuration
```bash
sudo nano /etc/nginx/nginx.conf
```

### Step 3: Add this inside the `http` block
```nginx
http {
    # ... existing config ...
    
    # Increase upload size limit
    client_max_body_size 1000M;
    client_body_timeout 300s;
    client_body_buffer_size 128k;
    
    # ... rest of config ...
}
```

### Step 4: OR edit your site-specific config
```bash
sudo nano /etc/nginx/sites-available/default
```

Add inside the `server` block:
```nginx
server {
    listen 80;
    
    # Increase upload size limit
    client_max_body_size 1000M;
    client_body_timeout 300s;
    
    location / {
        proxy_pass http://localhost:8502;
        proxy_http_version 1.1;
        
        # WebSocket support for Streamlit
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        
        # Standard headers
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        
        # Timeout settings
        proxy_connect_timeout 300s;
        proxy_send_timeout 300s;
        proxy_read_timeout 300s;
    }
}
```

### Step 5: Test and reload NGINX
```bash
# Test configuration
sudo nginx -t

# Reload NGINX
sudo systemctl reload nginx

# Check status
sudo systemctl status nginx
```

---

## Solution 3: Using AWS Application Load Balancer (ALB)

If you're using AWS ALB, you don't need to worry - ALB supports up to **1GB** file uploads by default.

However, ensure your **Target Group health checks** are configured properly:
- Health check path: `/`
- Healthy threshold: 2
- Unhealthy threshold: 2
- Timeout: 30 seconds
- Interval: 60 seconds

---

## Solution 4: Check if NGINX is Running

### Check if NGINX is installed and running
```bash
# Check if nginx is running
sudo systemctl status nginx

# Or check with ps
ps aux | grep nginx
```

### If NGINX is not running but installed
```bash
sudo systemctl start nginx
sudo systemctl enable nginx
```

### If you don't want to use NGINX
```bash
# Stop nginx
sudo systemctl stop nginx
sudo systemctl disable nginx

# Access directly on port 8502
# Update EC2 security group to allow port 8502
# Access: http://your-ec2-ip:8502
```

---

## Verification

After applying any solution, verify the configuration:

### Test NGINX config (if using NGINX)
```bash
sudo nginx -t
```

### Check upload limit
```bash
# On EC2 instance
curl -X POST -F "file=@large-file.csv" http://localhost/upload

# Or check NGINX error logs
sudo tail -f /var/log/nginx/error.log
```

### Monitor Docker logs
```bash
cd docker
docker-compose logs -f
```

---

## Quick Reference: Common Upload Size Limits

| Service | Default Limit | Fix Location |
|---------|--------------|--------------|
| Streamlit | 200MB | `.streamlit/config.toml` ✅ (Already fixed) |
| NGINX | 1MB | `/etc/nginx/nginx.conf` ⚠️ (Needs fix) |
| AWS ALB | 1GB | No fix needed ✅ |
| Apache | Varies | `/etc/apache2/apache2.conf` |
| Docker | No limit | N/A |

---

## Troubleshooting

### Still getting 413 error?

1. **Check all NGINX configs:**
   ```bash
   sudo grep -r "client_max_body_size" /etc/nginx/
   ```

2. **Check if multiple NGINX instances:**
   ```bash
   ps aux | grep nginx
   ```

3. **Check Docker logs:**
   ```bash
   docker-compose logs aml-dashboard
   docker-compose logs nginx
   ```

4. **Check EC2 system limits:**
   ```bash
   ulimit -a
   ```

5. **Verify the fix is applied:**
   ```bash
   # For system NGINX
   sudo nginx -T | grep client_max_body_size
   
   # For Docker NGINX
   docker exec aml-nginx cat /etc/nginx/conf.d/default.conf
   ```

---

## Contact

If you continue to experience issues, please provide:
1. NGINX configuration: `sudo nginx -T`
2. Docker logs: `docker-compose logs`
3. EC2 security group settings
4. Whether you're using ALB/CloudFront

