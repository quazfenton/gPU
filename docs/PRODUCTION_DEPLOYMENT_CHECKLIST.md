# Production Deployment Checklist

**Version:** 1.0.0  
**Last Updated:** March 3, 2026

---

## Pre-Deployment Requirements

### 1. Generate Secure Secrets

```bash
# Generate MASTER_KEY (32+ bytes)
export MASTER_KEY=$(python -c "import secrets; print(secrets.token_hex(32))")

# Generate JWT_SECRET (32+ bytes)
export JWT_SECRET=$(python -c "import secrets; print(secrets.token_hex(32))")

# Generate CREDENTIAL_SALT (64 hex characters)
export CREDENTIAL_SALT=$(python -c "import secrets; print(secrets.token_hex(32))")

# Verify lengths
echo "MASTER_KEY length: ${#MASTER_KEY}"  # Should be 64 (32 bytes hex)
echo "JWT_SECRET length: ${#JWT_SECRET}"  # Should be 64 (32 bytes hex)
echo "CREDENTIAL_SALT length: ${#CREDENTIAL_SALT}"  # Should be 64 (32 bytes hex)
```

### 2. Backend Credentials

Collect and securely store:
- [ ] Modal API credentials (token_id, token_secret)
- [ ] HuggingFace API token
- [ ] Kaggle credentials (username, key)
- [ ] Google OAuth credentials (if using Colab backend)
- [ ] AWS credentials (if using AWS backend or Secrets Manager)
- [ ] Azure credentials (if using Azure backend or Key Vault)

### 3. Infrastructure Requirements

**Minimum Resources:**
- [ ] CPU: 2 cores
- [ ] Memory: 4GB RAM
- [ ] Storage: 10GB for application + 50GB for uploads
- [ ] Network: Outbound HTTPS access

**Recommended Resources:**
- [ ] CPU: 4 cores
- [ ] Memory: 8GB RAM
- [ ] Storage: 20GB for application + 100GB for uploads
- [ ] Network: Load balancer with SSL termination

---

## Docker Deployment

### 1. Create .env File

```bash
# Copy example
cp .env.example .env

# Edit with secure values
vim .env
```

**Required Variables:**
```env
# Security (MUST be set)
MASTER_KEY=<generated-above>
JWT_SECRET=<generated-above>
CREDENTIAL_SALT=<generated-above>

# Application
APP_ENV=production
DEBUG=false

# GUI
GUI_ENABLE_AUTH=true
GUI_ENABLE_RATE_LIMITING=true
```

### 2. Deploy with Docker Compose

```bash
# Start services
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f orchestrator

# Verify health
curl http://localhost:7860/health
```

### 3. Configure Backends

```bash
# Access container
docker-compose exec orchestrator bash

# Store backend credentials securely
python -c "
from notebook_ml_orchestrator.security import CredentialStore
store = CredentialStore()
store.set_credential('modal', 'token_id', 'your-modal-token-id')
store.set_credential('modal', 'token_secret', 'your-modal-token-secret')
store.set_credential('huggingface', 'token', 'your-hf-token')
"
```

---

## Kubernetes Deployment

### 1. Create Secrets

```bash
# Create namespace (optional)
kubectl create namespace orchestrator

# Create secrets with secure values
kubectl create secret generic orchestrator-secrets \
  --from-literal=master-key=$(python -c "import secrets; print(secrets.token_hex(32))") \
  --from-literal=jwt-secret=$(python -c "import secrets; print(secrets.token_hex(32))") \
  --from-literal=credential-salt=$(python -c "import secrets; print(secrets.token_hex(32))") \
  -n orchestrator
```

### 2. Apply Manifests

```bash
# Apply all manifests
kubectl apply -f k8s/deployment.yaml

# Check deployment
kubectl get pods -l app=notebook-ml-orchestrator -n orchestrator
kubectl get svc notebook-ml-orchestrator -n orchestrator

# View logs
kubectl logs -l app=notebook-ml-orchestrator -n orchestrator -f
```

### 3. Configure Backends

```bash
# Store backend credentials
kubectl exec -it <pod-name> -n orchestrator -- python -c "
from notebook_ml_orchestrator.security import CredentialStore
store = CredentialStore()
store.set_credential('modal', 'token_id', 'your-modal-token-id')
store.set_credential('modal', 'token_secret', 'your-modal-token-secret')
"
```

---

## Helm Deployment

### 1. Create Values File

```yaml
# my-values.yaml
replicaCount: 3

security:
  # These will be created as Kubernetes secrets
  masterKey: ""  # Set via --set or secrets
  jwtSecret: ""
  credentialSalt: ""

config:
  appEnv: production
  debug: false
  gui:
    enableAuth: true
    enableRateLimiting: true
```

### 2. Install Chart

```bash
# Install with secrets
helm install orchestrator ./helm/notebook-ml-orchestrator \
  --values my-values.yaml \
  --set security.masterKey=$(python -c "import secrets; print(secrets.token_hex(32))") \
  --set security.jwtSecret=$(python -c "import secrets; print(secrets.token_hex(32))") \
  --set security.credentialSalt=$(python -c "import secrets; print(secrets.token_hex(32))") \
  --namespace orchestrator \
  --create-namespace
```

### 3. Verify Installation

```bash
# Check status
helm status orchestrator -n orchestrator

# Check pods
kubectl get pods -l app.kubernetes.io/name=notebook-ml-orchestrator -n orchestrator
```

---

## Post-Deployment Verification

### 1. Health Checks

```bash
# Application health
curl http://<host>:7860/health

# Expected response:
# {"status": "healthy", "version": "1.0.0", ...}
```

### 2. Authentication Test

```bash
# Try to access without auth (should fail if enabled)
curl http://<host>:7860/

# Login via GUI
# Open browser: http://<host>:7860
# Default admin user: admin / admin (CHANGE IMMEDIATELY!)
```

### 3. Backend Connectivity

```bash
# Test backend registration via GUI
# Navigate to: Backend Registration tab
# Register a backend and verify health status
```

### 4. Job Submission Test

```bash
# Submit a test job via GUI
# Navigate to: Job Submission tab
# Select a template and submit
# Verify job appears in Job Monitoring tab
```

---

## Security Hardening

### 1. Change Default Credentials

```bash
# Access GUI and change default admin password
# Or via API if available
```

### 2. Enable TLS/SSL

**Docker Compose:**
```yaml
# Add nginx proxy with SSL
services:
  nginx:
    image: nginx:alpine
    ports:
      - "443:443"
    volumes:
      - ./ssl/cert.pem:/etc/nginx/ssl/cert.pem
      - ./ssl/key.pem:/etc/nginx/ssl/key.pem
```

**Kubernetes:**
```yaml
# Add Ingress with TLS
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: orchestrator-ingress
spec:
  tls:
    - hosts:
        - orchestrator.example.com
      secretName: orchestrator-tls
```

### 3. Configure Network Policies

```bash
# Kubernetes network policy already included
# Verify it's applied:
kubectl get networkpolicy -n orchestrator
```

### 4. Enable Audit Logging

```bash
# Verify security logs are being written
docker-compose exec orchestrator tail -f /app/logs/security.log
# or
kubectl logs <pod-name> -c log-collector -n orchestrator
```

---

## Monitoring Setup

### 1. Prometheus Metrics

```bash
# Access metrics endpoint
curl http://<host>:8080/metrics

# Add to Prometheus config:
# scrape_configs:
#   - job_name: 'orchestrator'
#     static_configs:
#       - targets: ['<host>:8080']
```

### 2. Log Aggregation

**ELK Stack:**
```yaml
# Add Filebeat sidecar or use Fluentd
# Configure to ship logs to Elasticsearch
```

**Splunk:**
```bash
# Install Splunk Universal Forwarder
# Configure to monitor /app/logs/*.log
```

### 3. Alerting Rules

**Critical Alerts:**
- Pod/container restarts > 3 in 5 minutes
- Error rate > 5% of requests
- Response time p99 > 5 seconds
- Disk usage > 80%
- Memory usage > 90%

**Warning Alerts:**
- Error rate > 1% of requests
- Response time p95 > 2 seconds
- Disk usage > 70%
- Failed authentication attempts > 10 in 5 minutes

---

## Backup & Recovery

### 1. Database Backup

```bash
# Daily backup script
#!/bin/bash
DATE=$(date +%Y%m%d-%H%M%S)
docker cp orchestrator:/app/data/orchestrator.db ./backups/orchestrator-$DATE.db
# Keep last 7 days
find ./backups -name "*.db" -mtime +7 -delete
```

### 2. Credential Backup

```bash
# Backup encrypted credential store
docker cp orchestrator:/app/credentials.enc.json ./backups/credentials-$(date +%Y%m%d).enc.json

# IMPORTANT: Also backup MASTER_KEY separately!
```

### 3. Recovery Procedure

```bash
# Stop application
docker-compose down

# Restore database
docker cp ./backups/orchestrator-YYYYMMDD.db orchestrator:/app/data/orchestrator.db

# Restore credentials
docker cp ./backups/credentials-YYYYMMDD.enc.json orchestrator:/app/credentials.enc.json

# Start application
docker-compose up -d
```

---

## Troubleshooting

### Common Issues

**1. Authentication Failures**
```bash
# Check secrets are set correctly
docker-compose exec orchestrator env | grep -E 'MASTER_KEY|JWT_SECRET'

# Verify credential store
docker-compose exec orchestrator python -c "
from notebook_ml_orchestrator.security import CredentialStore
try:
    store = CredentialStore()
    print('Credential store OK')
except Exception as e:
    print(f'Credential store error: {e}')
"
```

**2. Database Errors**
```bash
# Check database file permissions
docker-compose exec orchestrator ls -la /app/data/

# Check disk space
docker-compose exec orchestrator df -h
```

**3. Backend Connection Issues**
```bash
# Test network connectivity
docker-compose exec orchestrator curl -I https://api.modal.com

# Check backend credentials
docker-compose exec orchestrator python -c "
from notebook_ml_orchestrator.security import CredentialStore
store = CredentialStore()
print('Modal token_id:', store.get_credential('modal', 'token_id'))
"
```

---

## Performance Tuning

### 1. Resource Limits

**Docker Compose:**
```yaml
services:
  orchestrator:
    deploy:
      resources:
        limits:
          cpus: '4.0'
          memory: 8G
        reservations:
          cpus: '1.0'
          memory: 2G
```

**Kubernetes:**
```yaml
resources:
  requests:
    cpu: 1000m
    memory: 2Gi
  limits:
    cpu: 4000m
    memory: 8Gi
```

### 2. Database Optimization

```python
# Enable WAL mode for better concurrency
sqlite3 /app/data/orchestrator.db "PRAGMA journal_mode=WAL;"
```

### 3. Caching

```bash
# Enable Redis caching in docker-compose.yml
# See docker-compose.yml for Redis service configuration
```

---

## Security Audit Checklist

- [ ] All secrets generated with secure random values
- [ ] No default credentials in production
- [ ] TLS/SSL enabled for external access
- [ ] Network policies applied (Kubernetes)
- [ ] Rate limiting enabled
- [ ] Authentication enabled
- [ ] Audit logging enabled
- [ ] Regular backup schedule configured
- [ ] Monitoring and alerting configured
- [ ] Security patches applied regularly

---

## Support & Maintenance

### Regular Tasks

**Daily:**
- [ ] Check error logs
- [ ] Verify backups completed
- [ ] Check disk space

**Weekly:**
- [ ] Review security logs
- [ ] Check for failed authentication attempts
- [ ] Review resource usage

**Monthly:**
- [ ] Apply security patches
- [ ] Review and rotate credentials
- [ ] Test backup restoration
- [ ] Review access logs

### Getting Help

- Documentation: `docs/` directory
- Issues: GitHub Issues
- Security: Contact security team for vulnerabilities

---

**Deployment Status:** [ ] Complete  
**Deployed By:** ________________  
**Date:** ________________  
**Verified By:** ________________
