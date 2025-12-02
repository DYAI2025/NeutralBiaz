# BiazNeutralize AI - Quick Start Deployment Guide

This guide helps you get BiazNeutralize AI up and running quickly in different environments.

## 🚀 Choose Your Deployment Method

### Option 1: Local Development with Docker Compose (5 minutes)

Perfect for testing and development.

```bash
# Clone repository
git clone https://github.com/your-org/BiazNeutralize_AI.git
cd BiazNeutralize_AI

# Copy environment file
cp .env.example .env

# Build and start services
docker-compose up -d

# Access the application
open http://localhost:3000
```

**What you get:**
- Full application stack
- PostgreSQL database
- Redis cache
- Frontend dashboard
- Background workers

### Option 2: Production with Kubernetes (15 minutes)

For scalable production deployments.

#### Prerequisites
- Kubernetes cluster (EKS, GKE, AKS, or local)
- kubectl configured
- Domain name with DNS access

```bash
# Navigate to Kubernetes configs
cd deployment/kubernetes

# Apply configurations
kubectl apply -f namespace.yaml
kubectl apply -f configmap.yaml
kubectl apply -f secrets.yaml  # Update with your secrets first
kubectl apply -f postgres-deployment.yaml
kubectl apply -f redis-deployment.yaml
kubectl apply -f backend-deployment.yaml
kubectl apply -f frontend-deployment.yaml

# Check deployment status
kubectl get pods -n biazneutralize
```

### Option 3: AWS Cloud with Terraform (30 minutes)

Complete cloud infrastructure setup.

```bash
# Prerequisites
aws configure  # Set up AWS credentials
terraform --version  # Ensure Terraform is installed

# Navigate to Terraform directory
cd deployment/terraform

# Initialize Terraform
terraform init

# Create configuration
cat > terraform.tfvars << EOF
project_name = "biazneutralize"
environment = "production"
aws_region = "us-east-1"
domain_name = "yourdomain.com"
route53_zone_id = "YOUR_ZONE_ID"
notification_email = "you@example.com"
EOF

# Deploy infrastructure
terraform plan -var-file=terraform.tfvars
terraform apply -var-file=terraform.tfvars
```

## ⚙️ Environment Configuration

### Required Environment Variables

Create a `.env` file with these essential variables:

```bash
# Application
ENVIRONMENT=production
SECRET_KEY=your-super-secret-key-here
DOMAIN_NAME=yourdomain.com

# Database
POSTGRES_DB=biazneutralize
POSTGRES_USER=postgres
POSTGRES_PASSWORD=secure-password

# Redis
REDIS_PASSWORD=redis-password

# Monitoring (optional)
SENTRY_DSN=your-sentry-dsn
```

### Generate Secret Key

```bash
python -c "import secrets; print(secrets.token_hex(32))"
```

## 🔧 Quick Configuration Checklist

### 1. Update Secrets (Kubernetes only)

```bash
# Update secrets.yaml with base64 encoded values
echo -n "your-secret-key" | base64
echo -n "your-postgres-password" | base64
```

### 2. Configure Domain (Production only)

Update these files with your domain:
- `deployment/kubernetes/frontend-deployment.yaml`
- `deployment/configs/environments/production.env`

### 3. SSL Certificate

For HTTPS, either:
- Use AWS Certificate Manager (recommended)
- Configure Let's Encrypt with cert-manager

## 🐛 Quick Troubleshooting

### Common Issues

1. **Pods won't start**
   ```bash
   kubectl describe pod <pod-name> -n biazneutralize
   kubectl logs <pod-name> -n biazneutralize
   ```

2. **Database connection errors**
   ```bash
   # Test database connectivity
   kubectl run -it --rm debug --image=postgres:15 --restart=Never -- \
     psql -h postgres-service -U postgres -d biazneutralize
   ```

3. **Frontend not loading**
   - Check if backend is running: `curl http://backend-url/health`
   - Verify CORS configuration
   - Check browser console for errors

## 📊 Verify Deployment

### Health Checks

```bash
# Basic health
curl https://yourdomain.com/health

# Detailed health
curl https://yourdomain.com/api/health/detailed

# Kubernetes health
kubectl get pods -n biazneutralize
```

### Test the Application

1. Open your browser to your domain
2. Try uploading a test document
3. Run bias detection
4. Check the results dashboard

## 🔍 Monitoring Setup

### Basic Monitoring

Access built-in monitoring:
- **Grafana**: `http://yourdomain.com:3001` (admin/your-grafana-password)
- **Prometheus**: `http://yourdomain.com:9090`
- **Flower** (Celery): `http://yourdomain.com:5555`

### Log Access

```bash
# Application logs
kubectl logs -f deployment/backend-deployment -n biazneutralize

# All services
kubectl logs -f -l app.kubernetes.io/name=biazneutralize -n biazneutralize
```

## 🚀 Production Readiness Checklist

- [ ] SSL/TLS certificate configured
- [ ] Secrets properly managed (not in plain text)
- [ ] Database backups configured
- [ ] Monitoring and alerting set up
- [ ] Resource limits and requests defined
- [ ] Health checks responding
- [ ] Load testing completed
- [ ] Security scanning passed
- [ ] Documentation updated

## 📞 Getting Help

### Useful Commands

```bash
# Check deployment status
kubectl get all -n biazneutralize

# View pod logs
kubectl logs -f <pod-name> -n biazneutralize

# Access pod shell
kubectl exec -it <pod-name> -n biazneutralize -- /bin/bash

# Port forward for local access
kubectl port-forward svc/frontend-service 8080:80 -n biazneutralize
```

### Support Resources

- **Full Documentation**: [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md)
- **Troubleshooting**: [TROUBLESHOOTING.md](TROUBLESHOOTING.md)
- **Security Guide**: [SECURITY.md](SECURITY.md)
- **Issue Tracker**: GitHub Issues

## 🔄 Next Steps

1. **Configure monitoring** for your environment
2. **Set up automated backups**
3. **Configure CI/CD pipeline**
4. **Review security settings**
5. **Plan scaling strategy**

---

🎉 **Congratulations!** Your BiazNeutralize AI system is now running. Start detecting and neutralizing bias in your content!