# BiazNeutralize AI - Production Deployment Guide

This comprehensive guide covers deploying the BiazNeutralize AI system to production environments using Docker, Kubernetes, and cloud infrastructure.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Infrastructure Overview](#infrastructure-overview)
3. [Environment Setup](#environment-setup)
4. [Deployment Methods](#deployment-methods)
5. [Configuration Management](#configuration-management)
6. [Security Considerations](#security-considerations)
7. [Monitoring and Observability](#monitoring-and-observability)
8. [Backup and Recovery](#backup-and-recovery)
9. [Troubleshooting](#troubleshooting)
10. [Maintenance Procedures](#maintenance-procedures)

## Prerequisites

### Required Tools

- **Docker**: Version 20.10 or higher
- **Docker Compose**: Version 2.0 or higher
- **kubectl**: Version 1.26 or higher
- **AWS CLI**: Version 2.0 or higher
- **Terraform**: Version 1.0 or higher
- **Helm**: Version 3.8 or higher (optional)

### Cloud Account Requirements

- **AWS Account** with appropriate IAM permissions
- **Domain name** with DNS management access
- **SSL certificate** (ACM or custom)

### System Requirements

#### Minimum Requirements
- **CPU**: 2 vCPUs
- **Memory**: 4 GB RAM
- **Storage**: 50 GB
- **Network**: 1 Gbps

#### Recommended Production Requirements
- **CPU**: 8+ vCPUs
- **Memory**: 16+ GB RAM
- **Storage**: 200+ GB SSD
- **Network**: 10 Gbps

## Infrastructure Overview

### Architecture Components

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Application   │    │     Database     │    │      Cache      │
│   Load Balancer │───▶│   PostgreSQL     │    │      Redis      │
│      (ALB)      │    │    (RDS/EKS)     │    │   (ElastiCache) │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │
         ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│    Frontend     │    │     Backend      │    │   Background    │
│   (React SPA)   │    │   (FastAPI)      │    │    Workers      │
│   Nginx/Docker  │    │   Python/Docker  │    │    (Celery)     │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

### Technology Stack

- **Frontend**: React 19, TypeScript, Vite, Tailwind CSS
- **Backend**: Python 3.11, FastAPI, SQLAlchemy, Celery
- **Database**: PostgreSQL 15
- **Cache**: Redis 7
- **Container Platform**: Docker, Kubernetes
- **Cloud Provider**: AWS (EKS, RDS, ElastiCache, S3)
- **Monitoring**: Prometheus, Grafana, CloudWatch

## Environment Setup

### 1. Clone Repository

```bash
git clone https://github.com/your-org/BiazNeutralize_AI.git
cd BiazNeutralize_AI
```

### 2. Configure Environment Variables

Copy the environment template:

```bash
cp .env.example .env.production
```

Edit `.env.production` with your production values:

```bash
# Application Settings
ENVIRONMENT=production
SECRET_KEY=your-super-secret-key-here
DOMAIN_NAME=biazneutralize.example.com

# Database Configuration
POSTGRES_DB=biazneutralize
POSTGRES_USER=postgres
POSTGRES_PASSWORD=secure-password-here

# Redis Configuration
REDIS_PASSWORD=redis-secure-password

# AWS Configuration
AWS_REGION=us-east-1
AWS_ACCOUNT_ID=123456789012

# Monitoring
SENTRY_DSN=your-sentry-dsn
GRAFANA_PASSWORD=grafana-admin-password
```

### 3. Set Up AWS Infrastructure

#### Option A: Using Terraform (Recommended)

```bash
cd deployment/terraform

# Initialize Terraform
terraform init

# Create terraform.tfvars
cat > terraform.tfvars << EOF
project_name = "biazneutralize"
environment = "production"
aws_region = "us-east-1"
domain_name = "biazneutralize.example.com"
route53_zone_id = "Z123456789ABCDEF"
notification_email = "alerts@example.com"
EOF

# Plan deployment
terraform plan -var-file=terraform.tfvars

# Apply infrastructure
terraform apply -var-file=terraform.tfvars
```

#### Option B: Using AWS Console

Manual setup steps are provided in [AWS_MANUAL_SETUP.md](AWS_MANUAL_SETUP.md).

## Deployment Methods

### Method 1: Docker Compose (Simple Deployment)

Best for: Small scale, single-server deployments

```bash
# Navigate to deployment directory
cd deployment/docker

# Copy environment file
cp ../configs/environments/production.env .env

# Deploy services
docker-compose -f docker-compose.prod.yml up -d

# Check status
docker-compose -f docker-compose.prod.yml ps
```

### Method 2: Kubernetes (Recommended for Production)

Best for: Scalable, enterprise deployments

#### Prerequisites

1. **Configure kubectl**:
```bash
aws eks update-kubeconfig --name biazneutralize-production --region us-east-1
```

2. **Verify cluster access**:
```bash
kubectl cluster-info
kubectl get nodes
```

#### Deployment Steps

1. **Apply configurations**:
```bash
cd deployment/kubernetes

# Apply in order
kubectl apply -f namespace.yaml
kubectl apply -f configmap.yaml
kubectl apply -f secrets.yaml
kubectl apply -f postgres-deployment.yaml
kubectl apply -f redis-deployment.yaml
kubectl apply -f backend-deployment.yaml
kubectl apply -f frontend-deployment.yaml
```

2. **Wait for deployment**:
```bash
kubectl rollout status deployment/backend-deployment -n biazneutralize
kubectl rollout status deployment/frontend-deployment -n biazneutralize
```

3. **Verify deployment**:
```bash
kubectl get pods -n biazneutralize
kubectl get services -n biazneutralize
```

### Method 3: Automated Deployment Script

```bash
# Make script executable
chmod +x deployment/scripts/deploy.sh

# Deploy to staging
./deployment/scripts/deploy.sh staging

# Deploy to production (with confirmation)
./deployment/scripts/deploy.sh production

# Deploy with specific tag
./deployment/scripts/deploy.sh production --tag v1.2.3

# Dry run (show what would be deployed)
./deployment/scripts/deploy.sh production --dry-run
```

## Configuration Management

### Environment-Specific Settings

Configuration files are located in `deployment/configs/environments/`:

- `production.env` - Production environment
- `staging.env` - Staging environment
- `development.env` - Development environment

### Kubernetes Configuration

#### ConfigMaps

Store non-sensitive configuration:

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: biazneutralize-config
  namespace: biazneutralize
data:
  API_HOST: "0.0.0.0"
  API_PORT: "8000"
  LOG_LEVEL: "info"
  ENVIRONMENT: "production"
```

#### Secrets

Store sensitive information:

```bash
# Create database secret
kubectl create secret generic postgres-secret \
  --from-literal=POSTGRES_PASSWORD=your-password \
  -n biazneutralize

# Create application secrets
kubectl create secret generic biazneutralize-secrets \
  --from-literal=SECRET_KEY=your-secret-key \
  --from-literal=SENTRY_DSN=your-sentry-dsn \
  -n biazneutralize
```

### SSL/TLS Configuration

#### Using AWS Certificate Manager

```bash
# Request certificate
aws acm request-certificate \
  --domain-name biazneutralize.example.com \
  --subject-alternative-names \"*.biazneutralize.example.com\" \
  --validation-method DNS \
  --region us-east-1
```

#### Using Let's Encrypt with cert-manager

```bash
# Install cert-manager
kubectl apply -f https://github.com/cert-manager/cert-manager/releases/download/v1.13.0/cert-manager.yaml

# Create ClusterIssuer
kubectl apply -f deployment/configs/cert-manager/cluster-issuer.yaml
```

## Security Considerations

### Network Security

1. **VPC Configuration**:
   - Private subnets for database and application
   - Public subnets only for load balancer
   - NAT Gateway for outbound traffic

2. **Security Groups**:
   - Restrictive ingress rules
   - Database access only from application
   - Redis access only from application

3. **Network Policies**:
   - Pod-to-pod communication restrictions
   - Ingress traffic control
   - Egress traffic control

### Application Security

1. **Container Security**:
   - Non-root user execution
   - Read-only root filesystem
   - Minimal base images
   - Regular security scans

2. **API Security**:
   - Rate limiting
   - Request validation
   - CORS configuration
   - Security headers

3. **Data Security**:
   - Encryption at rest
   - Encryption in transit
   - Secure secret management
   - Data retention policies

### Compliance

1. **GDPR Compliance**:
   - Data privacy controls
   - User consent management
   - Right to erasure
   - Data portability

2. **CCPA Compliance**:
   - Privacy disclosures
   - User rights implementation
   - Data deletion procedures

## Monitoring and Observability

### Health Checks

The application provides multiple health check endpoints:

- `/health` - Basic application status
- `/health/detailed` - Comprehensive system status
- `/ready` - Kubernetes readiness probe
- `/live` - Kubernetes liveness probe
- `/metrics` - Prometheus metrics

### Prometheus Metrics

Key metrics to monitor:

- **Application Metrics**:
  - Request rate and latency
  - Error rate
  - Active users
  - Model inference time

- **System Metrics**:
  - CPU and memory usage
  - Disk I/O and storage
  - Network traffic
  - Container restart count

- **Business Metrics**:
  - Bias detection accuracy
  - Processing throughput
  - User engagement

### Grafana Dashboards

Pre-configured dashboards are available in `deployment/configs/grafana/dashboards/`:

- Application Performance Dashboard
- Infrastructure Overview Dashboard
- Business Metrics Dashboard
- Error Tracking Dashboard

### Log Management

Centralized logging with structured JSON logs:

```json
{
  \"timestamp\": \"2024-01-15T10:30:00Z\",
  \"level\": \"INFO\",
  \"service\": \"bias-detection\",
  \"user_id\": \"12345\",
  \"request_id\": \"abc-123-def\",
  \"message\": \"Bias analysis completed\",
  \"metrics\": {
    \"processing_time_ms\": 250,
    \"confidence_score\": 0.85
  }
}
```

### Alerting

Alert rules are configured for:

- High error rates (>5%)
- Response time degradation (>2s)
- Resource utilization (>80%)
- Failed health checks
- Security incidents

## Backup and Recovery

### Database Backup

#### Automated Backups

```bash
# PostgreSQL backup script
#!/bin/bash
BACKUP_DIR=\"/backups/postgres\"
TIMESTAMP=$(date +\"%Y%m%d_%H%M%S\")
BACKUP_FILE=\"biazneutralize_backup_$TIMESTAMP.sql\"

# Create backup
pg_dump -h $POSTGRES_HOST -U $POSTGRES_USER -d $POSTGRES_DB > \"$BACKUP_DIR/$BACKUP_FILE\"

# Compress backup
gzip \"$BACKUP_DIR/$BACKUP_FILE\"

# Upload to S3
aws s3 cp \"$BACKUP_DIR/$BACKUP_FILE.gz\" s3://your-backup-bucket/postgres/
```

#### Backup Schedule

- **Daily**: Full database backup
- **Weekly**: Archive backup to cold storage
- **Monthly**: Test restore procedures

### Disaster Recovery

#### Recovery Time Objectives (RTO)

- **Critical Services**: 15 minutes
- **Supporting Services**: 1 hour
- **Full System**: 4 hours

#### Recovery Point Objectives (RPO)

- **Database**: 15 minutes
- **File Storage**: 1 hour
- **Configuration**: Real-time

#### Recovery Procedures

1. **Database Recovery**:
```bash
# Restore from latest backup
aws s3 cp s3://your-backup-bucket/postgres/latest.sql.gz ./
gunzip latest.sql.gz
psql -h $POSTGRES_HOST -U $POSTGRES_USER -d $POSTGRES_DB < latest.sql
```

2. **Application Recovery**:
```bash
# Redeploy from last known good configuration
kubectl rollout undo deployment/backend-deployment -n biazneutralize
kubectl rollout undo deployment/frontend-deployment -n biazneutralize
```

## Troubleshooting

### Common Issues

#### 1. Pod Startup Failures

**Symptoms**: Pods in CrashLoopBackOff state

**Diagnosis**:
```bash
kubectl describe pod <pod-name> -n biazneutralize
kubectl logs <pod-name> -n biazneutralize
```

**Common Causes**:
- Missing environment variables
- Invalid configuration
- Database connection failures
- Resource limits too low

#### 2. Database Connection Issues

**Symptoms**: Connection timeouts or refused connections

**Diagnosis**:
```bash
# Test database connectivity
kubectl run -it --rm debug --image=postgres:15 --restart=Never -- \
  psql -h postgres-service -U postgres -d biazneutralize
```

**Solutions**:
- Check security group rules
- Verify network policies
- Confirm credentials

#### 3. High Memory Usage

**Symptoms**: Out of memory errors

**Diagnosis**:
```bash
# Check resource usage
kubectl top pods -n biazneutralize
kubectl describe node <node-name>
```

**Solutions**:
- Increase memory limits
- Enable horizontal pod autoscaling
- Optimize application memory usage

### Performance Optimization

#### Database Optimization

1. **Connection Pooling**:
```python
# SQLAlchemy configuration
engine = create_engine(
    DATABASE_URL,
    pool_size=20,
    max_overflow=30,
    pool_timeout=30,
    pool_recycle=3600
)
```

2. **Query Optimization**:
- Use appropriate indexes
- Optimize N+1 queries
- Implement query caching

#### Application Optimization

1. **Caching Strategy**:
```python
# Redis caching
@cache.memoize(timeout=3600)
def expensive_computation(data):
    # Cached for 1 hour
    return process_data(data)
```

2. **Resource Management**:
- Implement connection pooling
- Use async operations
- Optimize model loading

### Log Analysis

Common log patterns to watch:

```bash
# Error patterns
kubectl logs -n biazneutralize -l app=backend | grep ERROR

# Performance issues
kubectl logs -n biazneutralize -l app=backend | grep \"slow_query\"

# Security events
kubectl logs -n biazneutralize -l app=backend | grep \"auth_failure\"
```

## Maintenance Procedures

### Regular Maintenance

#### Daily Tasks

- [ ] Check system health dashboards
- [ ] Review error logs
- [ ] Verify backup completion
- [ ] Monitor resource usage

#### Weekly Tasks

- [ ] Update security patches
- [ ] Review performance metrics
- [ ] Test alert systems
- [ ] Clean up old logs and backups

#### Monthly Tasks

- [ ] Review capacity planning
- [ ] Update dependencies
- [ ] Conduct security assessment
- [ ] Test disaster recovery procedures

### Update Procedures

#### Application Updates

1. **Staging Deployment**:
```bash
# Deploy to staging
./deployment/scripts/deploy.sh staging --tag v1.2.3

# Run integration tests
./scripts/integration-tests.sh staging

# Performance testing
./scripts/load-tests.sh staging
```

2. **Production Deployment**:
```bash
# Rolling deployment
./deployment/scripts/deploy.sh production --tag v1.2.3

# Monitor deployment
kubectl rollout status deployment/backend-deployment -n biazneutralize

# Verify health
curl https://biazneutralize.example.com/health
```

3. **Rollback if Needed**:
```bash
# Quick rollback
./deployment/scripts/deploy.sh production --rollback

# Or manual rollback
kubectl rollout undo deployment/backend-deployment -n biazneutralize
```

#### Infrastructure Updates

1. **Terraform Updates**:
```bash
cd deployment/terraform

# Plan changes
terraform plan -var-file=terraform.tfvars

# Apply with approval
terraform apply -var-file=terraform.tfvars
```

2. **Kubernetes Updates**:
```bash
# Update cluster
aws eks update-cluster-version --name biazneutralize-production --version 1.28

# Update node groups
aws eks update-nodegroup-version --cluster-name biazneutralize-production --nodegroup-name general
```

### Security Updates

#### OS and Container Updates

1. **Rebuild containers** with latest base images
2. **Scan for vulnerabilities** using Trivy or similar tools
3. **Update dependencies** in requirements files
4. **Deploy updates** using standard procedures

#### Certificate Renewal

1. **Monitor certificate expiration**
2. **Renew certificates** 30 days before expiration
3. **Update Kubernetes secrets** with new certificates
4. **Verify SSL/TLS configuration**

---

## Support and Resources

- **Documentation**: [docs/](../README.md)
- **Issue Tracking**: GitHub Issues
- **Monitoring Dashboards**: Grafana instance
- **Log Analysis**: CloudWatch or ELK Stack
- **Security Alerts**: Email notifications

For additional support or questions, please contact the DevOps team or create an issue in the project repository.