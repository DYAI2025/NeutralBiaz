# Security Guide for BiazNeutralize AI

This document outlines security best practices, configurations, and procedures for deploying and maintaining BiazNeutralize AI in production environments.

## Table of Contents

1. [Security Architecture](#security-architecture)
2. [Network Security](#network-security)
3. [Application Security](#application-security)
4. [Data Security](#data-security)
5. [Container Security](#container-security)
6. [Kubernetes Security](#kubernetes-security)
7. [Monitoring and Incident Response](#monitoring-and-incident-response)
8. [Compliance](#compliance)
9. [Security Checklist](#security-checklist)

## Security Architecture

### Defense in Depth

BiazNeutralize AI implements multiple layers of security:

```
┌─────────────────────────────────────────────┐
│                 Edge/CDN                    │  ← DDoS Protection, WAF
├─────────────────────────────────────────────┤
│              Load Balancer                  │  ← SSL Termination, Rate Limiting
├─────────────────────────────────────────────┤
│               Application                   │  ← Authentication, Authorization
├─────────────────────────────────────────────┤
│            Container Runtime                │  ← Runtime Security, Isolation
├─────────────────────────────────────────────┤
│                 Network                     │  ← Firewalls, Segmentation
├─────────────────────────────────────────────┤
│              Infrastructure                 │  ← Encryption, Access Control
└─────────────────────────────────────────────┘
```

### Security Principles

1. **Principle of Least Privilege**: Grant minimal necessary permissions
2. **Zero Trust**: Verify every request, regardless of source
3. **Defense in Depth**: Multiple security layers
4. **Security by Design**: Built-in security from the start
5. **Continuous Monitoring**: Real-time threat detection

## Network Security

### VPC Configuration

```terraform
# VPC with private subnets
resource "aws_vpc" "main" {
  cidr_block           = "10.0.0.0/16"
  enable_dns_hostnames = true
  enable_dns_support   = true

  tags = {
    Name = "biazneutralize-vpc"
  }
}

# Private subnets for application and database
resource "aws_subnet" "private" {
  count             = 2
  vpc_id            = aws_vpc.main.id
  cidr_block        = "10.0.${count.index + 1}.0/24"
  availability_zone = data.aws_availability_zones.available.names[count.index]

  tags = {
    Name = "biazneutralize-private-${count.index + 1}"
  }
}

# Public subnets only for load balancer
resource "aws_subnet" "public" {
  count                   = 2
  vpc_id                  = aws_vpc.main.id
  cidr_block              = "10.0.${count.index + 100}.0/24"
  availability_zone       = data.aws_availability_zones.available.names[count.index]
  map_public_ip_on_launch = true

  tags = {
    Name = "biazneutralize-public-${count.index + 1}"
  }
}
```

### Security Groups

#### Application Load Balancer
```terraform
resource "aws_security_group" "alb" {
  name_prefix = "biazneutralize-alb"
  vpc_id      = aws_vpc.main.id

  ingress {
    from_port   = 443
    to_port     = 443
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]  # HTTPS from anywhere
  }

  ingress {
    from_port   = 80
    to_port     = 80
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]  # HTTP redirect to HTTPS
  }

  egress {
    from_port   = 0
    to_port     = 65535
    protocol    = "tcp"
    cidr_blocks = [aws_vpc.main.cidr_block]  # Only to VPC
  }
}
```

#### Application Tier
```terraform
resource "aws_security_group" "app" {
  name_prefix = "biazneutralize-app"
  vpc_id      = aws_vpc.main.id

  ingress {
    from_port       = 8000
    to_port         = 8000
    protocol        = "tcp"
    security_groups = [aws_security_group.alb.id]  # Only from ALB
  }

  egress {
    from_port   = 443
    to_port     = 443
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]  # HTTPS for external APIs
  }

  egress {
    from_port       = 5432
    to_port         = 5432
    protocol        = "tcp"
    security_groups = [aws_security_group.rds.id]  # To database
  }

  egress {
    from_port       = 6379
    to_port         = 6379
    protocol        = "tcp"
    security_groups = [aws_security_group.redis.id]  # To Redis
  }
}
```

#### Database Tier
```terraform
resource "aws_security_group" "rds" {
  name_prefix = "biazneutralize-rds"
  vpc_id      = aws_vpc.main.id

  ingress {
    from_port       = 5432
    to_port         = 5432
    protocol        = "tcp"
    security_groups = [aws_security_group.app.id]  # Only from app
  }
}
```

### Network Policies (Kubernetes)

```yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: deny-all-default
  namespace: biazneutralize
spec:
  podSelector: {}
  policyTypes:
  - Ingress
  - Egress

---
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: allow-backend-to-database
  namespace: biazneutralize
spec:
  podSelector:
    matchLabels:
      app.kubernetes.io/component: backend
  policyTypes:
  - Egress
  egress:
  - to:
    - podSelector:
        matchLabels:
          app.kubernetes.io/component: database
    ports:
    - protocol: TCP
      port: 5432
```

## Application Security

### Authentication and Authorization

#### JWT Token Configuration

```python
# JWT settings
JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY")
JWT_ALGORITHM = "HS256"
JWT_EXPIRATION_TIME = timedelta(hours=24)
JWT_REFRESH_EXPIRATION_TIME = timedelta(days=30)

# Token validation
def validate_token(token: str) -> Dict[str, Any]:
    try:
        payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")
```

#### Role-Based Access Control (RBAC)

```python
class UserRole(Enum):
    USER = "user"
    ADMIN = "admin"
    ANALYST = "analyst"

class Permission(Enum):
    READ_ANALYSIS = "read:analysis"
    WRITE_ANALYSIS = "write:analysis"
    ADMIN_PANEL = "admin:panel"
    MANAGE_USERS = "manage:users"

ROLE_PERMISSIONS = {
    UserRole.USER: [Permission.READ_ANALYSIS, Permission.WRITE_ANALYSIS],
    UserRole.ANALYST: [Permission.READ_ANALYSIS, Permission.WRITE_ANALYSIS],
    UserRole.ADMIN: [perm for perm in Permission]
}
```

### Input Validation and Sanitization

```python
from pydantic import BaseModel, validator, Field
from typing import List, Optional
import bleach

class BiasAnalysisRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=50000)
    analysis_type: str = Field(..., regex="^(sentiment|cultural|cognitive)$")
    language: Optional[str] = Field("en", regex="^(en|de|fr|es)$")

    @validator('text')
    def sanitize_text(cls, v):
        # Remove potentially harmful content
        cleaned = bleach.clean(v, tags=[], attributes={}, strip=True)
        return cleaned.strip()

    @validator('text')
    def validate_text_content(cls, v):
        # Check for suspicious patterns
        if any(pattern in v.lower() for pattern in ['<script', 'javascript:', 'data:']):
            raise ValueError('Invalid text content')
        return v
```

### Rate Limiting

```python
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

# Rate limiter configuration
limiter = Limiter(
    key_func=get_remote_address,
    storage_uri="redis://redis:6379",
    default_limits=["1000 per hour"]
)

app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

@app.post("/api/analyze")
@limiter.limit("10 per minute")
async def analyze_text(
    request: Request,
    analysis_request: BiasAnalysisRequest
):
    # Analysis logic
    pass
```

### Security Headers

```python
from starlette.middleware.base import BaseHTTPMiddleware

class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        response = await call_next(request)

        # Security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; "
            "script-src 'self' 'unsafe-inline'; "
            "style-src 'self' 'unsafe-inline'; "
            "img-src 'self' data: https:; "
            "font-src 'self' data:; "
            "connect-src 'self';"
        )

        return response

app.add_middleware(SecurityHeadersMiddleware)
```

## Data Security

### Encryption at Rest

#### Database Encryption

```terraform
resource "aws_db_instance" "postgres" {
  # ... other configuration ...

  storage_encrypted = true
  kms_key_id       = aws_kms_key.database.arn

  # Backup encryption
  backup_retention_period = 7
  backup_window          = "03:00-04:00"
  copy_tags_to_snapshot  = true
}

resource "aws_kms_key" "database" {
  description             = "KMS key for database encryption"
  deletion_window_in_days = 7

  tags = {
    Name = "biazneutralize-database-key"
  }
}
```

#### S3 Encryption

```terraform
resource "aws_s3_bucket" "storage" {
  bucket = "biazneutralize-storage"

  server_side_encryption_configuration {
    rule {
      apply_server_side_encryption_by_default {
        sse_algorithm     = "aws:kms"
        kms_master_key_id = aws_kms_key.s3.arn
      }
    }
  }

  versioning {
    enabled = true
  }

  public_access_block {
    block_public_acls       = true
    block_public_policy     = true
    ignore_public_acls      = true
    restrict_public_buckets = true
  }
}
```

### Encryption in Transit

#### TLS Configuration

```nginx
server {
    listen 443 ssl http2;
    server_name biazneutralize.example.com;

    # SSL configuration
    ssl_certificate /etc/ssl/certs/server.crt;
    ssl_certificate_key /etc/ssl/private/server.key;

    # Strong SSL settings
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-RSA-AES256-GCM-SHA512:DHE-RSA-AES256-GCM-SHA512:ECDHE-RSA-AES256-GCM-SHA384;
    ssl_prefer_server_ciphers off;

    # SSL security headers
    add_header Strict-Transport-Security "max-age=63072000; includeSubDomains; preload";

    # OCSP stapling
    ssl_stapling on;
    ssl_stapling_verify on;
}
```

### Secrets Management

#### AWS Secrets Manager

```terraform
resource "aws_secretsmanager_secret" "database_password" {
  name        = "biazneutralize/database/password"
  description = "Database password for BiazNeutralize AI"

  rotation_rules {
    automatically_after_days = 90
  }
}

resource "aws_secretsmanager_secret_version" "database_password" {
  secret_id     = aws_secretsmanager_secret.database_password.id
  secret_string = random_password.database_password.result
}
```

#### Kubernetes Secrets

```yaml
apiVersion: v1
kind: Secret
metadata:
  name: biazneutralize-secrets
  namespace: biazneutralize
type: Opaque
data:
  # Base64 encoded values
  SECRET_KEY: <base64-encoded-secret>
  POSTGRES_PASSWORD: <base64-encoded-password>
  REDIS_PASSWORD: <base64-encoded-password>
```

## Container Security

### Secure Base Images

```dockerfile
# Use official, minimal base image
FROM python:3.11-slim

# Update packages and remove package manager
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    curl && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    apt-get remove -y apt-get

# Create non-root user
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Install dependencies as root
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application and change ownership
COPY --chown=appuser:appuser . /app
WORKDIR /app

# Switch to non-root user
USER appuser

# Set security options
LABEL security.no-new-privileges=true
```

### Container Security Scanning

```yaml
# GitHub Actions security scanning
- name: Run Trivy vulnerability scanner
  uses: aquasecurity/trivy-action@master
  with:
    image-ref: 'biazneutralize/backend:${{ github.sha }}'
    format: 'sarif'
    output: 'trivy-results.sarif'

- name: Upload Trivy scan results to GitHub Security tab
  uses: github/codeql-action/upload-sarif@v2
  with:
    sarif_file: 'trivy-results.sarif'
```

### Runtime Security

```yaml
# Pod Security Context
securityContext:
  runAsNonRoot: true
  runAsUser: 1000
  runAsGroup: 1000
  fsGroup: 1000
  seccompProfile:
    type: RuntimeDefault

# Container Security Context
securityContext:
  allowPrivilegeEscalation: false
  readOnlyRootFilesystem: true
  capabilities:
    drop:
      - ALL
  runAsNonRoot: true
  runAsUser: 1000
```

## Kubernetes Security

### Pod Security Standards

```yaml
apiVersion: v1
kind: Namespace
metadata:
  name: biazneutralize
  labels:
    pod-security.kubernetes.io/enforce: restricted
    pod-security.kubernetes.io/audit: restricted
    pod-security.kubernetes.io/warn: restricted
```

### Service Accounts and RBAC

```yaml
apiVersion: v1
kind: ServiceAccount
metadata:
  name: biazneutralize-app
  namespace: biazneutralize
automountServiceAccountToken: false

---
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  namespace: biazneutralize
  name: biazneutralize-role
rules:
- apiGroups: [""]
  resources: ["pods", "services"]
  verbs: ["get", "list"]
- apiGroups: [""]
  resources: ["configmaps", "secrets"]
  verbs: ["get"]

---
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  name: biazneutralize-rolebinding
  namespace: biazneutralize
subjects:
- kind: ServiceAccount
  name: biazneutralize-app
  namespace: biazneutralize
roleRef:
  kind: Role
  name: biazneutralize-role
  apiGroup: rbac.authorization.k8s.io
```

### Admission Controllers

```yaml
# OPA Gatekeeper Policy
apiVersion: templates.gatekeeper.sh/v1beta1
kind: ConstraintTemplate
metadata:
  name: k8srequiredsecuritycontext
spec:
  crd:
    spec:
      names:
        kind: K8sRequiredSecurityContext
      validation:
        type: object
  targets:
    - target: admission.k8s.gatekeeper.sh
      rego: |
        package k8srequiredsecuritycontext

        violation[{"msg": msg}] {
            container := input.review.object.spec.containers[_]
            not container.securityContext.runAsNonRoot
            msg := "Container must run as non-root user"
        }

        violation[{"msg": msg}] {
            container := input.review.object.spec.containers[_]
            container.securityContext.allowPrivilegeEscalation
            msg := "Container must not allow privilege escalation"
        }
```

## Monitoring and Incident Response

### Security Monitoring

#### Log Analysis

```yaml
# Falco rules for container security
- rule: Shell in Container
  desc: Detect shell access in container
  condition: >
    spawned_process and container and
    shell_procs and proc.pname exists
  output: >
    Shell spawned in container (user=%user.name container=%container.name
    shell=%proc.name parent=%proc.pname cmdline=%proc.cmdline)
  priority: WARNING

- rule: Sensitive File Access
  desc: Detect access to sensitive files
  condition: >
    open_read and fd.name startswith /etc and
    (fd.name contains passwd or fd.name contains shadow)
  output: >
    Sensitive file access (user=%user.name file=%fd.name
    command=%proc.cmdline container=%container.name)
  priority: HIGH
```

#### Prometheus Security Metrics

```python
from prometheus_client import Counter, Histogram

# Security metrics
failed_auth_total = Counter('auth_failures_total', 'Total authentication failures')
request_duration = Histogram('request_duration_seconds', 'Request duration')
suspicious_activity = Counter('suspicious_activity_total', 'Suspicious activity detected')

# Track failed authentication
@app.middleware("http")
async def security_middleware(request: Request, call_next):
    start_time = time.time()

    try:
        response = await call_next(request)

        # Track failed authentication
        if response.status_code == 401:
            failed_auth_total.inc()

        return response
    except Exception as e:
        suspicious_activity.inc()
        raise e
    finally:
        request_duration.observe(time.time() - start_time)
```

### Incident Response Plan

#### 1. Detection and Analysis

**Automated Alerts:**
- Failed authentication attempts (>10 in 5 minutes)
- Unusual API usage patterns
- Container security violations
- Network anomalies

**Response Team:**
- Security Lead
- DevOps Engineer
- Application Developer
- Business Stakeholder

#### 2. Containment

**Immediate Actions:**
```bash
# Block suspicious IP
kubectl create -f - <<EOF
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: block-suspicious-ip
  namespace: biazneutralize
spec:
  podSelector: {}
  policyTypes:
  - Ingress
  ingress:
  - from:
    - ipBlock:
        cidr: 0.0.0.0/0
        except:
        - SUSPICIOUS_IP/32
EOF

# Scale down compromised service
kubectl scale deployment compromised-service --replicas=0 -n biazneutralize

# Rotate secrets
kubectl delete secret compromised-secret -n biazneutralize
kubectl create secret generic new-secret --from-literal=key=new-value
```

#### 3. Eradication and Recovery

**Clean Environment:**
```bash
# Redeploy from clean images
kubectl rollout restart deployment/backend-deployment -n biazneutralize

# Update all secrets
./scripts/rotate-secrets.sh

# Apply security patches
./scripts/security-update.sh
```

#### 4. Post-Incident Activities

- Document incident timeline
- Update security procedures
- Conduct lessons learned session
- Implement additional controls

## Compliance

### GDPR Compliance

#### Data Minimization

```python
class UserData(BaseModel):
    # Only collect necessary data
    user_id: str
    email: str
    created_at: datetime
    last_login: Optional[datetime]

    # No sensitive personal data stored
    # Preferences stored separately with consent

class DataRetentionPolicy:
    USER_DATA_RETENTION = timedelta(days=2555)  # 7 years
    LOG_RETENTION = timedelta(days=90)
    ANALYSIS_DATA_RETENTION = timedelta(days=365)
```

#### Data Subject Rights

```python
@app.delete("/api/user/{user_id}/data")
async def delete_user_data(user_id: str, current_user: User = Depends(get_current_user)):
    # Right to erasure implementation
    if current_user.user_id != user_id and not current_user.is_admin:
        raise HTTPException(status_code=403, detail="Not authorized")

    # Delete user data across all systems
    await delete_user_from_database(user_id)
    await delete_user_files(user_id)
    await delete_user_logs(user_id)

    return {"message": "User data deleted successfully"}

@app.get("/api/user/{user_id}/data")
async def export_user_data(user_id: str, current_user: User = Depends(get_current_user)):
    # Data portability implementation
    if current_user.user_id != user_id and not current_user.is_admin:
        raise HTTPException(status_code=403, detail="Not authorized")

    user_data = await collect_user_data(user_id)
    return {"data": user_data, "format": "json"}
```

### SOC 2 Type II

#### Access Control

```python
class AccessControl:
    def __init__(self):
        self.access_log = logging.getLogger("access_control")

    def verify_access(self, user: User, resource: str, action: str) -> bool:
        # Log all access attempts
        self.access_log.info(f"Access attempt: {user.id} -> {resource} ({action})")

        # Check permissions
        if not user.has_permission(resource, action):
            self.access_log.warning(f"Access denied: {user.id} -> {resource} ({action})")
            return False

        self.access_log.info(f"Access granted: {user.id} -> {resource} ({action})")
        return True
```

#### Audit Logging

```python
import structlog

audit_logger = structlog.get_logger("audit")

@app.middleware("http")
async def audit_middleware(request: Request, call_next):
    # Log all API requests
    audit_logger.info(
        "api_request",
        method=request.method,
        path=request.url.path,
        client_ip=request.client.host,
        user_agent=request.headers.get("user-agent"),
        timestamp=datetime.utcnow().isoformat()
    )

    response = await call_next(request)

    audit_logger.info(
        "api_response",
        status_code=response.status_code,
        response_time=time.time() - start_time
    )

    return response
```

## Security Checklist

### Pre-Deployment Security Checklist

#### Infrastructure Security
- [ ] VPC configured with private subnets
- [ ] Security groups follow least privilege
- [ ] Network ACLs configured
- [ ] NAT Gateway for outbound traffic only
- [ ] VPC Flow Logs enabled
- [ ] CloudTrail enabled with log integrity validation

#### Application Security
- [ ] All secrets stored in secure secret management
- [ ] Strong authentication implemented
- [ ] Role-based access control configured
- [ ] Input validation and sanitization in place
- [ ] Rate limiting configured
- [ ] Security headers implemented
- [ ] CORS properly configured

#### Container Security
- [ ] Base images scanned for vulnerabilities
- [ ] Containers run as non-root users
- [ ] Read-only root filesystem enabled
- [ ] Security contexts properly configured
- [ ] No unnecessary capabilities granted
- [ ] Resource limits defined

#### Kubernetes Security
- [ ] Pod Security Standards enforced
- [ ] Network policies implemented
- [ ] RBAC configured with least privilege
- [ ] Service accounts properly configured
- [ ] Admission controllers configured
- [ ] Container runtime security enabled

#### Data Security
- [ ] Encryption at rest enabled
- [ ] Encryption in transit enforced
- [ ] Database access restricted
- [ ] Backup encryption enabled
- [ ] Data retention policies implemented

#### Monitoring Security
- [ ] Security monitoring tools deployed
- [ ] Log aggregation configured
- [ ] Alerting rules defined
- [ ] Incident response procedures documented
- [ ] Security metrics collection enabled

### Post-Deployment Security Checklist

#### Operational Security
- [ ] Security patches applied regularly
- [ ] Vulnerability scans performed
- [ ] Penetration testing conducted
- [ ] Security training completed
- [ ] Incident response plan tested

#### Compliance
- [ ] GDPR compliance verified
- [ ] Data retention policies enforced
- [ ] Audit logs maintained
- [ ] User consent mechanisms implemented
- [ ] Data subject rights procedures tested

---

For additional security guidance or to report security issues, please contact the security team or create a confidential security issue in the project repository.