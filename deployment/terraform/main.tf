# BiazNeutralize AI Infrastructure
# Terraform configuration for AWS deployment

terraform {
  required_version = ">= 1.0"
  
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = "~> 2.20"
    }
    helm = {
      source  = "hashicorp/helm"
      version = "~> 2.10"
    }
    tls = {
      source  = "hashicorp/tls"
      version = "~> 4.0"
    }
  }

  backend "s3" {
    bucket = "biazneutralize-terraform-state"
    key    = "infrastructure/terraform.tfstate"
    region = "us-east-1"
    
    # Enable state locking and consistency
    dynamodb_table = "terraform-state-lock"
    encrypt        = true
  }
}

# Configure AWS Provider
provider "aws" {
  region = var.aws_region
  
  default_tags {
    tags = {
      Project     = "BiazNeutralize-AI"
      Environment = var.environment
      ManagedBy   = "Terraform"
      Owner       = var.project_owner
    }
  }
}

# Data sources
data "aws_availability_zones" "available" {
  state = "available"
}

data "aws_caller_identity" "current" {}

# Local variables
locals {
  cluster_name = "${var.project_name}-${var.environment}"
  
  common_tags = {
    Project     = var.project_name
    Environment = var.environment
    ManagedBy   = "Terraform"
    Owner       = var.project_owner
  }
}

# VPC Module
module "vpc" {
  source = "./modules/vpc"
  
  project_name        = var.project_name
  environment         = var.environment
  vpc_cidr           = var.vpc_cidr
  availability_zones = data.aws_availability_zones.available.names
  
  tags = local.common_tags
}

# EKS Module
module "eks" {
  source = "./modules/eks"
  
  cluster_name    = local.cluster_name
  cluster_version = var.kubernetes_version
  
  vpc_id         = module.vpc.vpc_id
  subnet_ids     = module.vpc.private_subnet_ids
  
  node_groups = var.eks_node_groups
  
  tags = local.common_tags
}

# RDS Module
module "rds" {
  source = "./modules/rds"
  
  identifier     = "${var.project_name}-${var.environment}-postgres"
  engine_version = var.postgres_version
  instance_class = var.rds_instance_class
  
  allocated_storage     = var.rds_allocated_storage
  max_allocated_storage = var.rds_max_allocated_storage
  
  db_name  = var.db_name
  username = var.db_username
  
  vpc_id                = module.vpc.vpc_id
  subnet_ids            = module.vpc.private_subnet_ids
  allowed_security_groups = [module.eks.node_security_group_id]
  
  backup_retention_period = var.rds_backup_retention
  backup_window          = var.rds_backup_window
  maintenance_window     = var.rds_maintenance_window
  
  tags = local.common_tags
}

# ElastiCache Module
module "elasticache" {
  source = "./modules/elasticache"
  
  cluster_id          = "${var.project_name}-${var.environment}-redis"
  node_type          = var.redis_node_type
  num_cache_nodes    = var.redis_num_nodes
  parameter_group_name = "default.redis7"
  
  vpc_id     = module.vpc.vpc_id
  subnet_ids = module.vpc.private_subnet_ids
  allowed_security_groups = [module.eks.node_security_group_id]
  
  tags = local.common_tags
}

# S3 Buckets for application storage
module "s3" {
  source = "./modules/s3"
  
  project_name = var.project_name
  environment  = var.environment
  
  bucket_names = [
    "model-cache",
    "user-uploads",
    "backups",
    "logs"
  ]
  
  tags = local.common_tags
}

# Application Load Balancer
module "alb" {
  source = "./modules/alb"
  
  name            = "${var.project_name}-${var.environment}-alb"
  vpc_id          = module.vpc.vpc_id
  public_subnets  = module.vpc.public_subnet_ids
  certificate_arn = module.acm.certificate_arn
  
  tags = local.common_tags
}

# ACM Certificate
module "acm" {
  source = "./modules/acm"
  
  domain_name = var.domain_name
  zone_id     = var.route53_zone_id
  
  tags = local.common_tags
}

# Route53 DNS
module "route53" {
  source = "./modules/route53"
  
  zone_id     = var.route53_zone_id
  domain_name = var.domain_name
  alb_dns_name = module.alb.dns_name
  alb_zone_id  = module.alb.zone_id
  
  tags = local.common_tags
}

# CloudWatch Monitoring
module "monitoring" {
  source = "./modules/monitoring"
  
  project_name = var.project_name
  environment  = var.environment
  
  cluster_name = local.cluster_name
  rds_identifier = module.rds.db_instance_identifier
  
  notification_email = var.notification_email
  
  tags = local.common_tags
}

# IAM Roles and Policies
module "iam" {
  source = "./modules/iam"
  
  project_name = var.project_name
  environment  = var.environment
  
  cluster_name = local.cluster_name
  s3_bucket_arns = module.s3.bucket_arns
  
  tags = local.common_tags
}

# Security Groups
module "security_groups" {
  source = "./modules/security-groups"
  
  project_name = var.project_name
  environment  = var.environment
  vpc_id       = module.vpc.vpc_id
  
  tags = local.common_tags
}