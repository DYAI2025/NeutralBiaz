# Outputs for BiazNeutralize AI Infrastructure

# VPC Outputs
output "vpc_id" {
  description = "VPC ID"
  value       = module.vpc.vpc_id
}

output "vpc_cidr_block" {
  description = "VPC CIDR block"
  value       = module.vpc.vpc_cidr_block
}

output "private_subnet_ids" {
  description = "Private subnet IDs"
  value       = module.vpc.private_subnet_ids
}

output "public_subnet_ids" {
  description = "Public subnet IDs"
  value       = module.vpc.public_subnet_ids
}

# EKS Outputs
output "cluster_id" {
  description = "EKS cluster ID"
  value       = module.eks.cluster_id
}

output "cluster_arn" {
  description = "EKS cluster ARN"
  value       = module.eks.cluster_arn
}

output "cluster_endpoint" {
  description = "EKS cluster endpoint"
  value       = module.eks.cluster_endpoint
}

output "cluster_security_group_id" {
  description = "EKS cluster security group ID"
  value       = module.eks.cluster_security_group_id
}

output "cluster_iam_role_arn" {
  description = "IAM role ARN associated with EKS cluster"
  value       = module.eks.cluster_iam_role_arn
}

output "node_groups" {
  description = "EKS node groups"
  value       = module.eks.node_groups
}

# RDS Outputs
output "rds_endpoint" {
  description = "RDS instance endpoint"
  value       = module.rds.db_instance_endpoint
  sensitive   = true
}

output "rds_port" {
  description = "RDS instance port"
  value       = module.rds.db_instance_port
}

output "rds_identifier" {
  description = "RDS instance identifier"
  value       = module.rds.db_instance_identifier
}

output "rds_arn" {
  description = "RDS instance ARN"
  value       = module.rds.db_instance_arn
}

# ElastiCache Outputs
output "redis_endpoint" {
  description = "Redis cluster endpoint"
  value       = module.elasticache.cache_cluster_address
  sensitive   = true
}

output "redis_port" {
  description = "Redis cluster port"
  value       = module.elasticache.cache_cluster_port
}

# S3 Outputs
output "s3_bucket_names" {
  description = "S3 bucket names"
  value       = module.s3.bucket_names
}

output "s3_bucket_arns" {
  description = "S3 bucket ARNs"
  value       = module.s3.bucket_arns
}

# Load Balancer Outputs
output "alb_dns_name" {
  description = "Application Load Balancer DNS name"
  value       = module.alb.dns_name
}

output "alb_zone_id" {
  description = "Application Load Balancer zone ID"
  value       = module.alb.zone_id
}

output "alb_arn" {
  description = "Application Load Balancer ARN"
  value       = module.alb.arn
}

# Certificate Outputs
output "certificate_arn" {
  description = "ACM certificate ARN"
  value       = module.acm.certificate_arn
}

output "certificate_status" {
  description = "ACM certificate status"
  value       = module.acm.certificate_status
}

# DNS Outputs
output "domain_name" {
  description = "Domain name"
  value       = var.domain_name
}

output "api_domain_name" {
  description = "API domain name"
  value       = "api.${var.domain_name}"
}

# IAM Outputs
output "eks_worker_iam_role_arn" {
  description = "IAM role ARN for EKS worker nodes"
  value       = module.iam.eks_worker_iam_role_arn
}

output "application_iam_role_arn" {
  description = "IAM role ARN for application services"
  value       = module.iam.application_iam_role_arn
}

# Monitoring Outputs
output "cloudwatch_log_group_names" {
  description = "CloudWatch log group names"
  value       = module.monitoring.log_group_names
}

output "sns_topic_arn" {
  description = "SNS topic ARN for alerts"
  value       = module.monitoring.sns_topic_arn
}

# Security Group Outputs
output "database_security_group_id" {
  description = "Database security group ID"
  value       = module.security_groups.database_security_group_id
}

output "redis_security_group_id" {
  description = "Redis security group ID"
  value       = module.security_groups.redis_security_group_id
}

output "alb_security_group_id" {
  description = "ALB security group ID"
  value       = module.security_groups.alb_security_group_id
}

# Configuration Outputs for kubectl
output "kubectl_config" {
  description = "kubectl config command"
  value       = "aws eks update-kubeconfig --name ${module.eks.cluster_id} --region ${var.aws_region}"
}

# Environment-specific outputs
output "environment_info" {
  description = "Environment information"
  value = {
    environment    = var.environment
    project_name   = var.project_name
    aws_region     = var.aws_region
    cluster_name   = local.cluster_name
    vpc_id         = module.vpc.vpc_id
    domain_name    = var.domain_name
  }
}

# Database connection information
output "database_config" {
  description = "Database configuration for applications"
  value = {
    host     = module.rds.db_instance_address
    port     = module.rds.db_instance_port
    database = var.db_name
    username = var.db_username
  }
  sensitive = true
}

# Redis connection information
output "redis_config" {
  description = "Redis configuration for applications"
  value = {
    host = module.elasticache.cache_cluster_address
    port = module.elasticache.cache_cluster_port
  }
  sensitive = true
}

# Application URLs
output "application_urls" {
  description = "Application URLs"
  value = {
    frontend = "https://${var.domain_name}"
    api      = "https://api.${var.domain_name}"
    admin    = "https://admin.${var.domain_name}"
  }
}