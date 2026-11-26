variable "aws_region" {
  description = "AWS region where the EC2 instance and S3 bucket will be created."
  type        = string
  default     = "us-west-2"
}

variable "environment" {
  description = "Deployment environment (development, staging, production, testing)."
  type        = string
  default     = "development"
}

variable "workload_type" {
  description = "Primary workload type for tagging and sizing defaults."
  type        = string
  default     = "development"
}

variable "instance_name" {
  description = "Name tag for the EC2 instance."
  type        = string
  default     = "socavones-gdl-dev"
}

variable "instance_type" {
  description = "EC2 instance type. t3.small (2 GB RAM) is recommended for rasterio/numpy/matplotlib."
  type        = string
  default     = "t3.small"
}

variable "root_volume_size" {
  description = "Root EBS volume size in GB (gp3, encrypted)."
  type        = number
  default     = 20
}

variable "allowed_ssh_cidr" {
  description = "CIDR block allowed to SSH (port 22) into the EC2 instance."
  type        = string
  default     = "187.188.226.101/32"
}

variable "key_name" {
  description = "Optional existing EC2 key pair name for SSH access. If null, use AWS SSM Session Manager."
  type        = string
  default     = null
}

variable "s3_bucket_prefix" {
  description = "Prefix for the S3 bucket storing analyzed preview images."
  type        = string
  default     = "socavones-gdl-previews"
}

variable "owner" {
  description = "Owner tag for cost tracking."
  type        = string
  default     = "SocavonesGDL"
}
