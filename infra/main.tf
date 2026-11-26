data "aws_vpc" "default" {
  default = true
}

data "aws_subnets" "default" {
  filter {
    name   = "vpc-id"
    values = [data.aws_vpc.default.id]
  }

  filter {
    name   = "default-for-az"
    values = ["true"]
  }
}

data "aws_ami" "al2023" {
  most_recent = true
  owners      = ["amazon"]

  filter {
    name   = "name"
    values = ["al2023-ami-2023.*-x86_64"]
  }

  filter {
    name   = "state"
    values = ["available"]
  }

  filter {
    name   = "virtualization-type"
    values = ["hvm"]
  }
}

resource "random_id" "bucket_suffix" {
  byte_length = 4
}

# ------------------------------------------------------------------------------
# S3 Bucket for Analyzed Preview Images
# ------------------------------------------------------------------------------
resource "aws_s3_bucket" "previews" {
  bucket        = "${var.s3_bucket_prefix}-${var.environment}-${random_id.bucket_suffix.hex}"
  force_destroy = var.environment == "development" || var.environment == "testing"

  tags = {
    Name = "${var.s3_bucket_prefix}-${var.environment}"
  }
}

resource "aws_s3_bucket_public_access_block" "previews" {
  bucket = aws_s3_bucket.previews.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_server_side_encryption_configuration" "previews" {
  bucket = aws_s3_bucket.previews.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

resource "aws_s3_bucket_versioning" "previews" {
  bucket = aws_s3_bucket.previews.id

  versioning_configuration {
    status = "Enabled"
  }
}

# ------------------------------------------------------------------------------
# IAM Role & Instance Profile (SSM + Least-Privilege S3 Access)
# ------------------------------------------------------------------------------
resource "aws_iam_role" "ec2" {
  name        = "${var.instance_name}-role"
  description = "Least-privilege IAM role for ${var.instance_name} (SSM + preview S3 bucket)"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Principal = {
          Service = "ec2.amazonaws.com"
        }
        Action = "sts:AssumeRole"
      }
    ]
  })
}

resource "aws_iam_role_policy_attachment" "ssm" {
  role       = aws_iam_role.ec2.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore"
}

resource "aws_iam_role_policy" "s3_previews" {
  name = "${var.instance_name}-s3-previews"
  role = aws_iam_role.ec2.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "s3:PutObject",
          "s3:GetObject"
        ]
        Resource = "${aws_s3_bucket.previews.arn}/*"
      },
      {
        Effect   = "Allow"
        Action   = ["s3:ListBucket"]
        Resource = aws_s3_bucket.previews.arn
      }
    ]
  })
}

resource "aws_iam_instance_profile" "ec2" {
  name = "${var.instance_name}-profile"
  role = aws_iam_role.ec2.name
}

# ------------------------------------------------------------------------------
# Hardened Security Group (SSH restricted to your IP only)
# ------------------------------------------------------------------------------
resource "aws_security_group" "ec2" {
  name        = "${var.instance_name}-sg"
  description = "Hardened security group for ${var.instance_name} (${var.workload_type})"
  vpc_id      = data.aws_vpc.default.id

  ingress {
    description = "SSH from user IP only"
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = [var.allowed_ssh_cidr]
  }

  egress {
    description = "Allow all outbound (STAC API, S3 COGs, package updates, SSM)"
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name = "${var.instance_name}-sg"
  }
}

# ------------------------------------------------------------------------------
# EC2 Instance (Amazon Linux 2023 + Pre-installed NDVI Script)
# ------------------------------------------------------------------------------
resource "aws_instance" "ndvi" {
  ami                         = data.aws_ami.al2023.id
  instance_type               = var.instance_type
  subnet_id                   = sort(data.aws_subnets.default.ids)[0]
  vpc_security_group_ids      = [aws_security_group.ec2.id]
  iam_instance_profile        = aws_iam_instance_profile.ec2.name
  key_name                    = var.key_name
  associate_public_ip_address = true
  monitoring                  = false
  disable_api_termination     = false

  credit_specification {
    cpu_credits = "standard"
  }

  metadata_options {
    http_endpoint               = "enabled"
    http_tokens                 = "required"
    http_put_response_hop_limit = 1
  }

  root_block_device {
    volume_size           = var.root_volume_size
    volume_type           = "gp3"
    encrypted             = true
    delete_on_termination = true
  }

  user_data = templatefile("${path.module}/user_data.sh.tftpl", {
    s3_bucket  = aws_s3_bucket.previews.bucket
    aws_region = var.aws_region
  })

  user_data_replace_on_change = true

  tags = {
    Name = var.instance_name
  }

  volume_tags = {
    Name = "${var.instance_name}-root"
  }
}
