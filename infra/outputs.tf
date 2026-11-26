output "instance_id" {
  description = "ID of the launched EC2 instance."
  value       = aws_instance.ndvi.id
}

output "public_ip" {
  description = "Public IP address of the EC2 instance."
  value       = aws_instance.ndvi.public_ip
}

output "s3_bucket_name" {
  description = "S3 bucket where preview images are saved when --save-image is used."
  value       = aws_s3_bucket.previews.bucket
}

output "security_group_id" {
  description = "Security group ID attached to the EC2 instance."
  value       = aws_security_group.ec2.id
}

output "iam_role_name" {
  description = "IAM role name attached to the EC2 instance."
  value       = aws_iam_role.ec2.name
}

output "ssm_connect_command" {
  description = "Connect to the instance via AWS SSM Session Manager (no SSH key required)."
  value       = "aws ssm start-session --target ${aws_instance.ndvi.id} --region ${var.aws_region}"
}

output "ssh_connect_command" {
  description = "Connect via SSH from 187.188.226.101/32 (requires key_name to be set)."
  value       = var.key_name != null ? "ssh -i ${var.key_name}.pem ec2-user@${aws_instance.ndvi.public_ip}" : "Set var.key_name to use SSH, or use ssm_connect_command."
}

output "run_script_commands" {
  description = "Commands to run the default NDVI script on the EC2 instance."
  value = {
    ndvi_only  = "source ~/geo_env/bin/activate && python ~/ndvi_gdl.py"
    with_image = "source ~/geo_env/bin/activate && python ~/ndvi_gdl.py --save-image"
  }
}
