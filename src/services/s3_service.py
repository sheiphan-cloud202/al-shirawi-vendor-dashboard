"""S3 service for uploading and managing files in AWS S3"""
import boto3
from pathlib import Path
from typing import Optional
from botocore.exceptions import ClientError, NoCredentialsError
import os

from src.utils.constants import AWS_PROFILE, AWS_REGION


class S3Service:
    """Service for uploading and managing files in AWS S3"""
    
    def __init__(self, bucket_name: Optional[str] = None):
        """
        Initialize S3 service
        
        Args:
            bucket_name: S3 bucket name. If not provided, uses default bucket from env
        """
        self.bucket_name = bucket_name or os.getenv("S3_BUCKET_NAME", "al-shirawi-orc-poc")
        self.region = AWS_REGION
        
        # Initialize boto3 session with profile
        try:
            self.session = boto3.Session(profile_name=AWS_PROFILE)
            self.s3_client = self.session.client('s3', region_name=self.region)
            print(f"✓ S3 service initialized with bucket: {self.bucket_name}")
        except Exception as e:
            print(f"⚠️  Warning: Could not initialize S3 client: {e}")
            self.s3_client = None
    
    def ensure_bucket_exists(self) -> bool:
        """
        Ensure the S3 bucket exists, create if it doesn't
        
        Returns:
            bool: True if bucket exists or was created successfully
        """
        if not self.s3_client:
            print("❌ S3 client not initialized")
            return False
        
        try:
            # Check if bucket exists
            self.s3_client.head_bucket(Bucket=self.bucket_name)
            print(f"✓ Bucket '{self.bucket_name}' exists")
            return True
        except ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code == '404':
                # Bucket doesn't exist, create it
                try:
                    print(f"📦 Creating bucket '{self.bucket_name}' in region {self.region}")
                    if self.region == 'us-east-1':
                        # us-east-1 doesn't need LocationConstraint
                        self.s3_client.create_bucket(Bucket=self.bucket_name)
                    else:
                        self.s3_client.create_bucket(
                            Bucket=self.bucket_name,
                            CreateBucketConfiguration={'LocationConstraint': self.region}
                        )
                    print(f"✓ Bucket '{self.bucket_name}' created successfully")
                    return True
                except Exception as create_error:
                    print(f"❌ Failed to create bucket: {create_error}")
                    return False
            else:
                print(f"❌ Error checking bucket: {e}")
                return False
    
    def upload_file(self, file_path: Path, s3_key: str) -> Optional[str]:
        """
        Upload a file to S3
        
        Args:
            file_path: Path to local file to upload
            s3_key: S3 object key (path in bucket)
            
        Returns:
            str: S3 URI of uploaded file, or None if upload failed
        """
        if not self.s3_client:
            print("❌ S3 client not initialized, cannot upload")
            return None
        
        if not file_path.exists():
            print(f"❌ File not found: {file_path}")
            return None
        
        # Ensure bucket exists
        if not self.ensure_bucket_exists():
            print("❌ Cannot upload - bucket not available")
            return None
        
        try:
            print(f"📤 Uploading {file_path.name} to s3://{self.bucket_name}/{s3_key}")
            self.s3_client.upload_file(
                str(file_path),
                self.bucket_name,
                s3_key,
                ExtraArgs={'ContentType': 'text/csv'}
            )
            s3_uri = f"s3://{self.bucket_name}/{s3_key}"
            print(f"✓ File uploaded successfully: {s3_uri}")
            return s3_uri
        except NoCredentialsError:
            print("❌ AWS credentials not found")
            return None
        except Exception as e:
            print(f"❌ Failed to upload file: {e}")
            return None
    
    def generate_presigned_url(self, s3_key: str, expiration: int = 3600) -> Optional[str]:
        """
        Generate a presigned URL for downloading a file from S3
        
        Args:
            s3_key: S3 object key
            expiration: URL expiration time in seconds (default: 1 hour)
            
        Returns:
            str: Presigned URL, or None if generation failed
        """
        if not self.s3_client:
            print("❌ S3 client not initialized")
            return None
        
        try:
            url = self.s3_client.generate_presigned_url(
                'get_object',
                Params={
                    'Bucket': self.bucket_name,
                    'Key': s3_key
                },
                ExpiresIn=expiration
            )
            print(f"✓ Generated presigned URL (expires in {expiration}s)")
            return url
        except Exception as e:
            print(f"❌ Failed to generate presigned URL: {e}")
            return None
    
    def upload_comparison_csv(self, session_id: str, csv_path: Path) -> Optional[dict]:
        """
        Upload comparison CSV to S3 with session-based organization
        
        Args:
            session_id: Session ID
            csv_path: Path to comparison CSV file
            
        Returns:
            dict: Upload details including S3 URI and presigned URL, or None if failed
        """
        # Organize by session ID in S3
        s3_key = f"sessions/{session_id}/comparisons/all_comparison.csv"
        
        s3_uri = self.upload_file(csv_path, s3_key)
        if not s3_uri:
            return None
        
        # Generate presigned URL for download
        download_url = self.generate_presigned_url(s3_key, expiration=86400)  # 24 hours
        
        return {
            "s3_uri": s3_uri,
            "s3_key": s3_key,
            "download_url": download_url,
            "bucket": self.bucket_name,
            "session_id": session_id,
            "file_name": csv_path.name
        }


def upload_comparison_to_s3(session_id: str, comparison_csv_path: Path) -> Optional[dict]:
    """
    Convenience function to upload comparison CSV to S3
    
    Args:
        session_id: Session ID
        comparison_csv_path: Path to all_comparison.csv file
        
    Returns:
        dict: Upload details, or None if failed
    """
    s3_service = S3Service()
    return s3_service.upload_comparison_csv(session_id, comparison_csv_path)

