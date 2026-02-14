"""
File Upload Handler for GUI interface.

This module provides file upload and hosting capabilities for the GUI,
allowing users to upload files that can be used as inputs for ML jobs.
"""

import os
import shutil
import hashlib
from pathlib import Path
from typing import Optional, Tuple
from datetime import datetime

from notebook_ml_orchestrator.core.logging_config import LoggerMixin


class FileUploadHandler(LoggerMixin):
    """Handles file uploads and provides URLs for uploaded files."""
    
    def __init__(self, upload_dir: str = "uploads"):
        """
        Initialize file upload handler.
        
        Args:
            upload_dir: Directory to store uploaded files
        """
        self.upload_dir = Path(upload_dir)
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.logger.info(f"FileUploadHandler initialized with upload_dir: {self.upload_dir}")
    
    def save_uploaded_file(
        self,
        file_path: str,
        original_filename: Optional[str] = None
    ) -> Tuple[bool, str, str]:
        """
        Save an uploaded file and return its path and URL.
        
        Args:
            file_path: Path to the temporary uploaded file
            original_filename: Original filename (optional)
            
        Returns:
            Tuple of (success, file_path, message)
        """
        try:
            if not file_path or not os.path.exists(file_path):
                return False, "", "No file provided or file not found"
            
            # Get original filename
            if not original_filename:
                original_filename = os.path.basename(file_path)
            
            # Generate unique filename using hash + timestamp
            file_hash = self._get_file_hash(file_path)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_ext = Path(original_filename).suffix
            unique_filename = f"{timestamp}_{file_hash[:8]}{file_ext}"
            
            # Save file
            dest_path = self.upload_dir / unique_filename
            shutil.copy2(file_path, dest_path)
            
            # Return absolute path
            abs_path = str(dest_path.absolute())
            
            self.logger.info(f"File uploaded successfully: {unique_filename}")
            
            return True, abs_path, f"File uploaded: {unique_filename}"
            
        except Exception as e:
            self.logger.error(f"Failed to save uploaded file: {e}")
            return False, "", f"Upload failed: {str(e)}"
    
    def _get_file_hash(self, file_path: str) -> str:
        """Calculate SHA256 hash of file."""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
    
    def get_file_url(self, file_path: str) -> str:
        """
        Get URL for an uploaded file.
        
        For local files, returns the file:// URL.
        In production, this would return an HTTP URL to a file server.
        
        Args:
            file_path: Path to the file
            
        Returns:
            URL to access the file
        """
        # For local development, return file:// URL
        # In production, you would upload to S3/GCS and return HTTP URL
        abs_path = Path(file_path).absolute()
        return f"file://{abs_path}"
    
    def list_uploaded_files(self) -> list:
        """List all uploaded files."""
        try:
            files = []
            for file_path in self.upload_dir.glob("*"):
                if file_path.is_file():
                    stat = file_path.stat()
                    files.append({
                        'filename': file_path.name,
                        'path': str(file_path.absolute()),
                        'size': stat.st_size,
                        'modified': datetime.fromtimestamp(stat.st_mtime).isoformat()
                    })
            return sorted(files, key=lambda x: x['modified'], reverse=True)
        except Exception as e:
            self.logger.error(f"Failed to list uploaded files: {e}")
            return []
    
    def delete_file(self, filename: str) -> Tuple[bool, str]:
        """
        Delete an uploaded file.
        
        Args:
            filename: Name of the file to delete
            
        Returns:
            Tuple of (success, message)
        """
        try:
            file_path = self.upload_dir / filename
            if file_path.exists() and file_path.is_file():
                file_path.unlink()
                self.logger.info(f"File deleted: {filename}")
                return True, f"File deleted: {filename}"
            else:
                return False, "File not found"
        except Exception as e:
            self.logger.error(f"Failed to delete file: {e}")
            return False, f"Delete failed: {str(e)}"
    
    def cleanup_old_files(self, days: int = 7) -> int:
        """
        Clean up files older than specified days.
        
        Args:
            days: Delete files older than this many days
            
        Returns:
            Number of files deleted
        """
        try:
            count = 0
            cutoff_time = datetime.now().timestamp() - (days * 24 * 60 * 60)
            
            for file_path in self.upload_dir.glob("*"):
                if file_path.is_file():
                    if file_path.stat().st_mtime < cutoff_time:
                        file_path.unlink()
                        count += 1
            
            if count > 0:
                self.logger.info(f"Cleaned up {count} old files")
            
            return count
        except Exception as e:
            self.logger.error(f"Failed to cleanup old files: {e}")
            return 0
