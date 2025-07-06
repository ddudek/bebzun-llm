import os
import sys
import math
from typing import Optional

def get_file_content(file_path: str) -> str:
    """
    Load a prompt template from a file
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        print(f"Error loading prompt template: {str(e)}")

def is_binary_file(file_path: str) -> bool:
    """
    Check if a file is binary by reading a small chunk and looking for null bytes
    """
    try:
        with open(file_path, 'rb') as f:
            chunk = f.read(1024)
            return b'\0' in chunk
    except Exception:
        return True

def get_file_content_safe(file_path: str, max_size: int = 230000) -> Optional[str]:
    """
    Get the content of a file safely, handling binary files and large files
    """
    try:
        # Check if file is binary
        if is_binary_file(file_path):
            return None
            
        # Check file size
        file_size = os.path.getsize(file_path)
        if file_size > max_size:
            return f"[File too large: {file_size} bytes]"
            
        # Read the file content
        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            return f.read()
    except Exception as e:
        return f"[Error reading file: {str(e)}]"

def format_file_size(size_in_bytes: int) -> str:
    """
    Format file size in a human-readable format (bytes, kB, MB, GB)
    
    Args:
        size_in_bytes: File size in bytes
        
    Returns:
        Formatted file size string
    """
    # Define size units and thresholds
    units = ['bytes', 'kB', 'MB', 'GB', 'TB']
    
    # Handle zero size
    if size_in_bytes == 0:
        return "0 bytes"
    
    # Calculate the appropriate unit
    unit_index = 1 #min(4, int(math.log(size_in_bytes, 1024)))
    size_value = size_in_bytes / (1024 ** unit_index)
    
    # Format with appropriate precision
    if unit_index == 0:  # bytes
        return f"{int(size_value)} {units[unit_index]}"
    else:
        return f"{size_value:.2f} {units[unit_index]}"