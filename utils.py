"""
utils.py - Utility functions for AI-powered bug bounty assistant

This module provides reusable helper functions for logging, file operations,
and language detection across all modules of the bug bounty assistant.
"""

import os
import json
import csv
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from pathlib import Path
import re

# Import configuration
try:
    import config
except ImportError:
    # Fallback configuration if config.py is not available
    class Config:
        LOG_LEVEL = "INFO"
        LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        DATA_DIR = "data"
        LOGS_DIR = "logs"
    config = Config()


class Logger:
    """Enhanced logging utility with timestamp support."""
    
    _loggers = {}
    
    @classmethod
    def get_logger(cls, name: str = "bug_bounty_assistant") -> logging.Logger:
        """Get or create a logger instance."""
        if name not in cls._loggers:
            logger = logging.getLogger(name)
            
            # Avoid adding handlers multiple times
            if not logger.handlers:
                # Create logs directory if it doesn't exist
                log_dir = getattr(config, 'LOGS_DIR', 'logs')
                os.makedirs(log_dir, exist_ok=True)
                
                # Set up file handler
                log_file = os.path.join(log_dir, f"{name}.log")
                file_handler = logging.FileHandler(log_file, encoding='utf-8')
                
                # Set up console handler
                console_handler = logging.StreamHandler()
                
                # Set formatting
                formatter = logging.Formatter(
                    getattr(config, 'LOG_FORMAT', 
                           "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
                )
                file_handler.setFormatter(formatter)
                console_handler.setFormatter(formatter)
                
                # Add handlers
                logger.addHandler(file_handler)
                logger.addHandler(console_handler)
                
                # Set level
                log_level = getattr(config, 'LOG_LEVEL', 'INFO')
                logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))
            
            cls._loggers[name] = logger
        
        return cls._loggers[name]
    
    @classmethod
    def log_with_timestamp(cls, message: str, level: str = "INFO", 
                          logger_name: str = "bug_bounty_assistant") -> None:
        """Log a message with timestamp."""
        logger = cls.get_logger(logger_name)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        formatted_message = f"[{timestamp}] {message}"
        
        log_method = getattr(logger, level.lower(), logger.info)
        log_method(formatted_message)


class FileHandler:
    """Utility class for file operations."""
    
    @staticmethod
    def ensure_directory(file_path: str) -> None:
        """Ensure the directory for a file path exists."""
        directory = os.path.dirname(file_path)
        if directory:
            os.makedirs(directory, exist_ok=True)
    
    @staticmethod
    def read_json(file_path: str, default: Any = None) -> Any:
        """
        Read JSON data from a file.
        
        Args:
            file_path: Path to the JSON file
            default: Default value to return if file doesn't exist or is invalid
            
        Returns:
            Parsed JSON data or default value
        """
        try:
            if not os.path.exists(file_path):
                Logger.log_with_timestamp(f"JSON file not found: {file_path}", "WARNING")
                return default
                
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                Logger.log_with_timestamp(f"Successfully read JSON from: {file_path}")
                return data
                
        except json.JSONDecodeError as e:
            Logger.log_with_timestamp(f"Invalid JSON in file {file_path}: {e}", "ERROR")
            return default
        except Exception as e:
            Logger.log_with_timestamp(f"Error reading JSON file {file_path}: {e}", "ERROR")
            return default
    
    @staticmethod
    def write_json(file_path: str, data: Any, indent: int = 2) -> bool:
        """
        Write data to a JSON file.
        
        Args:
            file_path: Path to the JSON file
            data: Data to write
            indent: JSON indentation level
            
        Returns:
            True if successful, False otherwise
        """
        try:
            FileHandler.ensure_directory(file_path)
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=indent, ensure_ascii=False)
                
            Logger.log_with_timestamp(f"Successfully wrote JSON to: {file_path}")
            return True
            
        except Exception as e:
            Logger.log_with_timestamp(f"Error writing JSON file {file_path}: {e}", "ERROR")
            return False
    
    @staticmethod
    def read_csv(file_path: str, delimiter: str = ',') -> List[Dict[str, str]]:
        """
        Read CSV data from a file.
        
        Args:
            file_path: Path to the CSV file
            delimiter: CSV delimiter
            
        Returns:
            List of dictionaries representing CSV rows
        """
        try:
            if not os.path.exists(file_path):
                Logger.log_with_timestamp(f"CSV file not found: {file_path}", "WARNING")
                return []
            
            data = []
            with open(file_path, 'r', encoding='utf-8', newline='') as f:
                reader = csv.DictReader(f, delimiter=delimiter)
                data = list(reader)
                
            Logger.log_with_timestamp(f"Successfully read {len(data)} rows from CSV: {file_path}")
            return data
            
        except Exception as e:
            Logger.log_with_timestamp(f"Error reading CSV file {file_path}: {e}", "ERROR")
            return []
    
    @staticmethod
    def write_csv(file_path: str, data: List[Dict[str, Any]], 
                  fieldnames: Optional[List[str]] = None, delimiter: str = ',') -> bool:
        """
        Write data to a CSV file.
        
        Args:
            file_path: Path to the CSV file
            data: List of dictionaries to write
            fieldnames: CSV column headers (auto-detected if None)
            delimiter: CSV delimiter
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if not data:
                Logger.log_with_timestamp("No data to write to CSV", "WARNING")
                return False
                
            FileHandler.ensure_directory(file_path)
            
            if fieldnames is None:
                fieldnames = list(data[0].keys()) if data else []
            
            with open(file_path, 'w', encoding='utf-8', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=delimiter)
                writer.writeheader()
                writer.writerows(data)
                
            Logger.log_with_timestamp(f"Successfully wrote {len(data)} rows to CSV: {file_path}")
            return True
            
        except Exception as e:
            Logger.log_with_timestamp(f"Error writing CSV file {file_path}: {e}", "ERROR")
            return False
    
    @staticmethod
    def read_txt(file_path: str, encoding: str = 'utf-8') -> Optional[str]:
        """
        Read text from a file.
        
        Args:
            file_path: Path to the text file
            encoding: File encoding
            
        Returns:
            File contents as string or None if error
        """
        try:
            if not os.path.exists(file_path):
                Logger.log_with_timestamp(f"Text file not found: {file_path}", "WARNING")
                return None
                
            with open(file_path, 'r', encoding=encoding) as f:
                content = f.read()
                
            Logger.log_with_timestamp(f"Successfully read text file: {file_path}")
            return content
            
        except Exception as e:
            Logger.log_with_timestamp(f"Error reading text file {file_path}: {e}", "ERROR")
            return None
    
    @staticmethod
    def write_txt(file_path: str, content: str, encoding: str = 'utf-8', 
                  append: bool = False) -> bool:
        """
        Write text to a file.
        
        Args:
            file_path: Path to the text file
            content: Content to write
            encoding: File encoding
            append: Whether to append to existing file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            FileHandler.ensure_directory(file_path)
            
            mode = 'a' if append else 'w'
            with open(file_path, mode, encoding=encoding) as f:
                f.write(content)
                
            action = "appended to" if append else "wrote"
            Logger.log_with_timestamp(f"Successfully {action} text file: {file_path}")
            return True
            
        except Exception as e:
            Logger.log_with_timestamp(f"Error writing text file {file_path}: {e}", "ERROR")
            return False


class LanguageDetector:
    """Simple language detection utility for text input."""
    
    # Common programming language keywords and patterns
    LANGUAGE_PATTERNS = {
        'python': [
            r'\bdef\s+\w+\s*\(', r'\bclass\s+\w+', r'\bimport\s+\w+',
            r'\bfrom\s+\w+\s+import', r'\bif\s+__name__\s*==\s*["\']__main__["\']',
            r'\bprint\s*\(', r'\belif\b', r'\btry\s*:', r'\bexcept\b'
        ],
        'javascript': [
            r'\bfunction\s+\w+\s*\(', r'\bvar\s+\w+', r'\blet\s+\w+', r'\bconst\s+\w+',
            r'\bconsole\.log\s*\(', r'\b=>\s*{', r'\brequire\s*\(', r'\bmodule\.exports'
        ],
        'java': [
            r'\bpublic\s+class\s+\w+', r'\bpublic\s+static\s+void\s+main',
            r'\bSystem\.out\.print', r'\bprivate\s+\w+', r'\bprotected\s+\w+'
        ],
        'c': [
            r'#include\s*<', r'\bint\s+main\s*\(', r'\bprintf\s*\(',
            r'\bmalloc\s*\(', r'\bfree\s*\('
        ],
        'cpp': [
            r'#include\s*<', r'\bstd::', r'\bcout\s*<<', r'\bcin\s*>>',
            r'\bnamespace\s+\w+', r'\busing\s+namespace'
        ],
        'php': [
            r'<\?php', r'\$\w+', r'\becho\s+', r'\bfunction\s+\w+\s*\(',
            r'\bclass\s+\w+', r'\b->\w+'
        ],
        'ruby': [
            r'\bdef\s+\w+', r'\bclass\s+\w+', r'\bend\b', r'\brequire\s+',
            r'\bputs\s+', r'\b@\w+', r'\b\|\w+\|'
        ],
        'go': [
            r'\bpackage\s+\w+', r'\bfunc\s+\w+\s*\(', r'\bimport\s+',
            r'\bfmt\.Print', r'\bvar\s+\w+', r'\btype\s+\w+'
        ],
        'rust': [
            r'\bfn\s+\w+\s*\(', r'\blet\s+\w+', r'\bmut\s+\w+',
            r'\bprintln!\s*\(', r'\buse\s+\w+', r'\bstruct\s+\w+'
        ]
    }
    
    @classmethod
    def detect_programming_language(cls, text: str) -> Optional[str]:
        """
        Detect programming language in text.
        
        Args:
            text: Text to analyze
            
        Returns:
            Detected language name or None if not detected
        """
        if not text or not isinstance(text, str):
            return None
            
        # Count matches for each language
        language_scores = {}
        
        for language, patterns in cls.LANGUAGE_PATTERNS.items():
            score = 0
            for pattern in patterns:
                matches = len(re.findall(pattern, text, re.IGNORECASE | re.MULTILINE))
                score += matches
                
            if score > 0:
                language_scores[language] = score
        
        if not language_scores:
            return None
            
        # Return language with highest score
        detected_language = max(language_scores, key=language_scores.get)
        
        Logger.log_with_timestamp(
            f"Detected programming language: {detected_language} "
            f"(score: {language_scores[detected_language]})"
        )
        
        return detected_language
    
    @classmethod
    def is_code_snippet(cls, text: str) -> bool:
        """
        Check if text appears to be a code snippet.
        
        Args:
            text: Text to analyze
            
        Returns:
            True if text appears to be code
        """
        if not text or not isinstance(text, str):
            return False
            
        # Check for common code indicators
        code_indicators = [
            r'[{}();]',  # Common code punctuation
            r'\b(function|class|def|var|let|const)\b',  # Keywords
            r'[=!<>]=',  # Comparison operators
            r'/\*.*?\*/',  # Block comments
            r'//.*$',  # Line comments
            r'#.*$',  # Hash comments
            r'\b\w+\.\w+\(',  # Method calls
        ]
        
        matches = 0
        for pattern in code_indicators:
            if re.search(pattern, text, re.MULTILINE):
                matches += 1
                
        return matches >= 2


# Utility functions for common operations
def get_timestamp(format_str: str = "%Y-%m-%d %H:%M:%S") -> str:
    """Get formatted timestamp."""
    return datetime.now().strftime(format_str)


def sanitize_filename(filename: str) -> str:
    """Sanitize a filename for safe file system usage."""
    # Remove or replace invalid characters
    invalid_chars = '<>:"/\\|?*'
    for char in invalid_chars:
        filename = filename.replace(char, '_')
    
    # Remove leading/trailing whitespace and dots
    filename = filename.strip(' .')
    
    # Ensure filename is not empty
    if not filename:
        filename = f"untitled_{get_timestamp('%Y%m%d_%H%M%S')}"
    
    return filename


def validate_file_path(file_path: str, check_extension: Optional[str] = None) -> bool:
    """
    Validate file path format and optionally check extension.
    
    Args:
        file_path: Path to validate
        check_extension: Expected file extension (e.g., '.json', '.csv')
        
    Returns:
        True if valid, False otherwise
    """
    if not file_path or not isinstance(file_path, str):
        return False
        
    try:
        path_obj = Path(file_path)
        
        # Check if path is valid
        if not path_obj.name:
            return False
            
        # Check extension if specified
        if check_extension and not file_path.lower().endswith(check_extension.lower()):
            return False
            
        return True
        
    except Exception:
        return False


# Initialize logger for the module
logger = Logger.get_logger("utils")
logger.info("Utils module initialized successfully")