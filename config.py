"""
Configuration module for AI-powered Bug Bounty Assistant
Contains all global settings, constants, and configuration parameters.
"""

import os
from pathlib import Path

# =============================================================================
# BASE DIRECTORIES AND FILE PATHS
# =============================================================================

# Base directory for the application
BASE_DIR = Path(__file__).parent.absolute()

# Data storage directories
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"
LOGS_DIR = BASE_DIR / "logs"
CACHE_DIR = BASE_DIR / "cache"
TEMP_DIR = BASE_DIR / "temp"

# Collected data file paths
VULNERABILITY_DATA_PATH = DATA_DIR / "vulnerabilities.json"
EXPLOIT_DATA_PATH = DATA_DIR / "exploits.json"
CVE_DATA_PATH = DATA_DIR / "cve_data.json"
THREAT_INTEL_PATH = DATA_DIR / "threat_intelligence.json"
PAYLOADS_DATA_PATH = DATA_DIR / "payloads.json"
WORDLISTS_DIR = DATA_DIR / "wordlists"
SIGNATURES_PATH = DATA_DIR / "signatures.json"

# Model storage paths
MODEL_CHECKPOINT_PATH = MODELS_DIR / "bug_bounty_model.pt"
TOKENIZER_PATH = MODELS_DIR / "tokenizer"
FINE_TUNED_MODEL_PATH = MODELS_DIR / "fine_tuned_model"
EMBEDDING_CACHE_PATH = CACHE_DIR / "embeddings.pkl"

# Report output paths
HTML_REPORT_PATH = REPORTS_DIR / "bug_bounty_report.html"
PDF_REPORT_PATH = REPORTS_DIR / "bug_bounty_report.pdf"
JSON_REPORT_PATH = REPORTS_DIR / "bug_bounty_report.json"
CSV_REPORT_PATH = REPORTS_DIR / "vulnerability_summary.csv"

# Log file paths
MAIN_LOG_PATH = LOGS_DIR / "bug_bounty_assistant.log"
ERROR_LOG_PATH = LOGS_DIR / "errors.log"
SCAN_LOG_PATH = LOGS_DIR / "scans.log"
API_LOG_PATH = LOGS_DIR / "api_requests.log"

# =============================================================================
# MODEL PARAMETERS AND CONFIGURATION
# =============================================================================

# Training parameters
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
NUM_EPOCHS = 10
MAX_SEQUENCE_LENGTH = 512
GRADIENT_ACCUMULATION_STEPS = 4
WARMUP_STEPS = 100
WEIGHT_DECAY = 0.01

# Model configuration
MODEL_NAME = "microsoft/codebert-base"
TOKENIZER_NAME = "microsoft/codebert-base"
EMBEDDING_DIM = 768
HIDDEN_DIM = 256
NUM_CLASSES = 10  # Number of vulnerability categories
DROPOUT_RATE = 0.1

# Fine-tuning parameters
FINE_TUNE_LAYERS = 3
FREEZE_EMBEDDINGS = False
USE_SCHEDULER = True
SCHEDULER_TYPE = "linear"

# Inference parameters
CONFIDENCE_THRESHOLD = 0.7
TOP_K_PREDICTIONS = 5
MAX_GENERATED_LENGTH = 200
TEMPERATURE = 0.8
NUCLEUS_SAMPLING_P = 0.9

# =============================================================================
# APPLICATION SETTINGS
# =============================================================================

# Default operation mode
DEFAULT_MODE = "interactive"  # Options: "interactive", "batch", "api"

# Scan configuration
DEFAULT_SCAN_DEPTH = "medium"  # Options: "light", "medium", "deep"
MAX_CONCURRENT_SCANS = 5
SCAN_TIMEOUT = 3600  # seconds
RATE_LIMIT_DELAY = 1.0  # seconds between requests

# Analysis settings
ENABLE_STATIC_ANALYSIS = True
ENABLE_DYNAMIC_ANALYSIS = True
ENABLE_FUZZING = False  # Disabled by default for safety
ENABLE_THREAT_MODELING = True

# Output preferences
VERBOSE_OUTPUT = False
SAVE_INTERMEDIATE_RESULTS = True
AUTO_GENERATE_REPORTS = True
COMPRESS_OUTPUTS = False

# =============================================================================
# API KEYS AND EXTERNAL SERVICES
# =============================================================================

# OpenRouter API (for additional LLM capabilities)
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_MODEL = "anthropic/claude-3-sonnet"

# Hugging Face API (for model downloads and inference)
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY", "")
HUGGINGFACE_HUB_CACHE = os.getenv("HF_HOME", str(CACHE_DIR / "huggingface"))

# CVE Database APIs
NVD_API_KEY = os.getenv("NVD_API_KEY", "")
MITRE_API_KEY = os.getenv("MITRE_API_KEY", "")

# Threat Intelligence APIs
VIRUSTOTAL_API_KEY = os.getenv("VIRUSTOTAL_API_KEY", "")
SHODAN_API_KEY = os.getenv("SHODAN_API_KEY", "")
CENSYS_API_ID = os.getenv("CENSYS_API_ID", "")
CENSYS_API_SECRET = os.getenv("CENSYS_API_SECRET", "")

# =============================================================================
# VULNERABILITY CATEGORIES AND CLASSIFICATIONS
# =============================================================================

# OWASP Top 10 categories
OWASP_TOP_10 = [
    "A01:2021 – Broken Access Control",
    "A02:2021 – Cryptographic Failures", 
    "A03:2021 – Injection",
    "A04:2021 – Insecure Design",
    "A05:2021 – Security Misconfiguration",
    "A06:2021 – Vulnerable and Outdated Components",
    "A07:2021 – Identification and Authentication Failures",
    "A08:2021 – Software and Data Integrity Failures",
    "A09:2021 – Security Logging and Monitoring Failures",
    "A10:2021 – Server-Side Request Forgery"
]

# Vulnerability severity levels
SEVERITY_LEVELS = ["Critical", "High", "Medium", "Low", "Informational"]

# Common vulnerability types
VULNERABILITY_TYPES = [
    "SQL Injection",
    "Cross-Site Scripting (XSS)",
    "Cross-Site Request Forgery (CSRF)",
    "Remote Code Execution (RCE)",
    "Local File Inclusion (LFI)",
    "Remote File Inclusion (RFI)",
    "Directory Traversal",
    "Authentication Bypass",
    "Privilege Escalation",
    "Information Disclosure",
    "Denial of Service (DoS)",
    "Buffer Overflow",
    "Race Condition",
    "Business Logic Flaw"
]

# =============================================================================
# SCANNING AND ANALYSIS SETTINGS
# =============================================================================

# Port scanning configuration
DEFAULT_PORTS = [21, 22, 23, 25, 53, 80, 110, 111, 135, 139, 143, 443, 993, 995]
COMMON_PORTS = list(range(1, 1025))  # Well-known ports
ALL_PORTS = list(range(1, 65536))    # All possible ports

# Web application testing
DEFAULT_USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36"
]

# Request configurations
REQUEST_TIMEOUT = 30
MAX_REDIRECTS = 5
VERIFY_SSL = False  # For testing environments
PROXY_SETTINGS = None

# =============================================================================
# REPORTING AND OUTPUT SETTINGS
# =============================================================================

# Report template settings
REPORT_TITLE = "AI-Powered Bug Bounty Assessment Report"
REPORT_AUTHOR = "Bug Bounty Assistant"
INCLUDE_REMEDIATION = True
INCLUDE_CVSS_SCORES = True
INCLUDE_PROOF_OF_CONCEPT = True

# Output formats
SUPPORTED_OUTPUT_FORMATS = ["html", "pdf", "json", "csv", "xml"]
DEFAULT_OUTPUT_FORMAT = "html"

# Chart and visualization settings
ENABLE_CHARTS = True
CHART_THEME = "dark"
CHART_WIDTH = 800
CHART_HEIGHT = 600

# =============================================================================
# SECURITY AND SAFETY SETTINGS
# =============================================================================

# Safety constraints
ENABLE_SAFE_MODE = True
BLOCK_DESTRUCTIVE_PAYLOADS = True
REQUIRE_EXPLICIT_CONFIRMATION = True
MAX_PAYLOAD_SIZE = 8192  # bytes

# Rate limiting
API_RATE_LIMIT = 100  # requests per minute
SCAN_RATE_LIMIT = 10   # scans per minute

# Logging levels
LOG_LEVEL = "INFO"  # Options: "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"
ENABLE_SENSITIVE_DATA_LOGGING = False

# =============================================================================
# PERFORMANCE AND RESOURCE SETTINGS
# =============================================================================

# Memory and processing limits
MAX_MEMORY_USAGE = 4 * 1024 * 1024 * 1024  # 4GB in bytes
MAX_CPU_CORES = os.cpu_count()
ENABLE_GPU = True
GPU_MEMORY_FRACTION = 0.8

# Cache settings
ENABLE_CACHING = True
CACHE_TTL = 3600  # seconds
MAX_CACHE_SIZE = 1000  # number of items

# Threading and concurrency
MAX_WORKER_THREADS = 10
ENABLE_ASYNC_PROCESSING = True

# =============================================================================
# VERSION AND METADATA
# =============================================================================

VERSION = "1.0.0"
BUILD_DATE = "2024-08-15"
AUTHOR = "AI Bug Bounty Assistant"
LICENSE = "MIT"
DESCRIPTION = "AI-powered bug bounty hunting and vulnerability assessment tool"

# Feature flags
EXPERIMENTAL_FEATURES = False
DEBUG_MODE = False
DEVELOPMENT_MODE = False