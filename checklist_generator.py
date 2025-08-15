"""
Bug Bounty Checklist Generator Module

This module converts natural language tasks into structured bug bounty checklists
using transformer-based NLP models and URL extraction capabilities.
"""

import re
import logging
from typing import List, Dict, Tuple, Optional, Union
from dataclasses import dataclass
from urllib.parse import urlparse
import json

# Optional imports with fallbacks
try:
    from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    logging.warning("Transformers library not available. Using fallback NLP processing.")

try:
    import spacy
    HAS_SPACY = True
except ImportError:
    HAS_SPACY = False
    logging.warning("spaCy library not available. Using basic text processing.")

# Import voice processor if available
try:
    from voice_text_processor import VoiceTextProcessor
    HAS_VOICE_PROCESSOR = True
except ImportError:
    HAS_VOICE_PROCESSOR = False
    logging.info("voice_text_processor module not found. Direct text input only.")


@dataclass
class ChecklistItem:
    """Represents a single checklist item with metadata."""
    task: str
    priority: str  # 'high', 'medium', 'low'
    category: str  # e.g., 'reconnaissance', 'vulnerability_testing', 'exploitation'
    estimated_time: str  # e.g., '15 minutes', '1 hour'
    tools: List[str]  # Recommended tools for this task
    completed: bool = False


@dataclass
class BugBountyChecklist:
    """Complete bug bounty checklist with metadata."""
    title: str
    target_info: Dict[str, str]
    items: List[ChecklistItem]
    urls: List[str]
    created_timestamp: str
    total_estimated_time: str


class ChecklistGenerator:
    """Main class for generating bug bounty checklists from natural language."""
    
    def __init__(self, use_advanced_nlp: bool = True):
        """
        Initialize the checklist generator.
        
        Args:
            use_advanced_nlp: Whether to use transformer models (requires transformers library)
        """
        self.use_advanced_nlp = use_advanced_nlp and HAS_TRANSFORMERS
        self.voice_processor = None
        
        # Initialize NLP components
        self._init_nlp_components()
        
        # Initialize voice processor if available
        if HAS_VOICE_PROCESSOR:
            try:
                self.voice_processor = VoiceTextProcessor()
            except Exception as e:
                logging.warning(f"Could not initialize voice processor: {e}")
        
        # Bug bounty task templates
        self.task_templates = self._load_task_templates()
        
        # URL extraction patterns
        self.url_patterns = [
            r'https?://[^\s<>"{}|\\^`\[\]]+',
            r'www\.[^\s<>"{}|\\^`\[\]]+',
            r'[a-zA-Z0-9]([a-zA-Z0-9\-]{0,61}[a-zA-Z0-9])?(\.[a-zA-Z0-9]([a-zA-Z0-9\-]{0,61}[a-zA-Z0-9])?)*\.[a-zA-Z]{2,}'
        ]
    
    def _init_nlp_components(self):
        """Initialize NLP processing components."""
        self.nlp_pipeline = None
        self.spacy_nlp = None
        
        if self.use_advanced_nlp:
            try:
                # Initialize transformer pipeline for text classification
                self.nlp_pipeline = pipeline(
                    "text-classification",
                    model="distilbert-base-uncased-finetuned-sst-2-english",
                    return_all_scores=True
                )
                logging.info("Transformer pipeline initialized successfully.")
            except Exception as e:
                logging.error(f"Failed to initialize transformer pipeline: {e}")
                self.use_advanced_nlp = False
        
        if HAS_SPACY:
            try:
                self.spacy_nlp = spacy.load("en_core_web_sm")
                logging.info("spaCy model loaded successfully.")
            except Exception as e:
                logging.warning(f"Failed to load spaCy model: {e}")
    
    def _load_task_templates(self) -> Dict[str, List[Dict]]:
        """Load predefined bug bounty task templates."""
        return {
            "reconnaissance": [
                {
                    "task": "Perform subdomain enumeration using tools like subfinder, assetfinder",
                    "priority": "high",
                    "category": "reconnaissance",
                    "estimated_time": "30 minutes",
                    "tools": ["subfinder", "assetfinder", "amass"]
                },
                {
                    "task": "Run port scanning with nmap on discovered subdomains",
                    "priority": "high",
                    "category": "reconnaissance",
                    "estimated_time": "45 minutes",
                    "tools": ["nmap", "masscan"]
                },
                {
                    "task": "Perform directory and file enumeration",
                    "priority": "medium",
                    "category": "reconnaissance",
                    "estimated_time": "20 minutes",
                    "tools": ["gobuster", "dirbuster", "ffuf"]
                }
            ],
            "vulnerability_testing": [
                {
                    "task": "Test for SQL injection vulnerabilities",
                    "priority": "high",
                    "category": "vulnerability_testing",
                    "estimated_time": "1 hour",
                    "tools": ["sqlmap", "burp suite", "manual testing"]
                },
                {
                    "task": "Check for Cross-Site Scripting (XSS) vulnerabilities",
                    "priority": "high",
                    "category": "vulnerability_testing",
                    "estimated_time": "45 minutes",
                    "tools": ["XSSHunter", "burp suite", "manual testing"]
                },
                {
                    "task": "Test for CSRF vulnerabilities",
                    "priority": "medium",
                    "category": "vulnerability_testing",
                    "estimated_time": "30 minutes",
                    "tools": ["burp suite", "manual testing"]
                }
            ],
            "exploitation": [
                {
                    "task": "Develop proof of concept for identified vulnerabilities",
                    "priority": "high",
                    "category": "exploitation",
                    "estimated_time": "2 hours",
                    "tools": ["custom scripts", "metasploit", "manual exploitation"]
                },
                {
                    "task": "Document impact and create detailed vulnerability report",
                    "priority": "high",
                    "category": "exploitation",
                    "estimated_time": "1 hour",
                    "tools": ["documentation tools", "screenshot tools"]
                }
            ]
        }
    
    def extract_urls(self, text: str) -> List[str]:
        """
        Extract URLs from the input text.
        
        Args:
            text: Input text to extract URLs from
            
        Returns:
            List of extracted URLs
        """
        urls = []
        
        for pattern in self.url_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            urls.extend(matches)
        
        # Clean and validate URLs
        cleaned_urls = []
        for url in urls:
            # Add protocol if missing
            if not url.startswith(('http://', 'https://')):
                if url.startswith('www.'):
                    url = 'https://' + url
                else:
                    # Basic domain validation
                    if '.' in url and len(url.split('.')) >= 2:
                        url = 'https://' + url
            
            # Validate URL format
            try:
                parsed = urlparse(url)
                if parsed.scheme and parsed.netloc:
                    cleaned_urls.append(url)
            except Exception:
                continue
        
        return list(set(cleaned_urls))  # Remove duplicates
    
    def analyze_intent(self, text: str) -> Dict[str, float]:
        """
        Analyze the intent of the input text using NLP.
        
        Args:
            text: Input text to analyze
            
        Returns:
            Dictionary with intent scores
        """
        intents = {
            "reconnaissance": 0.0,
            "vulnerability_testing": 0.0,
            "exploitation": 0.0,
            "general": 0.0
        }
        
        # Keywords for different bug bounty phases
        keywords = {
            "reconnaissance": ["subdomain", "enumerate", "scan", "discover", "recon", "information gathering"],
            "vulnerability_testing": ["test", "vulnerability", "injection", "xss", "csrf", "security flaw"],
            "exploitation": ["exploit", "proof of concept", "poc", "payload", "attack", "penetrate"],
        }
        
        text_lower = text.lower()
        
        # Simple keyword-based intent analysis
        for intent, words in keywords.items():
            score = sum(1 for word in words if word in text_lower)
            intents[intent] = score / len(words)  # Normalize score
        
        # Use transformer model if available
        if self.use_advanced_nlp and self.nlp_pipeline:
            try:
                # This is a simplified example - you'd want to fine-tune on bug bounty data
                results = self.nlp_pipeline(text)
                # Adjust scores based on sentiment/classification
                positive_score = next((r['score'] for r in results[0] if r['label'] == 'POSITIVE'), 0)
                intents['general'] = positive_score
            except Exception as e:
                logging.error(f"Error in NLP pipeline: {e}")
        
        return intents
    
    def generate_tasks_from_intent(self, intents: Dict[str, float], target_urls: List[str]) -> List[ChecklistItem]:
        """
        Generate specific tasks based on detected intents.
        
        Args:
            intents: Dictionary of intent scores
            target_urls: List of target URLs
            
        Returns:
            List of ChecklistItem objects
        """
        tasks = []
        
        # Get the primary intent
        primary_intent = max(intents.items(), key=lambda x: x[1])
        
        if primary_intent[1] > 0:  # If we have a clear intent
            intent_name = primary_intent[0]
            if intent_name in self.task_templates:
                for template in self.task_templates[intent_name]:
                    task = ChecklistItem(
                        task=template["task"],
                        priority=template["priority"],
                        category=template["category"],
                        estimated_time=template["estimated_time"],
                        tools=template["tools"].copy()
                    )
                    
                    # Customize task for specific URLs if available
                    if target_urls and "{target}" not in task.task:
                        task.task += f" on target: {target_urls[0]}"
                    
                    tasks.append(task)
        
        # If no clear intent or general intent, provide comprehensive checklist
        if not tasks or primary_intent[0] == "general":
            # Add basic reconnaissance tasks
            for template in self.task_templates["reconnaissance"][:2]:  # First 2 recon tasks
                tasks.append(ChecklistItem(
                    task=template["task"],
                    priority=template["priority"],
                    category=template["category"],
                    estimated_time=template["estimated_time"],
                    tools=template["tools"].copy()
                ))
            
            # Add basic vulnerability testing
            for template in self.task_templates["vulnerability_testing"][:2]:  # First 2 vuln tasks
                tasks.append(ChecklistItem(
                    task=template["task"],
                    priority=template["priority"],
                    category=template["category"],
                    estimated_time=template["estimated_time"],
                    tools=template["tools"].copy()
                ))
        
        return tasks
    
    def process_text_input(self, text: str) -> BugBountyChecklist:
        """
        Process direct text input and generate checklist.
        
        Args:
            text: Natural language task description
            
        Returns:
            BugBountyChecklist object
        """
        # Extract URLs
        urls = self.extract_urls(text)
        
        # Analyze intent
        intents = self.analyze_intent(text)
        
        # Generate tasks
        tasks = self.generate_tasks_from_intent(intents, urls)
        
        # Calculate total estimated time
        total_time = self._calculate_total_time(tasks)
        
        # Create target info
        target_info = {
            "primary_target": urls[0] if urls else "Not specified",
            "additional_targets": ", ".join(urls[1:]) if len(urls) > 1 else "None"
        }
        
        # Create checklist
        checklist = BugBountyChecklist(
            title=f"Bug Bounty Checklist - {target_info['primary_target']}",
            target_info=target_info,
            items=tasks,
            urls=urls,
            created_timestamp=self._get_timestamp(),
            total_estimated_time=total_time
        )
        
        return checklist
    
    def process_voice_input(self, voice_data) -> Optional[BugBountyChecklist]:
        """
        Process voice input through voice_text_processor and generate checklist.
        
        Args:
            voice_data: Voice data to process
            
        Returns:
            BugBountyChecklist object or None if processing fails
        """
        if not self.voice_processor:
            logging.error("Voice processor not available")
            return None
        
        try:
            # Process voice to text
            text = self.voice_processor.process_voice_to_text(voice_data)
            
            if not text:
                logging.error("Failed to convert voice to text")
                return None
            
            # Process the extracted text
            return self.process_text_input(text)
            
        except Exception as e:
            logging.error(f"Error processing voice input: {e}")
            return None
    
    def generate_checklist(self, input_data: Union[str, object], input_type: str = "text") -> Optional[BugBountyChecklist]:
        """
        Main method to generate checklist from various input types.
        
        Args:
            input_data: Either text string or voice data object
            input_type: "text" or "voice"
            
        Returns:
            BugBountyChecklist object or None if processing fails
        """
        try:
            if input_type == "text":
                return self.process_text_input(input_data)
            elif input_type == "voice":
                return self.process_voice_input(input_data)
            else:
                logging.error(f"Unsupported input type: {input_type}")
                return None
        except Exception as e:
            logging.error(f"Error generating checklist: {e}")
            return None
    
    def export_for_connector(self, checklist: BugBountyChecklist) -> Dict:
        """
        Export checklist in format suitable for connector.py.
        
        Args:
            checklist: BugBountyChecklist object to export
            
        Returns:
            Dictionary with checklist data and URLs
        """
        return {
            "checklist": {
                "title": checklist.title,
                "target_info": checklist.target_info,
                "items": [
                    {
                        "task": item.task,
                        "priority": item.priority,
                        "category": item.category,
                        "estimated_time": item.estimated_time,
                        "tools": item.tools,
                        "completed": item.completed
                    }
                    for item in checklist.items
                ],
                "created_timestamp": checklist.created_timestamp,
                "total_estimated_time": checklist.total_estimated_time
            },
            "urls": checklist.urls,
            "metadata": {
                "total_tasks": len(checklist.items),
                "high_priority_tasks": len([item for item in checklist.items if item.priority == "high"]),
                "categories": list(set(item.category for item in checklist.items))
            }
        }
    
    def _calculate_total_time(self, tasks: List[ChecklistItem]) -> str:
        """Calculate total estimated time for all tasks."""
        total_minutes = 0
        
        for task in tasks:
            time_str = task.estimated_time.lower()
            if "minute" in time_str:
                minutes = int(re.findall(r'\d+', time_str)[0])
                total_minutes += minutes
            elif "hour" in time_str:
                hours = float(re.findall(r'[\d.]+', time_str)[0])
                total_minutes += int(hours * 60)
        
        if total_minutes < 60:
            return f"{total_minutes} minutes"
        else:
            hours = total_minutes // 60
            minutes = total_minutes % 60
            if minutes == 0:
                return f"{hours} hour{'s' if hours != 1 else ''}"
            else:
                return f"{hours} hour{'s' if hours != 1 else ''} {minutes} minutes"
    
    def _get_timestamp(self) -> str:
        """Get current timestamp."""
        from datetime import datetime
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# Example usage and testing
if __name__ == "__main__":
    # Initialize generator
    generator = ChecklistGenerator()
    
    # Example 1: Text input with URL
    test_text = "I want to test example.com for vulnerabilities, focusing on SQL injection and XSS"
    checklist = generator.generate_checklist(test_text, "text")
    
    if checklist:
        print("Generated Checklist:")
        print(f"Title: {checklist.title}")
        print(f"URLs found: {checklist.urls}")
        print("\nTasks:")
        for i, item in enumerate(checklist.items, 1):
            print(f"{i}. {item.task} [{item.priority} priority, {item.estimated_time}]")
        
        # Export for connector
        export_data = generator.export_for_connector(checklist)
        print(f"\nExport data ready for connector.py: {len(export_data)} keys")
    
    # Example 2: General reconnaissance request
    test_text2 = "Perform reconnaissance on target https://testsite.com and www.example.org"
    checklist2 = generator.generate_checklist(test_text2, "text")
    
    if checklist2:
        print(f"\nSecond checklist created with {len(checklist2.items)} tasks and {len(checklist2.urls)} URLs")