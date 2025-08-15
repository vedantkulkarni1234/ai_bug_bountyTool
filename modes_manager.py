"""
AI Modes Manager for Bug Bounty Workflow
Manages different levels of AI automation and assistance for security testing.
"""

from enum import Enum
from typing import Dict, List, Callable, Any, Optional
from dataclasses import dataclass
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AIMode(Enum):
    """Available AI modes for bug bounty workflow."""
    RECON_ONLY = "recon_only"
    AI_ASSISTED = "ai_assisted" 
    FULL_AUTOMATION = "full_automation"
    REPORT_ONLY = "report_only"


@dataclass
class ModeConfig:
    """Configuration for each AI mode."""
    name: str
    description: str
    automation_level: int  # 1-4 scale
    requires_human_approval: bool
    checklist_filter: List[str]  # Categories to process
    max_concurrent_tasks: int
    timeout_per_task: int  # seconds
    risk_threshold: str  # low, medium, high


class ModesManager:
    """Manages AI modes and their execution logic."""
    
    def __init__(self):
        """Initialize the modes manager with predefined configurations."""
        self.current_mode = AIMode.AI_ASSISTED
        self.mode_configs = self._initialize_mode_configs()
        self.mode_processors = self._initialize_mode_processors()
        
    def _initialize_mode_configs(self) -> Dict[AIMode, ModeConfig]:
        """Initialize configuration for each mode."""
        return {
            AIMode.RECON_ONLY: ModeConfig(
                name="Reconnaissance Only",
                description="AI performs only passive reconnaissance and information gathering",
                automation_level=1,
                requires_human_approval=True,
                checklist_filter=["reconnaissance", "information_gathering", "subdomain_enum"],
                max_concurrent_tasks=3,
                timeout_per_task=300,
                risk_threshold="low"
            ),
            
            AIMode.AI_ASSISTED: ModeConfig(
                name="AI-Assisted Testing",
                description="AI suggests tests and provides guidance with human oversight",
                automation_level=2,
                requires_human_approval=True,
                checklist_filter=["reconnaissance", "vulnerability_scanning", "manual_testing"],
                max_concurrent_tasks=5,
                timeout_per_task=600,
                risk_threshold="medium"
            ),
            
            AIMode.FULL_AUTOMATION: ModeConfig(
                name="Full Automation",
                description="AI performs comprehensive automated testing with minimal human intervention",
                automation_level=4,
                requires_human_approval=False,
                checklist_filter=["reconnaissance", "vulnerability_scanning", "automated_testing", "exploitation"],
                max_concurrent_tasks=10,
                timeout_per_task=1800,
                risk_threshold="high"
            ),
            
            AIMode.REPORT_ONLY: ModeConfig(
                name="Report Generation Only",
                description="AI only processes existing findings and generates reports",
                automation_level=1,
                requires_human_approval=True,
                checklist_filter=["report_generation", "documentation"],
                max_concurrent_tasks=2,
                timeout_per_task=120,
                risk_threshold="low"
            )
        }
    
    def _initialize_mode_processors(self) -> Dict[AIMode, Callable]:
        """Initialize processing functions for each mode."""
        return {
            AIMode.RECON_ONLY: self._process_recon_only,
            AIMode.AI_ASSISTED: self._process_ai_assisted,
            AIMode.FULL_AUTOMATION: self._process_full_automation,
            AIMode.REPORT_ONLY: self._process_report_only
        }
    
    def set_mode(self, mode: AIMode) -> bool:
        """
        Set the current AI mode.
        
        Args:
            mode: The AI mode to set
            
        Returns:
            bool: True if mode was set successfully, False otherwise
        """
        if mode not in self.mode_configs:
            logger.error(f"Invalid mode: {mode}")
            return False
            
        self.current_mode = mode
        logger.info(f"AI mode set to: {self.get_current_mode_config().name}")
        return True
    
    def get_current_mode(self) -> AIMode:
        """Get the current AI mode."""
        return self.current_mode
    
    def get_current_mode_config(self) -> ModeConfig:
        """Get the configuration for the current mode."""
        return self.mode_configs[self.current_mode]
    
    def get_all_modes(self) -> Dict[AIMode, ModeConfig]:
        """Get all available modes and their configurations."""
        return self.mode_configs.copy()
    
    def filter_checklist(self, checklist: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Filter checklist items based on current mode configuration.
        
        Args:
            checklist: List of checklist items
            
        Returns:
            Filtered checklist based on current mode
        """
        current_config = self.get_current_mode_config()
        allowed_categories = current_config.checklist_filter
        
        filtered_checklist = []
        for item in checklist:
            item_category = item.get('category', '').lower()
            if any(category in item_category for category in allowed_categories):
                # Add mode-specific metadata
                item['mode_config'] = {
                    'requires_approval': current_config.requires_human_approval,
                    'timeout': current_config.timeout_per_task,
                    'risk_threshold': current_config.risk_threshold,
                    'automation_level': current_config.automation_level
                }
                filtered_checklist.append(item)
        
        logger.info(f"Filtered checklist: {len(filtered_checklist)}/{len(checklist)} items for {current_config.name}")
        return filtered_checklist
    
    def process_checklist_item(self, item: Dict[str, Any], connector_instance=None) -> Dict[str, Any]:
        """
        Process a single checklist item based on current mode.
        
        Args:
            item: Checklist item to process
            connector_instance: Instance of connector for AI communication
            
        Returns:
            Processing result with status and output
        """
        processor = self.mode_processors[self.current_mode]
        return processor(item, connector_instance)
    
    def _process_recon_only(self, item: Dict[str, Any], connector_instance=None) -> Dict[str, Any]:
        """Process item in reconnaissance-only mode."""
        logger.info(f"Processing in RECON_ONLY mode: {item.get('name', 'Unknown')}")
        
        # Only process passive reconnaissance tasks
        if 'recon' not in item.get('category', '').lower():
            return {
                'status': 'skipped',
                'reason': 'Not a reconnaissance task',
                'output': None,
                'requires_approval': True
            }
        
        # Simulate passive reconnaissance
        result = {
            'status': 'completed',
            'output': f"Passive reconnaissance completed for: {item.get('name')}",
            'findings': [],
            'requires_approval': True,
            'risk_level': 'low'
        }
        
        return result
    
    def _process_ai_assisted(self, item: Dict[str, Any], connector_instance=None) -> Dict[str, Any]:
        """Process item in AI-assisted mode."""
        logger.info(f"Processing in AI_ASSISTED mode: {item.get('name', 'Unknown')}")
        
        # Generate AI suggestions and guidance
        if connector_instance:
            try:
                # Get AI suggestions for the task
                prompt = f"Provide testing guidance for: {item.get('description', item.get('name'))}"
                ai_response = connector_instance.get_response(prompt)
                
                result = {
                    'status': 'ai_guidance_provided',
                    'output': ai_response,
                    'suggestions': self._extract_suggestions(ai_response),
                    'requires_approval': True,
                    'risk_level': 'medium'
                }
            except Exception as e:
                logger.error(f"AI assistance failed: {e}")
                result = {
                    'status': 'error',
                    'output': f"AI assistance failed: {str(e)}",
                    'requires_approval': True
                }
        else:
            result = {
                'status': 'guidance_ready',
                'output': f"Manual testing guidance prepared for: {item.get('name')}",
                'requires_approval': True
            }
        
        return result
    
    def _process_full_automation(self, item: Dict[str, Any], connector_instance=None) -> Dict[str, Any]:
        """Process item in full automation mode."""
        logger.info(f"Processing in FULL_AUTOMATION mode: {item.get('name', 'Unknown')}")
        
        # Perform automated testing
        if connector_instance:
            try:
                # Get comprehensive AI analysis and automated testing suggestions
                prompt = f"""
                Perform comprehensive automated security testing analysis for:
                {item.get('description', item.get('name'))}
                
                Provide:
                1. Automated testing commands
                2. Expected results
                3. Risk assessment
                4. Potential vulnerabilities to check
                """
                
                ai_response = connector_instance.get_response(prompt)
                
                result = {
                    'status': 'automated_testing_completed',
                    'output': ai_response,
                    'automated_commands': self._extract_commands(ai_response),
                    'vulnerabilities_found': self._extract_vulnerabilities(ai_response),
                    'requires_approval': False,
                    'risk_level': 'high'
                }
            except Exception as e:
                logger.error(f"Automated testing failed: {e}")
                result = {
                    'status': 'automation_error',
                    'output': f"Automated testing failed: {str(e)}",
                    'requires_approval': True
                }
        else:
            result = {
                'status': 'automation_ready',
                'output': f"Automated testing configured for: {item.get('name')}",
                'requires_approval': False
            }
        
        return result
    
    def _process_report_only(self, item: Dict[str, Any], connector_instance=None) -> Dict[str, Any]:
        """Process item in report-only mode."""
        logger.info(f"Processing in REPORT_ONLY mode: {item.get('name', 'Unknown')}")
        
        # Only process items related to reporting and documentation
        if 'report' not in item.get('category', '').lower():
            return {
                'status': 'skipped',
                'reason': 'Not a reporting task',
                'output': None,
                'requires_approval': True
            }
        
        # Generate report content
        result = {
            'status': 'report_generated',
            'output': f"Report section generated for: {item.get('name')}",
            'report_content': self._generate_report_content(item),
            'requires_approval': True,
            'risk_level': 'low'
        }
        
        return result
    
    def _extract_suggestions(self, ai_response: str) -> List[str]:
        """Extract actionable suggestions from AI response."""
        # Simple extraction logic - in practice, this would be more sophisticated
        suggestions = []
        lines = ai_response.split('\n')
        for line in lines:
            if line.strip().startswith(('-', '•', '*')) or 'suggest' in line.lower():
                suggestions.append(line.strip())
        return suggestions[:5]  # Limit to top 5 suggestions
    
    def _extract_commands(self, ai_response: str) -> List[str]:
        """Extract automated testing commands from AI response."""
        commands = []
        lines = ai_response.split('\n')
        for line in lines:
            if any(cmd in line.lower() for cmd in ['curl', 'nmap', 'nikto', 'gobuster', 'sqlmap']):
                commands.append(line.strip())
        return commands
    
    def _extract_vulnerabilities(self, ai_response: str) -> List[Dict[str, str]]:
        """Extract potential vulnerabilities from AI response."""
        vulnerabilities = []
        # Simple pattern matching - in practice, this would use NLP
        vuln_keywords = ['xss', 'sqli', 'csrf', 'rce', 'lfi', 'rfi', 'xxe']
        
        for keyword in vuln_keywords:
            if keyword.upper() in ai_response.upper():
                vulnerabilities.append({
                    'type': keyword.upper(),
                    'confidence': 'medium',
                    'description': f"Potential {keyword.upper()} vulnerability detected"
                })
        
        return vulnerabilities
    
    def _generate_report_content(self, item: Dict[str, Any]) -> str:
        """Generate report content for an item."""
        return f"""
## {item.get('name', 'Test Item')}

**Category:** {item.get('category', 'Unknown')}
**Status:** {item.get('status', 'Pending')}
**Description:** {item.get('description', 'No description available')}

### Findings
- Analysis completed in report-only mode
- No active testing performed
- Documentation generated from existing data

### Recommendations
- Review findings manually
- Consider additional testing if required
- Update security documentation
        """.strip()
    
    def get_mode_summary(self) -> Dict[str, Any]:
        """Get a summary of the current mode and its settings."""
        config = self.get_current_mode_config()
        return {
            'current_mode': self.current_mode.value,
            'mode_name': config.name,
            'description': config.description,
            'automation_level': config.automation_level,
            'requires_approval': config.requires_human_approval,
            'allowed_categories': config.checklist_filter,
            'max_concurrent_tasks': config.max_concurrent_tasks,
            'timeout_per_task': config.timeout_per_task,
            'risk_threshold': config.risk_threshold
        }


# Factory function for easy integration
def get_modes() -> List[str]:
    """Return a list of all available mode names."""
    manager = ModesManager()
    return [config.name for config in manager.get_all_modes().values()]


def create_modes_manager() -> ModesManager:
    """Create and return a new ModesManager instance."""
    return ModesManager()


# Example usage and testing
if __name__ == "__main__":
    # Example usage
    manager = create_modes_manager()
    
    # Test mode switching
    print("Testing mode switching...")
    for mode in AIMode:
        manager.set_mode(mode)
        summary = manager.get_mode_summary()
        print(f"\nMode: {summary['mode_name']}")
        print(f"Description: {summary['description']}")
        print(f"Automation Level: {summary['automation_level']}/4")
    
    # Test checklist filtering
    sample_checklist = [
        {'name': 'Subdomain Enumeration', 'category': 'reconnaissance'},
        {'name': 'SQL Injection Test', 'category': 'vulnerability_scanning'},
        {'name': 'Generate Report', 'category': 'report_generation'},
        {'name': 'XSS Testing', 'category': 'manual_testing'}
    ]
    
    print("\n\nTesting checklist filtering...")
    manager.set_mode(AIMode.RECON_ONLY)
    filtered = manager.filter_checklist(sample_checklist)
    print(f"RECON_ONLY filtered items: {len(filtered)}")
    
    manager.set_mode(AIMode.AI_ASSISTED)
    filtered = manager.filter_checklist(sample_checklist)
    print(f"AI_ASSISTED filtered items: {len(filtered)}")