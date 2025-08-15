"""
connector.py - Central Controller Module

This module acts as the central controller for all project modules,
orchestrating task input processing, checklist generation, mode selection,
and execution coordination.
"""

import logging
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import asyncio
import traceback

# Import all required modules
def _import_module(module_name):
    try:
        return __import__(module_name)
    except ImportError as e:
        logging.error(f"Failed to import module: {module_name} - {e}")
        return None

# Import all required modules
voice_text_processor = _import_module('voice_text_processor')
checklist_generator = _import_module('checklist_generator')
data_collector = _import_module('data_collector')
data_preprocessor = _import_module('data_preprocessor')
vulnerability_predictor = _import_module('vulnerability_predictor')
ai_trainer = _import_module('ai_trainer')
ai_model = _import_module('ai_model')
utils = _import_module('utils')
modes_manager = _import_module('modes_manager')


class InputType(Enum):
    """Enumeration for input types"""
    TEXT = "text"
    VOICE = "voice"


class TaskStatus(Enum):
    """Enumeration for task execution status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class TaskResult:
    """Data class for task execution results"""
    status: TaskStatus
    result: Optional[Any] = None
    error: Optional[str] = None
    execution_time: Optional[float] = None


@dataclass
class ProcessedInput:
    """Data class for processed input results"""
    text: str
    checklist: List[Dict[str, Any]]
    urls: List[str]
    mode: str
    metadata: Dict[str, Any]


class ProjectConnector:
    """
    Central controller class that orchestrates all project modules
    """
    
    def __init__(self):
        """Initialize the connector with required components"""
        self.logger = self._setup_logging()
        self.task_history: List[Dict[str, Any]] = []
        self.current_task_id = 0
        self.active_tasks: Dict[int, Dict[str, Any]] = {}
        
        # Initialize module instances
        self._initialize_modules()
        
        self.logger.info("ProjectConnector initialized successfully")
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for the connector"""
        logger = logging.getLogger('ProjectConnector')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def _initialize_modules(self):
        """Initialize all required modules"""
        try:
            self.voice_processor = voice_text_processor
            self.checklist_gen = checklist_generator
            self.data_collector = data_collector
            self.data_preprocessor = data_preprocessor
            self.vuln_predictor = vulnerability_predictor
            self.ai_trainer = ai_trainer
            self.ai_model = ai_model
            self.utils = utils
            self.modes_manager = modes_manager
            
            self.logger.info("All modules initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize modules: {e}")
            raise
    
    def accept_task_input(
        self, 
        input_data: Union[str, bytes], 
        input_type: InputType,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Accept task input (text or voice) and process it
        
        Args:
            input_data: Either text string or voice data (bytes)
            input_type: Type of input (TEXT or VOICE)
            **kwargs: Additional parameters for processing
        
        Returns:
            Dict containing processed input results
        """
        task_id = self._generate_task_id()
        
        try:
            self.logger.info(f"Processing task {task_id} with input type: {input_type.value}")
            
            # Process input based on type
            if input_type == InputType.VOICE:
                text = self._process_voice_input(input_data, **kwargs)
            else:
                text = str(input_data)
            
            # Generate checklist and extract URLs
            checklist_result = self._generate_checklist(text, **kwargs)
            
            # Get mode selection
            mode = self._select_mode(text, checklist_result['checklist'], **kwargs)
            
            # Create processed input object
            processed_input = ProcessedInput(
                text=text,
                checklist=checklist_result['checklist'],
                urls=checklist_result['urls'],
                mode=mode,
                metadata={
                    'task_id': task_id,
                    'input_type': input_type.value,
                    'timestamp': utils.get_current_timestamp() if hasattr(utils, 'get_current_timestamp') else None,
                    'processing_time': 0  # Will be updated later
                }
            )
            
            # Store task information
            self.active_tasks[task_id] = {
                'processed_input': processed_input,
                'status': TaskStatus.PENDING,
                'created_at': utils.get_current_timestamp() if hasattr(utils, 'get_current_timestamp') else None
            }
            
            result = {
                'success': True,
                'task_id': task_id,
                'processed_input': processed_input,
                'message': 'Input processed successfully'
            }
            
            self.logger.info(f"Task {task_id} processed successfully")
            return result
            
        except Exception as e:
            error_msg = f"Failed to process task input: {str(e)}"
            self.logger.error(f"{error_msg}\n{traceback.format_exc()}")
            
            return {
                'success': False,
                'task_id': task_id,
                'error': error_msg,
                'message': 'Failed to process input'
            }
    
    def _process_voice_input(self, voice_data: bytes, **kwargs) -> str:
        """
        Process voice input using voice_text_processor
        
        Args:
            voice_data: Voice data in bytes
            **kwargs: Additional parameters for voice processing
        
        Returns:
            Transcribed text
        """
        try:
            # Call voice text processor
            if hasattr(self.voice_processor, 'transcribe_audio'):
                result = self.voice_processor.transcribe_audio(voice_data, **kwargs)
            elif hasattr(self.voice_processor, 'process_voice'):
                result = self.voice_processor.process_voice(voice_data, **kwargs)
            else:
                # Fallback method
                result = self.voice_processor.transcribe(voice_data, **kwargs)
            
            # Extract text from result
            if isinstance(result, dict):
                text = result.get('text', result.get('transcription', ''))
            else:
                text = str(result)
            
            if not text:
                raise ValueError("Voice transcription returned empty text")
            
            self.logger.info("Voice input transcribed successfully")
            return text
            
        except Exception as e:
            self.logger.error(f"Voice processing failed: {e}")
            raise
    
    def _generate_checklist(self, text: str, **kwargs) -> Dict[str, Any]:
        """
        Generate checklist and extract URLs using checklist_generator
        
        Args:
            text: Input text to process
            **kwargs: Additional parameters
        
        Returns:
            Dict containing checklist and URLs
        """
        try:
            # Call checklist generator
            if hasattr(self.checklist_gen, 'generate_checklist'):
                result = self.checklist_gen.generate_checklist(text, **kwargs)
            else:
                result = self.checklist_gen.process_text(text, **kwargs)
            
            # Ensure result has required format
            if isinstance(result, dict):
                checklist = result.get('checklist', [])
                urls = result.get('urls', [])
            else:
                checklist = result if isinstance(result, list) else []
                urls = []
            
            # Extract URLs if not already done
            if not urls and hasattr(self.checklist_gen, 'extract_urls'):
                urls = self.checklist_gen.extract_urls(text)
            
            self.logger.info(f"Generated checklist with {len(checklist)} items and {len(urls)} URLs")
            
            return {
                'checklist': checklist,
                'urls': urls
            }
            
        except Exception as e:
            self.logger.error(f"Checklist generation failed: {e}")
            raise
    
    def _select_mode(self, text: str, checklist: List[Dict], **kwargs) -> str:
        """
        Select appropriate mode using modes_manager
        
        Args:
            text: Input text
            checklist: Generated checklist
            **kwargs: Additional parameters
        
        Returns:
            Selected mode
        """
        try:
            if hasattr(self.modes_manager, 'select_mode'):
                mode = self.modes_manager.select_mode(text, checklist, **kwargs)
            else:
                mode = self.modes_manager.determine_mode(text, checklist, **kwargs)
            
            self.logger.info(f"Selected mode: {mode}")
            return mode
            
        except Exception as e:
            self.logger.error(f"Mode selection failed: {e}")
            # Return default mode
            return "default"
    
    def update_url_list(self, task_id: int, additional_urls: List[str] = None) -> Dict[str, Any]:
        """
        Update URL list for main_gui.py consumption
        
        Args:
            task_id: Task identifier
            additional_urls: Additional URLs to add
        
        Returns:
            Dict containing updated URL information
        """
        try:
            if task_id not in self.active_tasks:
                raise ValueError(f"Task {task_id} not found")
            
            task = self.active_tasks[task_id]
            current_urls = task['processed_input'].urls.copy()
            
            if additional_urls:
                current_urls.extend(additional_urls)
                # Remove duplicates while preserving order
                current_urls = list(dict.fromkeys(current_urls))
                
                # Update the task's URL list
                task['processed_input'].urls = current_urls
            
            result = {
                'success': True,
                'task_id': task_id,
                'urls': current_urls,
                'url_count': len(current_urls),
                'message': 'URL list updated successfully'
            }
            
            self.logger.info(f"Updated URL list for task {task_id}: {len(current_urls)} URLs")
            return result
            
        except Exception as e:
            error_msg = f"Failed to update URL list: {str(e)}"
            self.logger.error(error_msg)
            return {
                'success': False,
                'task_id': task_id,
                'error': error_msg
            }
    
    def execute_checklist(self, task_id: int, **kwargs) -> Dict[str, Any]:
        """
        Execute checklist items for a given task
        
        Args:
            task_id: Task identifier
            **kwargs: Additional execution parameters
        
        Returns:
            Dict containing execution results
        """
        try:
            if task_id not in self.active_tasks:
                raise ValueError(f"Task {task_id} not found")
            
            task = self.active_tasks[task_id]
            task['status'] = TaskStatus.RUNNING
            
            processed_input = task['processed_input']
            checklist = processed_input.checklist
            mode = processed_input.mode
            
            self.logger.info(f"Executing checklist for task {task_id} in {mode} mode")
            
            # Execute checklist items based on mode
            execution_results = []
            
            for i, item in enumerate(checklist):
                try:
                    result = self._execute_checklist_item(item, mode, processed_input, **kwargs)
                    execution_results.append({
                        'item_index': i,
                        'item': item,
                        'result': result,
                        'status': 'completed'
                    })
                except Exception as item_error:
                    execution_results.append({
                        'item_index': i,
                        'item': item,
                        'error': str(item_error),
                        'status': 'failed'
                    })
            
            # Update task status
            task['status'] = TaskStatus.COMPLETED
            task['execution_results'] = execution_results
            
            # Calculate success rate
            completed_items = len([r for r in execution_results if r['status'] == 'completed'])
            success_rate = completed_items / len(checklist) if checklist else 0
            
            result = {
                'success': True,
                'task_id': task_id,
                'execution_results': execution_results,
                'success_rate': success_rate,
                'completed_items': completed_items,
                'total_items': len(checklist),
                'message': f'Checklist execution completed with {success_rate:.1%} success rate'
            }
            
            self.logger.info(f"Checklist execution completed for task {task_id}")
            return result
            
        except Exception as e:
            error_msg = f"Failed to execute checklist: {str(e)}"
            self.logger.error(f"{error_msg}\n{traceback.format_exc()}")
            
            # Update task status
            if task_id in self.active_tasks:
                self.active_tasks[task_id]['status'] = TaskStatus.FAILED
            
            return {
                'success': False,
                'task_id': task_id,
                'error': error_msg
            }
    
    def _execute_checklist_item(
        self, 
        item: Dict[str, Any], 
        mode: str, 
        processed_input: ProcessedInput,
        **kwargs
    ) -> Any:
        """
        Execute a single checklist item
        
        Args:
            item: Checklist item to execute
            mode: Execution mode
            processed_input: Original processed input
            **kwargs: Additional parameters
        
        Returns:
            Execution result
        """
        item_type = item.get('type', 'unknown')
        
        if item_type == 'data_collection':
            return self._execute_data_collection(item, processed_input, **kwargs)
        elif item_type == 'data_preprocessing':
            return self._execute_data_preprocessing(item, processed_input, **kwargs)
        elif item_type == 'vulnerability_prediction':
            return self._execute_vulnerability_prediction(item, processed_input, **kwargs)
        elif item_type == 'ai_training':
            return self._execute_ai_training(item, processed_input, **kwargs)
        else:
            # Generic execution
            return self._execute_generic_item(item, mode, processed_input, **kwargs)
    
    def _execute_data_collection(self, item: Dict, processed_input: ProcessedInput, **kwargs) -> Any:
        """Execute data collection item"""
        if hasattr(self.data_collector, 'collect_data'):
            return self.data_collector.collect_data(item, processed_input.urls, **kwargs)
        return {'status': 'not_implemented'}
    
    def _execute_data_preprocessing(self, item: Dict, processed_input: ProcessedInput, **kwargs) -> Any:
        """Execute data preprocessing item"""
        if hasattr(self.data_preprocessor, 'preprocess_data'):
            return self.data_preprocessor.preprocess_data(item, **kwargs)
        return {'status': 'not_implemented'}
    
    def _execute_vulnerability_prediction(self, item: Dict, processed_input: ProcessedInput, **kwargs) -> Any:
        """Execute vulnerability prediction item"""
        if hasattr(self.vuln_predictor, 'predict_vulnerabilities'):
            return self.vuln_predictor.predict_vulnerabilities(item, **kwargs)
        return {'status': 'not_implemented'}
    
    def _execute_ai_training(self, item: Dict, processed_input: ProcessedInput, **kwargs) -> Any:
        """Execute AI training item"""
        if hasattr(self.ai_trainer, 'train_model'):
            return self.ai_trainer.train_model(item, **kwargs)
        return {'status': 'not_implemented'}
    
    def _execute_generic_item(self, item: Dict, mode: str, processed_input: ProcessedInput, **kwargs) -> Any:
        """Execute generic checklist item"""
        # Default execution logic
        return {
            'item_type': item.get('type', 'unknown'),
            'mode': mode,
            'status': 'completed',
            'message': f"Executed {item.get('description', 'item')} in {mode} mode"
        }
    
    def get_task_status(self, task_id: int) -> Dict[str, Any]:
        """
        Get status of a specific task
        
        Args:
            task_id: Task identifier
        
        Returns:
            Dict containing task status information
        """
        try:
            if task_id not in self.active_tasks:
                return {
                    'success': False,
                    'error': f'Task {task_id} not found'
                }
            
            task = self.active_tasks[task_id]
            return {
                'success': True,
                'task_id': task_id,
                'status': task['status'].value,
                'processed_input': task['processed_input'],
                'execution_results': task.get('execution_results', []),
                'created_at': task.get('created_at')
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def get_all_tasks(self) -> Dict[str, Any]:
        """
        Get information about all tasks
        
        Returns:
            Dict containing all task information
        """
        try:
            tasks_info = []
            for task_id, task in self.active_tasks.items():
                tasks_info.append({
                    'task_id': task_id,
                    'status': task['status'].value,
                    'input_type': task['processed_input'].metadata.get('input_type'),
                    'mode': task['processed_input'].mode,
                    'checklist_items': len(task['processed_input'].checklist),
                    'urls_count': len(task['processed_input'].urls),
                    'created_at': task.get('created_at')
                })
            
            return {
                'success': True,
                'tasks': tasks_info,
                'total_tasks': len(tasks_info)
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def cancel_task(self, task_id: int) -> Dict[str, Any]:
        """
        Cancel a running task
        
        Args:
            task_id: Task identifier
        
        Returns:
            Dict containing cancellation result
        """
        try:
            if task_id not in self.active_tasks:
                return {
                    'success': False,
                    'error': f'Task {task_id} not found'
                }
            
            task = self.active_tasks[task_id]
            if task['status'] != TaskStatus.RUNNING:
                return {
                    'success': False,
                    'error': f'Task {task_id} is not running (status: {task["status"].value})'
                }
            
            task['status'] = TaskStatus.CANCELLED
            
            return {
                'success': True,
                'task_id': task_id,
                'message': 'Task cancelled successfully'
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def _generate_task_id(self) -> int:
        """Generate unique task ID"""
        self.current_task_id += 1
        return self.current_task_id
    
    def cleanup_completed_tasks(self, max_completed_tasks: int = 100) -> Dict[str, Any]:
        """
        Clean up completed tasks to free memory
        
        Args:
            max_completed_tasks: Maximum number of completed tasks to keep
        
        Returns:
            Dict containing cleanup results
        """
        try:
            completed_tasks = [
                (task_id, task) for task_id, task in self.active_tasks.items()
                if task['status'] in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED]
            ]
            
            if len(completed_tasks) <= max_completed_tasks:
                return {
                    'success': True,
                    'message': 'No cleanup needed',
                    'removed_tasks': 0
                }
            
            # Sort by creation time and remove oldest
            completed_tasks.sort(key=lambda x: x[1].get('created_at', 0))
            tasks_to_remove = completed_tasks[:-max_completed_tasks]
            
            removed_count = 0
            for task_id, _ in tasks_to_remove:
                del self.active_tasks[task_id]
                removed_count += 1
            
            return {
                'success': True,
                'message': f'Cleaned up {removed_count} completed tasks',
                'removed_tasks': removed_count
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }


# Create singleton instance
connector = ProjectConnector()

# Convenience functions for external use
def accept_task_input(input_data: Union[str, bytes], input_type: str, **kwargs) -> Dict[str, Any]:
    """Convenience function to accept task input"""
    input_type_enum = InputType.TEXT if input_type.lower() == 'text' else InputType.VOICE
    return connector.accept_task_input(input_data, input_type_enum, **kwargs)

def update_url_list(task_id: int, additional_urls: List[str] = None) -> Dict[str, Any]:
    """Convenience function to update URL list"""
    return connector.update_url_list(task_id, additional_urls)

def execute_checklist(task_id: int, **kwargs) -> Dict[str, Any]:
    """Convenience function to execute checklist"""
    return connector.execute_checklist(task_id, **kwargs)

def get_task_status(task_id: int) -> Dict[str, Any]:
    """Convenience function to get task status"""
    return connector.get_task_status(task_id)

def get_all_tasks() -> Dict[str, Any]:
    """Convenience function to get all tasks"""
    return connector.get_all_tasks()

def cancel_task(task_id: int) -> Dict[str, Any]:
    """Convenience function to cancel task"""
    return connector.cancel_task(task_id)


if __name__ == "__main__":
    # Example usage
    print("ProjectConnector initialized successfully")
    
    # Test with text input
    result = accept_task_input("Analyze security vulnerabilities in web application", "text")
    print(f"Text input result: {result}")
    
    if result['success']:
        task_id = result['task_id']
        
        # Check task status
        status = get_task_status(task_id)
        print(f"Task status: {status}")
        
        # Execute checklist
        execution_result = execute_checklist(task_id)
        print(f"Execution result: {execution_result}")