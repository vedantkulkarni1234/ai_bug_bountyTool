import tkinter as tk
import speech_recognition as sr
from tkinter import ttk, scrolledtext, messagebox, filedialog
import threading
import queue
import datetime
import json
from typing import List, Dict, Any

# Import the required modules (these would be your actual modules)
try:
    import connector
    import modes_manager
except ImportError as e:
    print(f"Warning: Could not import modules: {e}")
    # Create mock modules for demonstration
    class MockConnector:
        @staticmethod
        def process_task(task_input: str, mode: str) -> Dict[str, Any]:
            return {
                "checklist": [
                    "Reconnaissance and Information Gathering",
                    "Subdomain Enumeration", 
                    "Port Scanning and Service Detection",
                    "Web Application Security Testing",
                    "SQL Injection Testing",
                    "Cross-Site Scripting (XSS) Testing",
                    "Authentication and Session Management",
                    "Business Logic Testing"
                ],
                "urls": [
                    "https://target.com",
                    "https://api.target.com", 
                    "https://admin.target.com",
                    "https://dev.target.com"
                ]
            }
    
    class MockModesManager:
        @staticmethod
        def get_modes() -> List[str]:
            return ["Quick Scan", "Comprehensive", "Web App Focus", "API Testing"]
        
        @staticmethod
        def get_mode_config(mode: str) -> Dict[str, Any]:
            return {"timeout": 30, "threads": 5, "depth": 3}
    
    connector = MockConnector()
    modes_manager = MockModesManager()


class BugBountyAssistantGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("AI-Powered Bug Bounty Assistant")
        self.root.geometry("1200x800")
        self.root.configure(bg="#2b2b2b")
        
        # Initialize variables
        self.current_checklist = []
        self.current_urls = []
        self.execution_queue = queue.Queue()
        self.log_queue = queue.Queue()
        self.is_executing = False
        
        # Configure styles
        self.setup_styles()
        
        # Create the UI
        self.create_widgets()
        
        # Start the log processing thread
        self.start_log_processor()
    
    def setup_styles(self):
        """Configure custom styles for the application"""
        style = ttk.Style()
        style.theme_use('clam')
        
        # Configure custom styles
        style.configure('Title.TLabel', 
                       background='#2b2b2b', 
                       foreground='#ffffff',
                       font=('Arial', 12, 'bold'))
        
        style.configure('Custom.TFrame',
                       background='#2b2b2b',
                       relief='flat')
        
        style.configure('Custom.TButton',
                       background='#4a9eff',
                       foreground='white',
                       font=('Arial', 10, 'bold'),
                       padding=(10, 5))
        
        style.configure('Accent.TButton', foreground='white', background='#0078D7')
        style.configure('Listening.TButton', foreground='white', background='#E74C3C')
        
        style.map('Custom.TButton',
                 background=[('active', '#3d8bdb')])
    
    def create_widgets(self):
        """Create and layout all GUI widgets"""
        # Main container
        main_frame = ttk.Frame(self.root, style='Custom.TFrame')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create sections
        self.create_header(main_frame)
        self.create_input_section(main_frame)
        self.create_control_section(main_frame)
        self.create_display_section(main_frame)
        self.create_log_section(main_frame)
    
    def create_header(self, parent):
        """Create the header section"""
        header_frame = ttk.Frame(parent, style='Custom.TFrame')
        header_frame.pack(fill=tk.X, pady=(0, 10))
        
        title_label = ttk.Label(header_frame, 
                               text="🛡️ AI-Powered Bug Bounty Assistant",
                               style='Title.TLabel',
                               font=('Arial', 16, 'bold'))
        title_label.pack(side=tk.LEFT)
        
        # Status indicator
        self.status_label = ttk.Label(header_frame,
                                     text="Status: Ready",
                                     style='Title.TLabel',
                                     font=('Arial', 10))
        self.status_label.pack(side=tk.RIGHT)
    
    def create_input_section(self, parent):
        """Create the task input section"""
        input_frame = ttk.LabelFrame(parent, text="Task Input", padding=10)
        input_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Text input
        text_frame = ttk.Frame(input_frame)
        text_frame.pack(fill=tk.X, pady=(0, 5))
        
        ttk.Label(text_frame, text="Target/Task Description:").pack(anchor=tk.W)
        self.task_input = scrolledtext.ScrolledText(text_frame, height=4, width=80)
        self.task_input.pack(fill=tk.X, pady=(5, 0))
        self.task_input.insert('1.0', 'Enter target URL, domain, or task description here...')
        
        # Voice input button (placeholder)
        voice_frame = ttk.Frame(input_frame)
        voice_frame.pack(fill=tk.X, pady=(5, 0))
        
        self.voice_button = ttk.Button(voice_frame, 
                                      text="🎤 Voice Input",
                                      command=self.handle_voice_input)
        self.voice_button.pack(side=tk.LEFT)
        
        # File input button
        self.file_button = ttk.Button(voice_frame,
                                     text="📁 Load from File",
                                     command=self.handle_file_input)
        self.file_button.pack(side=tk.LEFT, padx=(10, 0))
    
    def create_control_section(self, parent):
        """Create the control buttons and mode selection"""
        control_frame = ttk.Frame(parent, style='Custom.TFrame')
        control_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Left side - Mode selection
        mode_frame = ttk.Frame(control_frame)
        mode_frame.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        ttk.Label(mode_frame, text="Scanning Mode:").pack(anchor=tk.W)
        self.mode_var = tk.StringVar()
        self.mode_dropdown = ttk.Combobox(mode_frame, 
                                         textvariable=self.mode_var,
                                         values=modes_manager.get_modes(),
                                         state="readonly",
                                         width=20)
        self.mode_dropdown.pack(anchor=tk.W, pady=(5, 0))
        self.mode_dropdown.set(modes_manager.get_modes()[0])
        
        # Right side - Control buttons
        button_frame = ttk.Frame(control_frame)
        button_frame.pack(side=tk.RIGHT)
        
        self.generate_button = ttk.Button(button_frame,
                                         text="Generate Checklist",
                                         style='Custom.TButton',
                                         command=self.handle_generate_checklist)
        self.generate_button.pack(side=tk.LEFT, padx=(0, 10))
        
        self.execute_button = ttk.Button(button_frame,
                                        text="Start Execution",
                                        style='Custom.TButton',
                                        command=self.handle_start_execution,
                                        state=tk.DISABLED)
        self.execute_button.pack(side=tk.LEFT, padx=(0, 10))
        
        self.stop_button = ttk.Button(button_frame,
                                     text="Stop",
                                     command=self.handle_stop_execution,
                                     state=tk.DISABLED)
        self.stop_button.pack(side=tk.LEFT)
    
    def create_display_section(self, parent):
        """Create the checklist and URL display section"""
        display_frame = ttk.Frame(parent, style='Custom.TFrame')
        display_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        # Checklist display
        checklist_frame = ttk.LabelFrame(display_frame, text="Generated Checklist", padding=10)
        checklist_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        # Checklist with checkboxes
        self.checklist_frame = ttk.Frame(checklist_frame)
        self.checklist_frame.pack(fill=tk.BOTH, expand=True)
        
        self.checklist_scroll = scrolledtext.ScrolledText(self.checklist_frame, 
                                                         height=15, width=50,
                                                         state=tk.DISABLED)
        self.checklist_scroll.pack(fill=tk.BOTH, expand=True)
        
        # URL list display
        url_frame = ttk.LabelFrame(display_frame, text="Discovered URLs", padding=10)
        url_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))
        
        self.url_listbox = tk.Listbox(url_frame, height=15)
        self.url_listbox.pack(fill=tk.BOTH, expand=True)
        
        # URL controls
        url_control_frame = ttk.Frame(url_frame)
        url_control_frame.pack(fill=tk.X, pady=(5, 0))
        
        ttk.Button(url_control_frame, 
                  text="Export URLs",
                  command=self.export_urls).pack(side=tk.LEFT)
        
        ttk.Button(url_control_frame,
                  text="Clear URLs", 
                  command=self.clear_urls).pack(side=tk.LEFT, padx=(5, 0))
    
    def create_log_section(self, parent):
        """Create the real-time log display"""
        log_frame = ttk.LabelFrame(parent, text="Execution Log", padding=10)
        log_frame.pack(fill=tk.X, pady=(0, 0))
        
        self.log_display = scrolledtext.ScrolledText(log_frame, 
                                                    height=8, 
                                                    state=tk.DISABLED,
                                                    bg="#1e1e1e",
                                                    fg="#00ff00",
                                                    font=("Consolas", 9))
        self.log_display.pack(fill=tk.BOTH, expand=True)
        
        # Log controls
        log_control_frame = ttk.Frame(log_frame)
        log_control_frame.pack(fill=tk.X, pady=(5, 0))
        
        ttk.Button(log_control_frame,
                  text="Clear Log",
                  command=self.clear_log).pack(side=tk.LEFT)
        
        ttk.Button(log_control_frame,
                  text="Save Log",
                  command=self.save_log).pack(side=tk.LEFT, padx=(5, 0))
    
    # Event Handler Methods
    def handle_voice_input(self):
        """Handle voice input using speech_recognition."""
        r = sr.Recognizer()
        with sr.Microphone() as source:
            self.voice_button.config(text="Listening...", style='Listening.TButton', state=tk.DISABLED)
            self.log_message("Listening for voice input...")
            self.update_status("Listening...")
            self.root.update()
            try:
                audio = r.listen(source, timeout=5, phrase_time_limit=5)
                self.log_message("Processing voice input...")
                self.update_status("Processing...")
                self.root.update()
                text = r.recognize_google(audio)
                self.task_input.delete(1.0, tk.END)
                self.task_input.insert(tk.END, text)
                self.log_message(f"Recognized text: {text}")
            except sr.WaitTimeoutError:
                self.log_message("Voice input timed out.")
                messagebox.showwarning("Voice Input", "No speech detected. Please try again.")
            except sr.UnknownValueError:
                self.log_message("Could not understand audio.")
                messagebox.showerror("Voice Input", "Sorry, I could not understand the audio.")
            except sr.RequestError as e:
                self.log_message(f"Could not request results from Google Speech Recognition service; {e}")
                messagebox.showerror("Voice Input", f"Could not request results; {e}")
            finally:
                self.voice_button.config(text="Voice Input", style='TButton', state=tk.NORMAL)
                self.update_status("Ready")
    
    def handle_file_input(self):
        """Handle file input for task description"""
        file_path = filedialog.askopenfilename(
            title="Select Task File",
            filetypes=[("Text files", "*.txt"), ("JSON files", "*.json"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    content = file.read()
                    self.task_input.delete('1.0', tk.END)
                    self.task_input.insert('1.0', content)
                    self.log_message(f"Loaded task from file: {file_path}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load file: {str(e)}")
    
    def handle_generate_checklist(self):
        """Handle checklist generation"""
        task_text = self.task_input.get('1.0', tk.END).strip()
        
        if not task_text or task_text == 'Enter target URL, domain, or task description here...':
            messagebox.showwarning("Warning", "Please enter a task description")
            return
        
        selected_mode = self.mode_var.get()
        
        if not selected_mode:
            messagebox.showwarning("Warning", "Please select a scanning mode")
            return
        
        # Disable button and show processing
        self.generate_button.config(state=tk.DISABLED)
        self.update_status("Generating checklist...")
        
        # Run in separate thread to avoid GUI freezing
        threading.Thread(target=self.generate_checklist_thread, 
                        args=(task_text, selected_mode),
                        daemon=True).start()
    
    def generate_checklist_thread(self, task_text, mode):
        """Thread function for checklist generation"""
        try:
            self.log_message(f"Generating checklist for mode: {mode}")
            
            # Call connector to process task
            result = connector.accept_task_input(task_text, input_type='text', mode=mode)
            
            # Update GUI in main thread
            self.root.after(0, self.update_checklist_display, result)
            
        except Exception as e:
            self.root.after(0, self.handle_generation_error, str(e))
    
    def update_checklist_display(self, result):
        """Update the checklist and URL displays"""
        self.current_checklist = result.get('checklist', [])
        self.current_urls = result.get('urls', [])
        
        # Update checklist display
        self.checklist_scroll.config(state=tk.NORMAL)
        self.checklist_scroll.delete('1.0', tk.END)
        
        for i, item in enumerate(self.current_checklist, 1):
            self.checklist_scroll.insert(tk.END, f"{i}. {item}\n")
        
        self.checklist_scroll.config(state=tk.DISABLED)
        
        # Update URL list
        self.update_url_display()
        
        # Enable execution button
        self.execute_button.config(state=tk.NORMAL)
        self.generate_button.config(state=tk.NORMAL)
        
        self.update_status("Checklist generated successfully")
        self.log_message(f"Generated checklist with {len(self.current_checklist)} items")
        self.log_message(f"Discovered {len(self.current_urls)} URLs")
    
    def handle_generation_error(self, error_msg):
        """Handle errors during checklist generation"""
        messagebox.showerror("Error", f"Failed to generate checklist: {error_msg}")
        self.generate_button.config(state=tk.NORMAL)
        self.update_status("Ready")
        self.log_message(f"Error generating checklist: {error_msg}")
    
    def handle_start_execution(self):
        """Handle execution start"""
        if not self.current_checklist:
            messagebox.showwarning("Warning", "No checklist available. Generate one first.")
            return
        
        self.is_executing = True
        self.execute_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        self.generate_button.config(state=tk.DISABLED)
        
        self.update_status("Executing checklist...")
        self.log_message("Starting checklist execution")
        
        # Start execution in separate thread
        threading.Thread(target=self.execute_checklist_thread, daemon=True).start()
    
    def execute_checklist_thread(self):
        """Thread function for checklist execution"""
        try:
            for i, task in enumerate(self.current_checklist, 1):
                if not self.is_executing:
                    break
                
                self.log_message(f"Executing task {i}/{len(self.current_checklist)}: {task}")
                
                # Simulate task execution (replace with actual implementation)
                import time
                time.sleep(2)  # Simulate work
                
                # Update progress
                progress = (i / len(self.current_checklist)) * 100
                self.root.after(0, self.update_execution_progress, i, progress)
            
            if self.is_executing:
                self.root.after(0, self.execution_completed)
            else:
                self.root.after(0, self.execution_stopped)
                
        except Exception as e:
            self.root.after(0, self.execution_error, str(e))
    
    def update_execution_progress(self, current_task, progress):
        """Update execution progress"""
        self.update_status(f"Executing... ({current_task}/{len(self.current_checklist)})")
    
    def execution_completed(self):
        """Handle execution completion"""
        self.is_executing = False
        self.execute_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        self.generate_button.config(state=tk.NORMAL)
        
        self.update_status("Execution completed")
        self.log_message("Checklist execution completed successfully")
        
        messagebox.showinfo("Execution Complete", "All tasks have been executed successfully!")
    
    def execution_stopped(self):
        """Handle execution stop"""
        self.is_executing = False
        self.execute_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        self.generate_button.config(state=tk.NORMAL)
        
        self.update_status("Execution stopped")
        self.log_message("Execution stopped by user")
    
    def execution_error(self, error_msg):
        """Handle execution errors"""
        self.is_executing = False
        self.execute_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        self.generate_button.config(state=tk.NORMAL)
        
        self.update_status("Execution failed")
        self.log_message(f"Execution error: {error_msg}")
        
        messagebox.showerror("Execution Error", f"Execution failed: {error_msg}")
    
    def handle_stop_execution(self):
        """Handle execution stop"""
        self.is_executing = False
        self.log_message("Stop requested by user")
    
    # Utility Methods
    def update_status(self, message):
        """Update status label"""
        self.status_label.config(text=f"Status: {message}")
        self.root.update_idletasks()
    
    def update_url_display(self):
        """Update URL list display"""
        self.url_listbox.delete(0, tk.END)
        for url in self.current_urls:
            self.url_listbox.insert(tk.END, url)
    
    def export_urls(self):
        """Export URLs to file"""
        if not self.current_urls:
            messagebox.showinfo("Info", "No URLs to export")
            return
        
        file_path = filedialog.asksaveasfilename(
            title="Save URLs",
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("JSON files", "*.json")]
        )
        
        if file_path:
            try:
                with open(file_path, 'w', encoding='utf-8') as file:
                    if file_path.endswith('.json'):
                        json.dump(self.current_urls, file, indent=2)
                    else:
                        file.write('\n'.join(self.current_urls))
                
                self.log_message(f"URLs exported to: {file_path}")
                messagebox.showinfo("Success", f"URLs exported to {file_path}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to export URLs: {str(e)}")
    
    def clear_urls(self):
        """Clear URL list"""
        self.current_urls.clear()
        self.update_url_display()
        self.log_message("URL list cleared")
    
    def log_message(self, message):
        """Add message to log queue"""
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        self.log_queue.put(f"[{timestamp}] {message}")
    
    def start_log_processor(self):
        """Start log processing thread"""
        def process_logs():
            while True:
                try:
                    message = self.log_queue.get(timeout=0.1)
                    self.root.after(0, self.display_log_message, message)
                except queue.Empty:
                    continue
        
        threading.Thread(target=process_logs, daemon=True).start()
    
    def display_log_message(self, message):
        """Display log message in the log area"""
        self.log_display.config(state=tk.NORMAL)
        self.log_display.insert(tk.END, message + '\n')
        self.log_display.see(tk.END)
        self.log_display.config(state=tk.DISABLED)
    
    def clear_log(self):
        """Clear log display"""
        self.log_display.config(state=tk.NORMAL)
        self.log_display.delete('1.0', tk.END)
        self.log_display.config(state=tk.DISABLED)
    
    def save_log(self):
        """Save log to file"""
        file_path = filedialog.asksaveasfilename(
            title="Save Log",
            defaultextension=".log",
            filetypes=[("Log files", "*.log"), ("Text files", "*.txt")]
        )
        
        if file_path:
            try:
                content = self.log_display.get('1.0', tk.END)
                with open(file_path, 'w', encoding='utf-8') as file:
                    file.write(content)
                
                messagebox.showinfo("Success", f"Log saved to {file_path}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save log: {str(e)}")


def main():
    """Main function to run the application"""
    root = tk.Tk()
    app = BugBountyAssistantGUI(root)
    
    # Handle window closing
    def on_closing():
        if app.is_executing:
            if messagebox.askokcancel("Quit", "Execution is in progress. Do you want to quit?"):
                app.is_executing = False
                root.destroy()
        else:
            root.destroy()
    
    root.protocol("WM_DELETE_WINDOW", on_closing)
    
    # Center the window
    root.update_idletasks()
    x = (root.winfo_screenwidth() // 2) - (root.winfo_width() // 2)
    y = (root.winfo_screenheight() // 2) - (root.winfo_height() // 2)
    root.geometry(f"+{x}+{y}")
    
    # Start the application
    root.mainloop()


if __name__ == "__main__":
    main()