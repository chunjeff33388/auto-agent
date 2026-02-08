"""
Autonomous AI Agent - Production Ready
A web-based autonomous agent similar to Manus/Kimi
Deployable to Hugging Face Spaces
"""

import streamlit as st
import requests
import json
import re
import os
import time
import threading
import pandas as pd
import io
import base64
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed
import traceback

# ==================== CONFIGURATION ====================

@dataclass
class AgentConfig:
    """Configuration for the autonomous agent"""
    groq_api_key: str = ""
    serpapi_key: str = ""
    model: str = "llama-3.1-70b-versatile"  # Free tier on Groq
    max_tokens: int = 4096
    temperature: float = 0.7
    max_iterations: int = 10
    enable_multi_agent: bool = True
    max_agents: int = 3

# ==================== MEMORY SYSTEM ====================

class AgentMemory:
    """Short-term memory for the agent using Streamlit session state"""
    
    def __init__(self):
        if 'memory' not in st.session_state:
            st.session_state.memory = {
                'conversations': [],
                'task_history': [],
                'learnings': [],
                'current_context': {}
            }
        self.memory = st.session_state.memory
    
    def add_conversation(self, role: str, content: str):
        """Add to conversation history"""
        self.memory['conversations'].append({
            'timestamp': datetime.now().isoformat(),
            'role': role,
            'content': content
        })
        # Keep last 20 messages
        self.memory['conversations'] = self.memory['conversations'][-20:]
    
    def add_task_result(self, task: str, result: Any):
        """Store task result"""
        self.memory['task_history'].append({
            'timestamp': datetime.now().isoformat(),
            'task': task,
            'result': result
        })
    
    def add_learning(self, learning: str):
        """Store learned information"""
        self.memory['learnings'].append({
            'timestamp': datetime.now().isoformat(),
            'content': learning
        })
    
    def get_context(self) -> str:
        """Get relevant context for current task"""
        recent_conv = self.memory['conversations'][-5:] if self.memory['conversations'] else []
        recent_tasks = self.memory['task_history'][-3:] if self.memory['task_history'] else []
        
        context = "=== RECENT CONTEXT ===\n"
        if recent_conv:
            context += "\nRecent Conversations:\n"
            for conv in recent_conv:
                context += f"- {conv['role']}: {conv['content'][:100]}...\n"
        
        if recent_tasks:
            context += "\nRecent Tasks:\n"
            for task in recent_tasks:
                context += f"- {task['task'][:100]}... -> {str(task['result'])[:100]}\n"
        
        return context
    
    def clear(self):
        """Clear memory"""
        self.memory = {
            'conversations': [],
            'task_history': [],
            'learnings': [],
            'current_context': {}
        }
        st.session_state.memory = self.memory

# ==================== LLM INTERFACE ====================

class GroqLLM:
    """Interface to Groq API (free tier)"""
    
    def __init__(self, api_key: str, model: str = "llama-3.1-70b-versatile"):
        self.api_key = api_key
        self.model = model
        self.base_url = "https://api.groq.com/openai/v1"
    
    def chat(self, messages: List[Dict[str, str]], temperature: float = 0.7, max_tokens: int = 4096) -> str:
        """Send chat completion request to Groq"""
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": self.model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens
            }
            
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=60
            )
            
            if response.status_code == 200:
                return response.json()['choices'][0]['message']['content']
            else:
                error_msg = f"API Error {response.status_code}: {response.text}"
                st.error(error_msg)
                return f"Error: {error_msg}"
                
        except requests.exceptions.Timeout:
            return "Error: Request timed out. Please try again."
        except Exception as e:
            return f"Error: {str(e)}"
    
    def generate(self, prompt: str, system_prompt: str = "") -> str:
        """Generate text from prompt"""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        return self.chat(messages)

# ==================== TOOLS ====================

class ToolRegistry:
    """Registry of available tools for the agent"""
    
    def __init__(self, serpapi_key: str = ""):
        self.serpapi_key = serpapi_key
    
    def web_search(self, query: str, num_results: int = 5) -> str:
        """Search the web using DuckDuckGo (free) or SerpAPI"""
        try:
            # Try DuckDuckGo first (free, no API key)
            from duckduckgo_search import DDGS
            
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=num_results))
            
            if results:
                output = "=== WEB SEARCH RESULTS ===\n\n"
                for i, result in enumerate(results, 1):
                    output += f"{i}. {result['title']}\n"
                    output += f"   URL: {result['href']}\n"
                    output += f"   Snippet: {result['body'][:200]}...\n\n"
                return output
            
            return "No results found."
            
        except Exception as e:
            # Fallback to SerpAPI if available
            if self.serpapi_key:
                try:
                    url = "https://serpapi.com/search"
                    params = {
                        "q": query,
                        "api_key": self.serpapi_key,
                        "engine": "google",
                        "num": num_results
                    }
                    response = requests.get(url, params=params, timeout=30)
                    data = response.json()
                    
                    output = "=== WEB SEARCH RESULTS ===\n\n"
                    for i, result in enumerate(data.get('organic_results', [])[:num_results], 1):
                        output += f"{i}. {result.get('title', 'No title')}\n"
                        output += f"   URL: {result.get('link', 'No link')}\n"
                        output += f"   Snippet: {result.get('snippet', 'No snippet')[:200]}...\n\n"
                    return output
                    
                except Exception as e2:
                    return f"Search error: {str(e)} | SerpAPI error: {str(e2)}"
            
            return f"Search error: {str(e)}"
    
    def fetch_url(self, url: str) -> str:
        """Fetch content from a URL"""
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.0'
            }
            response = requests.get(url, headers=headers, timeout=30)
            
            if response.status_code == 200:
                # Simple HTML to text extraction
                text = re.sub('<[^<]+?>', '', response.text)
                text = re.sub('\\s+', ' ', text).strip()
                return text[:5000]  # Limit length
            else:
                return f"Error: HTTP {response.status_code}"
                
        except Exception as e:
            return f"Error fetching URL: {str(e)}"
    
    def execute_python(self, code: str) -> str:
        """Safely execute Python code"""
        # Create restricted environment
        safe_globals = {
            "__builtins__": {
                "len": len,
                "range": range,
                "enumerate": enumerate,
                "zip": zip,
                "map": map,
                "filter": filter,
                "sum": sum,
                "min": min,
                "max": max,
                "abs": abs,
                "round": round,
                "str": str,
                "int": int,
                "float": float,
                "list": list,
                "dict": dict,
                "set": set,
                "tuple": tuple,
                "print": print,
                "type": type,
                "isinstance": isinstance,
                "hasattr": hasattr,
                "getattr": getattr,
                "Exception": Exception
            }
        }
        
        # Capture output
        import sys
        from io import StringIO
        
        old_stdout = sys.stdout
        redirected_output = StringIO()
        
        try:
            sys.stdout = redirected_output
            exec(code, safe_globals)
            sys.stdout = old_stdout
            
            output = redirected_output.getvalue()
            return output if output else "Code executed successfully (no output)"
            
        except Exception as e:
            sys.stdout = old_stdout
            return f"Error executing code: {str(e)}\n{traceback.format_exc()}"
    
    def create_csv(self, data: List[Dict], filename: str = "data.csv") -> bytes:
        """Create CSV file from data"""
        df = pd.DataFrame(data)
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False)
        return csv_buffer.getvalue().encode('utf-8')
    
    def create_excel(self, data: List[Dict], filename: str = "data.xlsx") -> bytes:
        """Create Excel file from data"""
        df = pd.DataFrame(data)
        excel_buffer = io.BytesIO()
        df.to_excel(excel_buffer, index=False, engine='openpyxl')
        return excel_buffer.getvalue()
    
    def get_image_from_url(self, url: str) -> Optional[bytes]:
        """Fetch image from URL"""
        try:
            response = requests.get(url, timeout=30)
            if response.status_code == 200 and 'image' in response.headers.get('content-type', ''):
                return response.content
            return None
        except:
            return None

# ==================== AUTONOMOUS AGENT ====================

class AutonomousAgent:
    """Main autonomous agent class"""
    
    def __init__(self, config: AgentConfig):
        self.config = config
        self.llm = GroqLLM(config.groq_api_key, config.model)
        self.tools = ToolRegistry(config.serpapi_key)
        self.memory = AgentMemory()
        self.execution_log = []
    
    def log(self, message: str, message_type: str = "info"):
        """Log message to execution log"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_entry = {
            'timestamp': timestamp,
            'type': message_type,
            'message': message
        }
        self.execution_log.append(log_entry)
        
        # Also update Streamlit
        if message_type == "error":
            st.error(f"[{timestamp}] {message}")
        elif message_type == "success":
            st.success(f"[{timestamp}] {message}")
        elif message_type == "warning":
            st.warning(f"[{timestamp}] {message}")
        else:
            st.info(f"[{timestamp}] {message}")
    
    def plan_task(self, task: str) -> List[Dict]:
        """Break down task into steps using chain-of-thought"""
        self.log("🧠 Analyzing task and creating plan...", "info")
        
        context = self.memory.get_context()
        
        planning_prompt = f"""You are an autonomous AI agent. Break down the following task into clear, actionable steps.

{context}

TASK: {task}

Create a step-by-step plan. Return ONLY a JSON array in this exact format:
[
    {{"step": 1, "action": "description of action", "tool": "tool_name or none", "expected_output": "what this should produce"}},
    {{"step": 2, "action": "description of action", "tool": "tool_name or none", "expected_output": "what this should produce"}}
]

Available tools:
- web_search: Search the internet for information
- fetch_url: Get content from a specific URL
- execute_python: Run Python code
- create_csv: Generate a CSV file
- create_excel: Generate an Excel file

If no tool is needed, use "tool": "none".

Respond with ONLY the JSON array, no other text."""

        response = self.llm.generate(planning_prompt, "You are a task planning assistant. Output only valid JSON.")
        
        # Extract JSON from response
        try:
            # Find JSON array in response
            json_match = re.search(r'\[.*\]', response, re.DOTALL)
            if json_match:
                plan = json.loads(json_match.group())
                self.log(f"✅ Plan created with {len(plan)} steps", "success")
                return plan
            else:
                # Fallback: create simple plan
                return [{"step": 1, "action": task, "tool": "none", "expected_output": "Complete the task"}]
        except Exception as e:
            self.log(f"⚠️ Could not parse plan, using fallback: {str(e)}", "warning")
            return [{"step": 1, "action": task, "tool": "none", "expected_output": "Complete the task"}]
    
    def execute_step(self, step: Dict, task: str) -> str:
        """Execute a single step"""
        action = step.get('action', '')
        tool = step.get('tool', 'none')
        expected = step.get('expected_output', '')
        
        self.log(f"▶️ Executing: {action[:80]}...", "info")
        
        result = ""
        
        try:
            if tool == 'web_search':
                # Extract search query from action
                query = action.replace('Search', '').replace('search for', '').replace('search', '').strip()
                result = self.tools.web_search(query)
                
            elif tool == 'fetch_url':
                # Extract URL
                urls = re.findall(r'https?://[^\s<>\"{}|\\^`\[\]]+', action)
                if urls:
                    result = self.tools.fetch_url(urls[0])
                else:
                    result = "No URL found in action"
                    
            elif tool == 'execute_python':
                # Extract code (assume it's after a code block or the whole action)
                code_match = re.search(r'```python\s*(.*?)\s*```', action, re.DOTALL)
                if code_match:
                    code = code_match.group(1)
                else:
                    # Generate code using LLM
                    code_prompt = f"Write Python code to: {action}\n\nReturn ONLY the code, no explanation."
                    code = self.llm.generate(code_prompt, "You are a Python programmer.")
                
                result = self.tools.execute_python(code)
                
            elif tool == 'create_csv':
                # Generate data using LLM
                data_prompt = f"""Generate sample data for: {action}
Return as a JSON array of objects. Example: [{{"name": "John", "age": 30}}, ...]"""
                data_response = self.llm.generate(data_prompt)
                try:
                    json_match = re.search(r'\[.*\]', data_response, re.DOTALL)
                    if json_match:
                        data = json.loads(json_match.group())
                        csv_bytes = self.tools.create_csv(data)
                        result = f"CSV created with {len(data)} rows. Download available."
                        # Store for download
                        st.session_state['generated_csv'] = csv_bytes
                        st.session_state['csv_filename'] = "output.csv"
                    else:
                        result = "Failed to generate data"
                except Exception as e:
                    result = f"Error creating CSV: {str(e)}"
                    
            elif tool == 'create_excel':
                data_prompt = f"""Generate sample data for: {action}
Return as a JSON array of objects."""
                data_response = self.llm.generate(data_prompt)
                try:
                    json_match = re.search(r'\[.*\]', data_response, re.DOTALL)
                    if json_match:
                        data = json.loads(json_match.group())
                        excel_bytes = self.tools.create_excel(data)
                        result = f"Excel file created with {len(data)} rows. Download available."
                        st.session_state['generated_excel'] = excel_bytes
                        st.session_state['excel_filename'] = "output.xlsx"
                    else:
                        result = "Failed to generate data"
                except Exception as e:
                    result = f"Error creating Excel: {str(e)}"
            else:
                # No tool - use LLM directly
                result = self.llm.generate(action, "You are a helpful assistant completing a task.")
            
            self.log(f"✅ Step completed", "success")
            return result
            
        except Exception as e:
            error_msg = f"❌ Step failed: {str(e)}"
            self.log(error_msg, "error")
            return error_msg
    
    def reflect_and_correct(self, step: Dict, result: str, expected: str) -> bool:
        """Review step result and determine if retry is needed"""
        if "Error" in result or "failed" in result.lower():
            self.log("🔄 Error detected, analyzing for retry...", "warning")
            return True
        return False
    
    def execute_task(self, task: str) -> str:
        """Execute full task with planning and execution"""
        self.execution_log = []
        self.log(f"🚀 Starting task: {task[:100]}...", "info")
        
        # Add to memory
        self.memory.add_conversation("user", task)
        
        # Create plan
        plan = self.plan_task(task)
        
        # Display plan
        with st.expander("📋 Execution Plan", expanded=True):
            for step in plan:
                st.write(f"**Step {step['step']}:** {step['action']}")
                st.caption(f"Tool: {step['tool']} | Expected: {step['expected_output']}")
        
        # Execute steps
        results = []
        for step in plan[:self.config.max_iterations]:
            with st.spinner(f"Executing step {step['step']}..."):
                result = self.execute_step(step, task)
                results.append({
                    'step': step['step'],
                    'action': step['action'],
                    'result': result
                })
                
                # Check if retry needed
                if self.reflect_and_correct(step, result, step.get('expected_output', '')):
                    self.log("🔄 Retrying with modified approach...", "warning")
                    # Modify step and retry
                    step['action'] += " (Retry with simpler approach)"
                    result = self.execute_step(step, task)
                    results[-1]['result'] = result
        
        # Generate final summary
        self.log("📝 Generating final response...", "info")
        
        summary_prompt = f"""Task: {task}

Execution Results:
{json.dumps(results, indent=2)}

Provide a comprehensive summary of what was accomplished. Include key findings, data insights, and any files generated. Be thorough but concise."""

        final_response = self.llm.generate(summary_prompt, "You are summarizing task execution results.")
        
        self.memory.add_task_result(task, final_response)
        self.memory.add_conversation("assistant", final_response)
        
        self.log("✅ Task completed!", "success")
        
        return final_response

# ==================== MULTI-AGENT SWARM ====================

class AgentSwarm:
    """Multi-agent coordination for parallel task execution"""
    
    def __init__(self, config: AgentConfig):
        self.config = config
        self.agents = []
    
    def create_sub_agents(self, subtasks: List[str]) -> List[AutonomousAgent]:
        """Create agents for each subtask"""
        agents = []
        for i, subtask in enumerate(subtasks[:self.config.max_agents]):
            agent = AutonomousAgent(self.config)
            agents.append((subtask, agent))
        return agents
    
    def execute_parallel(self, main_task: str) -> Dict[str, str]:
        """Execute subtasks in parallel"""
        # First, break down the main task
        llm = GroqLLM(self.config.groq_api_key, self.config.model)
        
        breakdown_prompt = f"""Break down this task into 2-3 parallel subtasks that can be executed simultaneously:

Task: {main_task}

Return as JSON array: [{{"subtask": "description", "focus": "what to focus on"}}]"""

        response = llm.generate(breakdown_prompt)
        
        try:
            json_match = re.search(r'\[.*\]', response, re.DOTALL)
            if json_match:
                subtasks_data = json.loads(json_match.group())
                subtasks = [s['subtask'] for s in subtasks_data]
            else:
                subtasks = [main_task]
        except:
            subtasks = [main_task]
        
        # Execute in parallel
        results = {}
        
        with ThreadPoolExecutor(max_workers=self.config.max_agents) as executor:
            future_to_task = {}
            
            for subtask in subtasks:
                agent = AutonomousAgent(self.config)
                future = executor.submit(agent.execute_task, subtask)
                future_to_task[future] = subtask
            
            for future in as_completed(future_to_task):
                subtask = future_to_task[future]
                try:
                    result = future.result()
                    results[subtask] = result
                except Exception as e:
                    results[subtask] = f"Error: {str(e)}"
        
        return results

# ==================== STREAMLIT UI ====================

def init_session_state():
    """Initialize Streamlit session state"""
    if 'agent_config' not in st.session_state:
        st.session_state.agent_config = None
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'generated_files' not in st.session_state:
        st.session_state.generated_files = {}

def render_sidebar():
    """Render sidebar with configuration"""
    with st.sidebar:
        st.title("⚙️ Configuration")
        
        # API Keys
        st.subheader("API Keys")
        groq_key = st.text_input(
            "Groq API Key",
            type="password",
            value=os.getenv("GROQ_API_KEY", ""),
            help="Get free key from groq.com"
        )
        
        serpapi_key = st.text_input(
            "SerpAPI Key (Optional)",
            type="password",
            value=os.getenv("SERPAPI_KEY", ""),
            help="Get from serpapi.com (100 free searches/month)"
        )
        
        # Model selection
        st.subheader("Model Settings")
        model = st.selectbox(
            "LLM Model",
            ["llama-3.1-70b-versatile", "llama-3.1-8b-instant", "mixtral-8x7b-32768", "gemma-7b-it"],
            help="Llama 3.1 70B recommended for best results"
        )
        
        temperature = st.slider("Temperature", 0.0, 1.0, 0.7, 0.1)
        
        # Multi-agent toggle
        enable_swarm = st.toggle("Enable Multi-Agent Swarm", value=True)
        
        # Save config
        if st.button("💾 Save Configuration"):
            st.session_state.agent_config = AgentConfig(
                groq_api_key=groq_key,
                serpapi_key=serpapi_key,
                model=model,
                temperature=temperature,
                enable_multi_agent=enable_swarm
            )
            st.success("Configuration saved!")
        
        st.divider()
        
        # Clear memory
        if st.button("🗑️ Clear Memory"):
            if 'memory' in st.session_state:
                del st.session_state.memory
            st.session_state.chat_history = []
            st.success("Memory cleared!")
        
        # Help section
        st.divider()
        st.subheader("📖 Help")
        st.markdown("""
        **Getting Started:**
        1. Enter your Groq API key
        2. Click Save Configuration
        3. Enter a task in the main area
        4. Click Execute Task
        
        **Example Tasks:**
        - "Research the top 5 AI companies and create a CSV"
        - "Write a Python script to calculate fibonacci"
        - "Search for climate change data and summarize"
        """)

def render_main_interface():
    """Render main chat interface"""
    st.title("🤖 Autonomous AI Agent")
    st.caption("Powered by Groq (Free Tier) | No OpenAI/Anthropic API needed")
    
    # Check if configured
    if st.session_state.agent_config is None:
        st.warning("⚠️ Please configure your API keys in the sidebar first!")
        return
    
    config = st.session_state.agent_config
    
    if not config.groq_api_key:
        st.error("❌ Groq API key is required! Get a free key from groq.com")
        return
    
    # Task input
    st.subheader("📝 Enter Your Task")
    task = st.text_area(
        "What would you like me to do?",
        placeholder="Example: Research the latest AI trends, find 5 articles, and create a summary report with citations...",
        height=100
    )
    
    # Execution options
    col1, col2, col3 = st.columns([1, 1, 2])
    
    with col1:
        use_swarm = st.checkbox("Use Multi-Agent", value=config.enable_multi_agent)
    
    with col2:
        show_details = st.checkbox("Show detailed logs", value=True)
    
    with col3:
        execute_btn = st.button("🚀 Execute Task", type="primary", use_container_width=True)
    
    # Execute task
    if execute_btn and task:
        # Create log container
        log_container = st.container()
        
        with log_container:
            st.subheader("📊 Execution Log")
            log_placeholder = st.empty()
        
        # Execute
        if use_swarm:
            st.info("🐝 Using multi-agent swarm for parallel execution...")
            swarm = AgentSwarm(config)
            results = swarm.execute_parallel(task)
            
            st.subheader("📋 Results from All Agents")
            for subtask, result in results.items():
                with st.expander(f"Agent: {subtask[:80]}..."):
                    st.write(result)
        else:
            agent = AutonomousAgent(config)
            result = agent.execute_task(task)
            
            st.subheader("📋 Final Result")
            st.markdown(result)
        
        # Show downloads if files generated
        if 'generated_csv' in st.session_state:
            st.download_button(
                "📥 Download CSV",
                st.session_state['generated_csv'],
                st.session_state.get('csv_filename', 'output.csv'),
                "text/csv"
            )
        
        if 'generated_excel' in st.session_state:
            st.download_button(
                "📥 Download Excel",
                st.session_state['generated_excel'],
                st.session_state.get('excel_filename', 'output.xlsx'),
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        
        # Add to chat history
        st.session_state.chat_history.append({
            'task': task,
            'timestamp': datetime.now().isoformat()
        })

def render_chat_history():
    """Render previous tasks"""
    if st.session_state.chat_history:
        with st.sidebar:
            st.divider()
            st.subheader("📜 Recent Tasks")
            for item in reversed(st.session_state.chat_history[-5:]):
                st.caption(f"• {item['task'][:50]}...")

def main():
    """Main application"""
    st.set_page_config(
        page_title="Autonomous AI Agent",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    init_session_state()
    render_sidebar()
    render_main_interface()
    render_chat_history()
    
    # Footer
    st.divider()
    st.caption("""
    🤖 Autonomous AI Agent | Built with Streamlit + Groq API (Free Tier)
    Deploy to Hugging Face Spaces for free web hosting!
    """)

if __name__ == "__main__":
    main()
