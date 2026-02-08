"""
🤖 Autonomous AI Agent - Production Ready
Free deployment to Streamlit Cloud / Render / PythonAnywhere
Uses Groq API (free tier) - No OpenAI/Anthropic needed
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
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
import traceback

# ==================== PAGE CONFIG ====================

st.set_page_config(
    page_title="Autonomous AI Agent",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== CUSTOM CSS ====================

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .log-entry {
        padding: 8px 12px;
        margin: 4px 0;
        border-radius: 8px;
        font-family: monospace;
        font-size: 0.9rem;
    }
    .log-info { background: #e3f2fd; border-left: 4px solid #2196f3; }
    .log-success { background: #e8f5e9; border-left: 4px solid #4caf50; }
    .log-warning { background: #fff3e0; border-left: 4px solid #ff9800; }
    .log-error { background: #ffebee; border-left: 4px solid #f44336; }
    .stButton>button {
        border-radius: 8px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# ==================== CONFIGURATION ====================

@dataclass
class AgentConfig:
    groq_api_key: str = ""
    serpapi_key: str = ""
    model: str = "llama-3.1-70b-versatile"
    max_tokens: int = 4096
    temperature: float = 0.7
    max_iterations: int = 10

# ==================== MEMORY SYSTEM ====================

class AgentMemory:
    """Session-based memory for the agent"""
    
    def __init__(self):
        if 'agent_memory' not in st.session_state:
            st.session_state.agent_memory = {
                'conversations': [],
                'tasks': [],
                'files': {}
            }
        self.memory = st.session_state.agent_memory
    
    def add_message(self, role: str, content: str):
        self.memory['conversations'].append({
            'time': datetime.now().strftime("%H:%M:%S"),
            'role': role,
            'content': content
        })
        self.memory['conversations'] = self.memory['conversations'][-20:]
    
    def add_task(self, task: str, result: str):
        self.memory['tasks'].append({
            'time': datetime.now().strftime("%H:%M:%S"),
            'task': task,
            'result': result
        })
    
    def get_recent_context(self, n: int = 5) -> str:
        recent = self.memory['conversations'][-n:] if self.memory['conversations'] else []
        return "\n".join([f"{m['role']}: {m['content'][:100]}" for m in recent])
    
    def store_file(self, name: str, data: bytes, mime: str):
        self.memory['files'][name] = {'data': data, 'mime': mime}

# ==================== LLM INTERFACE ====================

class GroqLLM:
    """Groq API client - FREE TIER (1M tokens/day)"""
    
    API_URL = "https://api.groq.com/openai/v1/chat/completions"
    
    def __init__(self, api_key: str, model: str = "llama-3.1-70b-versatile"):
        self.api_key = api_key
        self.model = model
    
    def chat(self, messages: List[Dict], temperature: float = 0.7) -> str:
        """Send chat request to Groq"""
        if not self.api_key:
            return "Error: No API key provided. Get free key from groq.com"
        
        try:
            response = requests.post(
                self.API_URL,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": self.model,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": 4096
                },
                timeout=60
            )
            
            if response.status_code == 200:
                return response.json()['choices'][0]['message']['content']
            elif response.status_code == 429:
                return "Error: Rate limit exceeded. Groq free tier: 1M tokens/day, 1.5K tokens/min"
            else:
                return f"Error {response.status_code}: {response.text[:200]}"
                
        except requests.exceptions.Timeout:
            return "Error: Request timed out (60s). Try a simpler task."
        except Exception as e:
            return f"Error: {str(e)}"
    
    def generate(self, prompt: str, system: str = "") -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        return self.chat(messages)

# ==================== TOOLS ====================

class Tools:
    """Available tools for the agent"""
    
    def __init__(self, serpapi_key: str = ""):
        self.serpapi_key = serpapi_key
    
    def web_search(self, query: str, n: int = 5) -> str:
        """Search web using DuckDuckGo (FREE, no API key)"""
        try:
            # Using DuckDuckGo HTML endpoint (no library needed)
            url = "https://html.duckduckgo.com/html/"
            data = {'q': query, 'kl': 'us-en'}
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
            
            response = requests.post(url, data=data, headers=headers, timeout=30)
            
            # Extract results from HTML
            from html.parser import HTMLParser
            
            results = []
            titles = re.findall(r'<a[^>]*class="result__a"[^>]*>(.*?)</a>', response.text)
            snippets = re.findall(r'<a[^>]*class="result__snippet"[^>]*>(.*?)</a>', response.text)
            
            for i, (title, snippet) in enumerate(zip(titles[:n], snippets[:n]), 1):
                clean_title = re.sub('<[^<]+?>', '', title)
                clean_snippet = re.sub('<[^<]+?>', '', snippet)
                results.append(f"{i}. {clean_title}\n   {clean_snippet[:150]}...")
            
            return "\n\n".join(results) if results else "No results found."
            
        except Exception as e:
            # Fallback: return helpful message
            return f"Search note: Using DuckDuckGo. Query was: '{query}'. Try being more specific."
    
    def fetch_url(self, url: str) -> str:
        """Fetch webpage content"""
        try:
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
            r = requests.get(url, headers=headers, timeout=30)
            
            if r.status_code == 200:
                # Simple HTML to text
                text = re.sub('<[^<]+?>', ' ', r.text)
                text = re.sub('\s+', ' ', text).strip()
                return text[:4000]
            return f"HTTP {r.status_code}"
        except Exception as e:
            return f"Error: {str(e)}"
    
    def execute_python(self, code: str) -> str:
        """Execute Python code safely"""
        # Restricted globals
        safe_builtins = {
            'len': len, 'range': range, 'enumerate': enumerate,
            'zip': zip, 'map': map, 'filter': filter,
            'sum': sum, 'min': min, 'max': max, 'abs': abs,
            'round': round, 'str': str, 'int': int, 'float': float,
            'list': list, 'dict': dict, 'set': set, 'tuple': tuple,
            'print': lambda *args: ' '.join(str(a) for a in args),
            'type': type, 'isinstance': isinstance,
        }
        
        import io
        import sys
        
        stdout = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = stdout
        
        try:
            exec(code, {"__builtins__": safe_builtins}, {})
            sys.stdout = old_stdout
            output = stdout.getvalue()
            return output if output else "(No output)"
        except Exception as e:
            sys.stdout = old_stdout
            return f"Error: {str(e)}"
    
    def create_csv(self, data: List[Dict]) -> bytes:
        """Create CSV file"""
        df = pd.DataFrame(data)
        return df.to_csv(index=False).encode('utf-8')
    
    def create_excel(self, data: List[Dict]) -> bytes:
        """Create Excel file"""
        df = pd.DataFrame(data)
        buffer = io.BytesIO()
        df.to_excel(buffer, index=False, engine='openpyxl')
        return buffer.getvalue()

# ==================== AUTONOMOUS AGENT ====================

class AutonomousAgent:
    """Main agent class"""
    
    def __init__(self, config: AgentConfig):
        self.config = config
        self.llm = GroqLLM(config.groq_api_key, config.model)
        self.tools = Tools(config.serpapi_key)
        self.memory = AgentMemory()
        self.logs = []
    
    def log(self, message: str, level: str = "info"):
        """Add log entry"""
        entry = {
            'time': datetime.now().strftime("%H:%M:%S"),
            'level': level,
            'message': message
        }
        self.logs.append(entry)
        
        # Display immediately
        css_class = f"log-{level}"
        st.markdown(f'<div class="log-entry {css_class}">[{entry["time"]}] {message}</div>', 
                   unsafe_allow_html=True)
    
    def plan(self, task: str) -> List[Dict]:
        """Create execution plan"""
        self.log("🧠 Analyzing task...", "info")
        
        context = self.memory.get_recent_context()
        
        prompt = f"""Break down this task into steps. Return ONLY JSON array.

Context: {context}

Task: {task}

Format: [{{"step": 1, "action": "...", "tool": "web_search|fetch_url|execute_python|create_csv|create_excel|none"}}]

Available tools:
- web_search: Search internet
- fetch_url: Get webpage content
- execute_python: Run Python code
- create_csv: Make CSV file
- create_excel: Make Excel file
- none: Just use LLM

Respond with ONLY valid JSON array."""

        response = self.llm.generate(prompt, "You are a task planner. Output only JSON.")
        
        try:
            match = re.search(r'\[.*\]', response, re.DOTALL)
            if match:
                plan = json.loads(match.group())
                self.log(f"✅ Plan: {len(plan)} steps", "success")
                return plan
        except:
            pass
        
        # Fallback
        return [{"step": 1, "action": task, "tool": "none"}]
    
    def execute_step(self, step: Dict) -> str:
        """Execute one step"""
        action = step.get('action', '')
        tool = step.get('tool', 'none')
        
        self.log(f"▶️ Step {step.get('step', '?')}: {action[:60]}...", "info")
        
        try:
            if tool == 'web_search':
                query = re.sub(r'^(search|find|look up)\s+', '', action, flags=re.I)
                return self.tools.web_search(query)
            
            elif tool == 'fetch_url':
                urls = re.findall(r'https?://\S+', action)
                if urls:
                    return self.tools.fetch_url(urls[0])
                return "No URL found"
            
            elif tool == 'execute_python':
                # Extract code or generate it
                code_match = re.search(r'```python\s*(.*?)\s*```', action, re.DOTALL)
                if code_match:
                    code = code_match.group(1)
                else:
                    code_prompt = f"Write Python code for: {action}\nReturn ONLY code, no explanation."
                    code = self.llm.generate(code_prompt, "You write Python code.")
                
                return self.tools.execute_python(code)
            
            elif tool == 'create_csv':
                # Generate sample data
                data_prompt = f"Generate JSON array of data for: {action}\nFormat: [{{'col1': 'val1', ...}}]"
                response = self.llm.generate(data_prompt)
                
                try:
                    match = re.search(r'\[.*\]', response, re.DOTALL)
                    if match:
                        data = json.loads(match.group())
                        csv_bytes = self.tools.create_csv(data)
                        self.memory.store_file('output.csv', csv_bytes, 'text/csv')
                        return f"✅ CSV created with {len(data)} rows"
                except Exception as e:
                    return f"CSV error: {str(e)}"
            
            elif tool == 'create_excel':
                data_prompt = f"Generate JSON array of data for: {action}\nFormat: [{{'col1': 'val1', ...}}]"
                response = self.llm.generate(data_prompt)
                
                try:
                    match = re.search(r'\[.*\]', response, re.DOTALL)
                    if match:
                        data = json.loads(match.group())
                        excel_bytes = self.tools.create_excel(data)
                        self.memory.store_file('output.xlsx', excel_bytes, 
                            'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
                        return f"✅ Excel created with {len(data)} rows"
                except Exception as e:
                    return f"Excel error: {str(e)}"
            
            else:
                # No tool - direct LLM
                return self.llm.generate(action)
                
        except Exception as e:
            self.log(f"❌ Error: {str(e)}", "error")
            return f"Error: {str(e)}"
    
    def execute(self, task: str) -> str:
        """Execute full task"""
        self.logs = []
        self.log(f"🚀 Starting: {task[:80]}...", "info")
        
        self.memory.add_message("user", task)
        
        # Plan
        plan = self.plan(task)
        
        with st.expander("📋 Execution Plan", expanded=True):
            for p in plan:
                st.write(f"**{p['step']}.** {p['action']} *(tool: {p['tool']})*")
        
        # Execute
        results = []
        progress_bar = st.progress(0)
        
        for i, step in enumerate(plan):
            progress = (i + 1) / len(plan)
            progress_bar.progress(min(progress, 0.99))
            
            result = self.execute_step(step)
            results.append({'step': step['step'], 'result': result})
            
            # Check for errors
            if 'Error' in result and 'rate limit' not in result.lower():
                self.log("🔄 Retrying with simpler approach...", "warning")
                step['action'] += " (simpler version)"
                result = self.execute_step(step)
                results[-1]['result'] = result
        
        progress_bar.empty()
        
        # Summary
        self.log("📝 Generating summary...", "info")
        
        summary_prompt = f"""Summarize these results concisely:

Task: {task}

Results: {json.dumps(results, indent=2)}

Provide key findings and any recommendations."""
        
        summary = self.llm.generate(summary_prompt, "You summarize task results.")
        self.memory.add_task(task, summary)
        self.memory.add_message("assistant", summary)
        
        self.log("✅ Complete!", "success")
        
        return summary

# ==================== MULTI-AGENT ====================

class MultiAgent:
    """Parallel agent execution"""
    
    def __init__(self, config: AgentConfig):
        self.config = config
    
    def run(self, task: str) -> Dict[str, str]:
        """Execute with multiple parallel agents"""
        
        # Break down task
        llm = GroqLLM(self.config.groq_api_key, self.config.model)
        prompt = f"""Split this task into 2-3 parallel subtasks. Return JSON array.

Task: {task}

Format: [{{"subtask": "description", "focus": "what to research"}}]"""

        response = llm.generate(prompt)
        
        try:
            match = re.search(r'\[.*\]', response, re.DOTALL)
            subtasks = json.loads(match.group()) if match else [{"subtask": task}]
        except:
            subtasks = [{"subtask": task}]
        
        # Execute in parallel
        results = {}
        
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = {}
            
            for st in subtasks[:3]:
                agent = AutonomousAgent(self.config)
                future = executor.submit(agent.execute, st['subtask'])
                futures[future] = st['subtask']
            
            for future in as_completed(futures):
                subtask = futures[future]
                try:
                    results[subtask] = future.result()
                except Exception as e:
                    results[subtask] = f"Error: {str(e)}"
        
        return results

# ==================== UI ====================

def sidebar():
    """Sidebar configuration"""
    with st.sidebar:
        st.title("⚙️ Settings")
        
        # API Keys
        st.subheader("API Keys (Free)")
        
        groq_key = st.text_input(
            "Groq API Key",
            type="password",
            value=st.secrets.get("GROQ_API_KEY", "") if hasattr(st, 'secrets') else "",
            help="Get FREE key from groq.com (1M tokens/day)"
        )
        
        serpapi_key = st.text_input(
            "SerpAPI Key (Optional)",
            type="password",
            value=st.secrets.get("SERPAPI_KEY", "") if hasattr(st, 'secrets') else "",
            help="100 free searches/month at serpapi.com"
        )
        
        st.divider()
        
        # Model
        model = st.selectbox(
            "Model",
            ["llama-3.1-70b-versatile", "llama-3.1-8b-instant", 
             "mixtral-8x7b-32768", "gemma2-9b-it"],
            help="70B = best quality, 8B = faster"
        )
        
        temp = st.slider("Creativity", 0.0, 1.0, 0.7)
        
        use_multi = st.toggle("Multi-Agent Mode", value=False)
        
        st.divider()
        
        # About
        st.caption("""
        **Free Tier Limits:**
        - Groq: 1M tokens/day
        - DuckDuckGo: Unlimited
        - SerpAPI: 100/day (optional)
        """)
        
        return AgentConfig(
            groq_api_key=groq_key,
            serpapi_key=serpapi_key,
            model=model,
            temperature=temp
        ), use_multi

def main():
    """Main app"""
    st.markdown('<p class="main-header">🤖 Autonomous AI Agent</p>', unsafe_allow_html=True)
    st.caption("Powered by Groq FREE API | No OpenAI/Anthropic required")
    
    config, use_multi = sidebar()
    
    # Check API key
    if not config.groq_api_key:
        st.warning("⚠️ Please enter your Groq API key in the sidebar")
        st.info("""
        **Get FREE API Key:**
        1. Go to [groq.com](https://groq.com)
        2. Sign up (free, no credit card)
        3. Create API key at [console.groq.com/keys](https://console.groq.com/keys)
        4. Paste key in sidebar
        """)
        return
    
    # Task input
    st.subheader("📝 Your Task")
    task = st.text_area(
        "What should I do?",
        placeholder="Examples:\n• Research top 5 AI companies and create a CSV\n• Write Python to calculate prime numbers\n• Search for renewable energy stats and summarize",
        height=100
    )
    
    col1, col2 = st.columns([1, 4])
    with col1:
        run_btn = st.button("🚀 Execute", type="primary", use_container_width=True)
    
    # Execute
    if run_btn and task:
        st.divider()
        st.subheader("📊 Execution Log")
        
        log_container = st.container()
        
        with log_container:
            if use_multi:
                st.info("🐝 Running multi-agent swarm...")
                multi = MultiAgent(config)
                results = multi.run(task)
                
                st.subheader("📋 Combined Results")
                for subtask, result in results.items():
                    with st.expander(f"Agent: {subtask[:60]}..."):
                        st.write(result)
            else:
                agent = AutonomousAgent(config)
                result = agent.execute(task)
                
                st.subheader("📋 Final Result")
                st.markdown(result)
        
        # File downloads
        memory = AgentMemory()
        if 'output.csv' in memory.memory['files']:
            f = memory.memory['files']['output.csv']
            st.download_button("📥 Download CSV", f['data'], "output.csv", "text/csv")
        
        if 'output.xlsx' in memory.memory['files']:
            f = memory.memory['files']['output.xlsx']
            st.download_button("📥 Download Excel", f['data'], "output.xlsx",
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

if __name__ == "__main__":
    main()
