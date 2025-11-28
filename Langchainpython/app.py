"""LangChain Tool-Calling Chat Application - All in One File"""

# ============================================================================
# IMPORTS
# ============================================================================
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from typing import Any, Dict, List, Optional
import os
import re
import random
import traceback
import socket
import requests
from dotenv import load_dotenv

# Try to import agent creation functions
# If not available, we'll use a fallback implementation
create_openai_tools_agent = None
AgentExecutor = None

try:
    from langchain.agents import create_openai_tools_agent, AgentExecutor
except ImportError:
    # Try to import AgentExecutor separately
    try:
        from langchain.agents import AgentExecutor
    except ImportError:
        try:
            from langchain_core.agents import AgentExecutor
        except ImportError:
            # Will use fallback implementation
            pass

# The legacy agent factory is implemented inline in this file (legacy compatibility)

load_dotenv()

# ============================================================================
# FLASK APP INITIALIZATION
# ============================================================================
app = Flask(__name__)
CORS(app)

# ============================================================================
# LANGCHAIN TOOLS
# ============================================================================

@tool
def positive_prompt_tool(query: str) -> str:
    """
    Use this tool when the user wants positive encouragement, motivation, 
    uplifting messages, or needs a positive perspective on something.
    
    Args:
        query: The user's query or message
        
    Returns:
        A positive, encouraging response
    """
    positive_responses = [
        "You're doing great! Keep up the excellent work!",
        "Remember, every challenge is an opportunity to grow. You've got this!",
        "Your efforts are making a difference. Stay positive and keep moving forward!",
        "You have the strength and capability to overcome any obstacle. Believe in yourself!",
        "Today is a new day full of possibilities. Make it count!"
    ]
    
    query_lower = query.lower()
    if "work" in query_lower or "job" in query_lower:
        return "You're making progress in your work! Your dedication will pay off. Keep pushing forward!"
    elif "study" in query_lower or "exam" in query_lower or "test" in query_lower:
        return "You're putting in the effort to learn and grow. Your hard work in studying will lead to success!"
    elif "relationship" in query_lower or "friend" in query_lower:
        return "Your relationships matter, and you're capable of building meaningful connections!"
    else:
        return random.choice(positive_responses)


@tool
def negative_prompt_tool(query: str) -> str:
    """
    Use this tool when the user expresses negative thoughts, complaints, 
    frustrations, or needs help processing negative emotions.
    
    Args:
        query: The user's query or message
        
    Returns:
        A supportive response that acknowledges their feelings
    """
    query_lower = query.lower()
    
    if "tired" in query_lower or "exhausted" in query_lower:
        return "I understand you're feeling tired. It's okay to rest and take care of yourself. Remember to prioritize your well-being."
    elif "frustrated" in query_lower or "angry" in query_lower:
        return "It's completely normal to feel frustrated sometimes. These feelings are valid. Take a deep breath and know that this feeling will pass."
    elif "sad" in query_lower or "down" in query_lower:
        return "I'm sorry you're feeling down. Your feelings are important and valid. Remember that difficult times don't last forever."
    elif "stressed" in query_lower or "anxious" in query_lower:
        return "Stress and anxiety can be overwhelming. Try to take things one step at a time. You're stronger than you think."
    else:
        return "I hear you, and your feelings are valid. It's okay to not be okay sometimes. Remember that you're not alone, and things can get better."


@tool
def student_marks_tool(query: str) -> str:
    """
    Use this tool when the user asks about grades, marks, scores, academic performance, 
    or anything related to student assessments.
    
    Args:
        query: The user's query about marks/grades
        
    Returns:
        Information or advice about student marks
    """
    query_lower = query.lower()
    numbers = re.findall(r'\d+', query)
    
    if "grade" in query_lower or "mark" in query_lower or "score" in query_lower:
        if numbers:
            score = int(numbers[0])
            if score >= 90:
                return f"Excellent! A score of {score} shows outstanding performance. Keep up the great work!"
            elif score >= 80:
                return f"Great job! A score of {score} is a solid performance. You're doing well!"
            elif score >= 70:
                return f"Good work! A score of {score} shows you're on the right track. There's room for improvement, but you're making progress!"
            elif score >= 60:
                return f"A score of {score} indicates you're passing, but there's definitely room for improvement. Consider reviewing the material and seeking help if needed."
            else:
                return f"A score of {score} suggests you may need additional support. Don't be discouraged - this is a learning opportunity. Consider talking to your teacher or tutor."
        else:
            return "I can help you understand your marks! If you share your score, I can provide feedback. Remember, grades are just one measure of learning - focus on understanding and growth."
    elif "improve" in query_lower or "better" in query_lower:
        return "To improve your marks, try these strategies: 1) Review your mistakes and learn from them, 2) Create a study schedule, 3) Ask questions when you don't understand, 4) Practice regularly, 5) Get enough rest before exams."
    else:
        return "I'm here to help with questions about your academic performance. Share your marks or concerns, and I can provide guidance and encouragement!"


@tool
def suicide_related_tool(query: str) -> str:
    """
    Use this tool when the user mentions suicide, self-harm, ending their life, 
    or expresses severe hopelessness or despair.
    
    IMPORTANT: This tool provides immediate support resources and should be used 
    for any suicide-related queries.
    
    Args:
        query: The user's query that may indicate suicidal thoughts
        
    Returns:
        Supportive response with crisis resources
    """
    crisis_resources = """
    I'm deeply concerned about what you're going through. Your life has value, and there are people who want to help.

    If you're in immediate danger, please call emergency services right away:
    - USA: 988 Suicide & Crisis Lifeline (call or text)
    - International: Find your local crisis line at https://www.iasp.info/resources/Crisis_Centres/

    You don't have to go through this alone. Please reach out to:
    - A trusted friend or family member
    - A mental health professional
    - A crisis helpline
    - Your doctor or healthcare provider

    These feelings, while overwhelming, can be managed with the right support. Please don't hesitate to seek help.
    """
    
    return crisis_resources


# ============================================================================
# AGENT SETUP AND MEMORY
# ============================================================================

# Simple in-memory chat history
chat_history = []

# Store agent instance
agent_executor = None


# ------------------------- legacy compatibility (inlined) -------------------------
class LegacyConversationBufferMemory:
    """
    Simple conversation buffer memory compatible with older LangChain versions.
    """
    def __init__(self, memory_key: str = "chat_history", k: int = 10):
        self.memory_key = memory_key
        self.k = max(1, int(k))
        self.chat_history: List[Dict[str, str]] = []

    def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        formatted = []
        for item in self.chat_history[-self.k:]:
            role = item.get("role", "human")
            content = item.get("content", "")
            if role == "human":
                formatted.append(f"Human: {content}")
            else:
                formatted.append(f"Assistant: {content}")
        return {self.memory_key: "\n".join(formatted)}

    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, Any]) -> None:
        user_input = inputs.get("input") or inputs.get("message") or ""
        assistant_output = outputs.get("output") or outputs.get("response") or ""
        if user_input:
            self.chat_history.append({"role": "human", "content": str(user_input)})
        if assistant_output:
            self.chat_history.append({"role": "assistant", "content": str(assistant_output)})
        max_items = self.k * 2
        if len(self.chat_history) > max_items:
            self.chat_history = self.chat_history[-max_items:]

    def clear(self) -> None:
        self.chat_history = []


class LegacyAgentExecutor:
    def __init__(self, llm, tools: List[Any], prompt: Any, memory: Optional[LegacyConversationBufferMemory] = None):
        self.llm = llm
        self.prompt = prompt
        self.tools = {t.name: t for t in tools}
        self.memory = memory or LegacyConversationBufferMemory()

    def _invoke_tool(self, tool, arg):
        if hasattr(tool, "invoke"):
            return tool.invoke(arg)
        else:
            return tool(arg)

    def invoke(self, inputs: Dict[str, Any]):
        try:
            user_input = inputs.get("input") or inputs.get("message") or ""
            memory_vars = self.memory.load_memory_variables(inputs)
            memory_text = memory_vars.get(self.memory.memory_key, "")
            ml_messages = []
            system_content = None
            try:
                if hasattr(self.prompt, "to_messages"):
                    prompt_messages = self.prompt.to_messages({"input": user_input, "chat_history": memory_text})
                    for pmsg in prompt_messages:
                        if isinstance(pmsg, SystemMessage):
                            system_content = pmsg.content
            except Exception:
                try:
                    system_content = str(self.prompt)
                except Exception:
                    system_content = "You are a helpful AI assistant."
            if system_content:
                ml_messages.append(SystemMessage(content=system_content))
            if memory_text:
                for line in memory_text.split("\n"):
                    if line.startswith("Human:"):
                        ml_messages.append(HumanMessage(content=line.replace("Human:", "").strip()))
                    elif line.startswith("Assistant:"):
                        ml_messages.append(AIMessage(content=line.replace("Assistant:", "").strip()))
                    else:
                        ml_messages.append(HumanMessage(content=line))
            ml_messages.append(HumanMessage(content=user_input))
            response = self.llm.invoke(ml_messages)
            if hasattr(response, "tool_calls") and response.tool_calls:
                tool_results = []
                for tc in response.tool_calls:
                    name = tc.get("name")
                    args = tc.get("args")
                    if name in self.tools:
                        try:
                            if isinstance(args, dict) and "query" in args:
                                tr = self._invoke_tool(self.tools[name], args["query"])
                            elif isinstance(args, str):
                                tr = self._invoke_tool(self.tools[name], args)
                            else:
                                tr = self._invoke_tool(self.tools[name], str(args))
                            tool_results.append(str(tr))
                        except Exception as e:
                            tool_results.append(f"Error: {str(e)}")
                    else:
                        tool_results.append(f"Error: Unknown tool {name}")
                ml_messages.append(HumanMessage(content="\n".join(tool_results)))
                final_response = self.llm.invoke(ml_messages)
                out_text = final_response.content if hasattr(final_response, "content") else str(final_response)
            else:
                out_text = response.content if hasattr(response, "content") else str(response)
            try:
                self.memory.save_context({"input": user_input}, {"output": out_text})
            except Exception:
                pass
            return {"output": out_text}
        except Exception as ex:
            tb = traceback.format_exc()
            return {"output": f"Error: {type(ex).__name__}: {str(ex)}\n{tb}"}


def create_legacy_agent_executor(llm, tools: List[Any], prompt: Any, memory_size: int = 10) -> LegacyAgentExecutor:
    mem = LegacyConversationBufferMemory(k=memory_size)
    return LegacyAgentExecutor(llm=llm, tools=tools, prompt=prompt, memory=mem)

# ---------------------- end legacy compatibility (inlined) ----------------------


def create_agent():
    """Create and configure the LangChain agent with tools."""
    
    # Initialize the LLM
    llm = ChatOpenAI(
        model="gpt-4o-mini",  # Using gpt-4o-mini for cost efficiency
        temperature=0.7,
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )
    
    # Define all tools
    tools = [
        positive_prompt_tool,
        negative_prompt_tool,
        student_marks_tool,
        suicide_related_tool
    ]
    
    # Create prompt template that guides the agent to use tools
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful AI assistant that routes user queries to the appropriate tool based on semantic relevance.

Available tools:
1. positive_prompt_tool - Use for positive encouragement, motivation, uplifting messages
2. negative_prompt_tool - Use for negative thoughts, complaints, frustrations, processing negative emotions
3. student_marks_tool - Use for questions about grades, marks, scores, academic performance
4. suicide_related_tool - Use for ANY mention of suicide, self-harm, ending life, or severe hopelessness

IMPORTANT: 
- Analyze the user's query semantically to determine which tool is most relevant
- If the query relates to suicide or self-harm, ALWAYS use suicide_related_tool
- Be empathetic and helpful in your responses
- Use the tools when appropriate, but you can also respond directly for general conversation
- Maintain context from previous messages in the conversation"""),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad")
    ])
    
    # Create the agent - try different methods based on what's available
    # Support a legacy agent mode via environment variable (USE_LEGACY_AGENT)
    use_legacy = os.getenv("USE_LEGACY_AGENT", "").lower() in ("1", "true", "yes")
    if use_legacy and create_legacy_agent_executor is not None:
        try:
            print("Using LegacyAgentExecutor (USE_LEGACY_AGENT is enabled)")
            agent_executor = create_legacy_agent_executor(llm, tools, prompt, memory_size=int(os.getenv('LEGACY_MEMORY_SIZE', '10')))
            return agent_executor
        except Exception as e:
            print(f"Error creating legacy agent executor: {e}")
    if create_openai_tools_agent is not None:
        try:
            # Use the standard agent creation
            agent = create_openai_tools_agent(llm, tools, prompt)
            agent_executor = AgentExecutor(
                agent=agent,
                tools=tools,
                verbose=True,
                handle_parsing_errors=True
            )
            return agent_executor
        except Exception as e:
            print(f"Error creating agent with create_openai_tools_agent: {e}")
    
    # Fallback: Use LLM with tools bound and handle tool calling manually
    # Create a simple wrapper that acts like an agent executor
    class SimpleAgentExecutor:
        def __init__(self, llm, tools, prompt):
            self.llm = llm.bind_tools(tools)
            self.tools = {tool.name: tool for tool in tools}
            self.prompt = prompt
            
        def invoke(self, inputs):
            chat_history = inputs.get("chat_history", [])
            user_input = inputs.get("input", "")
            
            # Format messages with system prompt
            messages = [
                ("system", """You are a helpful AI assistant that routes user queries to the appropriate tool based on semantic relevance.

Available tools:
1. positive_prompt_tool - Use for positive encouragement, motivation, uplifting messages
2. negative_prompt_tool - Use for negative thoughts, complaints, frustrations, processing negative emotions
3. student_marks_tool - Use for questions about grades, marks, scores, academic performance
4. suicide_related_tool - Use for ANY mention of suicide, self-harm, ending life, or severe hopelessness

IMPORTANT: 
- Analyze the user's query semantically to determine which tool is most relevant
- If the query relates to suicide or self-harm, ALWAYS use suicide_related_tool
- Be empathetic and helpful in your responses
- Use the tools when appropriate, but you can also respond directly for general conversation
- Maintain context from previous messages in the conversation""")
            ]
            
            # Add chat history
            for msg in chat_history:
                if isinstance(msg, HumanMessage):
                    messages.append(("human", msg.content))
                elif isinstance(msg, AIMessage):
                    messages.append(("assistant", msg.content))
            
            # Add current user input
            messages.append(("human", user_input))
            
            # Get response from LLM
            from langchain_core.messages import SystemMessage
            formatted_messages = []
            for role, content in messages:
                if role == "system":
                    formatted_messages.append(SystemMessage(content=content))
                elif role == "human":
                    formatted_messages.append(HumanMessage(content=content))
                elif role == "assistant":
                    formatted_messages.append(AIMessage(content=content))
            
            try:
                response = self.llm.invoke(formatted_messages)
            except Exception as e:
                # If LLM call fails due to network error, raise it
                error_type = type(e).__name__
                if "Connection" in error_type or "Network" in error_type or "timeout" in str(e).lower() or "API" in error_type:
                    raise e
                else:
                    return {"output": f"Error processing request: {str(e)}"}
            
            # Check if tool calls are needed
            if hasattr(response, 'tool_calls') and response.tool_calls:
                # Execute tool calls
                tool_results = []
                for tool_call in response.tool_calls:
                    tool_name = tool_call.get('name', '')
                    tool_args = tool_call.get('args', {})
                    if tool_name in self.tools:
                        try:
                            # Handle different tool argument formats
                            if isinstance(tool_args, dict) and 'query' in tool_args:
                                tool_result = self.tools[tool_name].invoke(tool_args['query'])
                            elif isinstance(tool_args, str):
                                tool_result = self.tools[tool_name].invoke(tool_args)
                            else:
                                tool_result = self.tools[tool_name].invoke(str(tool_args))
                            # Append only the tool result (do not include the tool name)
                            tool_results.append(str(tool_result))
                        except Exception as e:
                            # Record the error message only (no tool name)
                            tool_results.append(f"Error: {str(e)}")
                
                # Get final response with tool results
                formatted_messages.append(response)
                formatted_messages.append(HumanMessage(content="\n".join(tool_results)))
                try:
                    final_response = self.llm.invoke(formatted_messages)
                    return {"output": final_response.content if hasattr(final_response, 'content') else str(final_response)}
                except Exception as e:
                    # If second LLM call fails, return tool results directly
                    return {"output": "\n".join(tool_results)}
            else:
                return {"output": response.content if hasattr(response, 'content') else str(response)}
    
    return SimpleAgentExecutor(llm, tools, prompt)


def check_internet_connection():
    """Check if internet connection is available."""
    try:
        # Try to connect to a reliable server
        socket.create_connection(("8.8.8.8", 53), timeout=3)
        return True
    except OSError:
        try:
            # Try Google DNS
            requests.get("https://www.google.com", timeout=3)
            return True
        except:
            return False


def check_openai_reachability():
    """Check that we can connect to OpenAI's models endpoint with the provided API key.

    Returns a dict with keys: reachable (bool), status (int|None), error (str|None).
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return {"reachable": False, "status": None, "error": "no_api_key"}
    headers = {"Authorization": f"Bearer {api_key}"}
    try:
        r = requests.get("https://api.openai.com/v1/models", headers=headers, timeout=10)
        return {"reachable": True if r.status_code == 200 else False, "status": r.status_code, "error": None if r.status_code == 200 else r.text[:200]}
    except Exception as e:
        return {"reachable": False, "status": None, "error": f"{type(e).__name__}: {str(e)}"}


@app.route('/api/debug', methods=['GET'])
def debug():
    """Return basic diagnostics: env key presence, internet connectivity, OpenAI reachability."""
    has_key = bool(os.getenv("OPENAI_API_KEY"))
    internet = check_internet_connection()
    openai_check = check_openai_reachability()
    return jsonify({
        "api_key_set": has_key,
        "internet": internet,
        "openai": openai_check
    })


# Offline/local tool routing removed per user request: the app now requires the OpenAI API
# and an active internet connection to operate. Local tool fallbacks were removed to
# simplify behavior and avoid inaccuracies from simplified heuristics.


def get_agent_response(agent_executor, user_input):
    """Get response from the agent with memory.

    The application now requires an OpenAI API key and an active internet
    connection; local/offline fallback behavior has been removed.
    """
    global chat_history

    # Check if API key is set
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or api_key == "your_openai_api_key_here":
        return "⚠️ No API key found. Please set OPENAI_API_KEY in your .env to enable AI features."

    # Check internet connection
    if not check_internet_connection():
        return "📡 No internet connection detected. Please connect to the internet to use this app."

    # Try to invoke the agent with the provided executor
    try:
        response = agent_executor.invoke({
            "input": user_input,
            "chat_history": chat_history
        })
        output = response.get("output", "I apologize, but I couldn't process that request.")
        # Save the conversation to memory
        chat_history.append(HumanMessage(content=user_input))
        chat_history.append(AIMessage(content=output))
        return output
    except Exception as e:
        # Log and return a generic error message; do not fallback to local tools
        print(f"Agent invocation error: {type(e).__name__}: {str(e)}")
        return f"❌ Error: {type(e).__name__}: {str(e)}"


def clear_agent_memory():
    """Clear the conversation memory."""
    global chat_history
    chat_history = []


def initialize_agent():
    """Initialize the agent on first use."""
    global agent_executor
    if agent_executor is None:
        agent_executor = create_agent()


# ============================================================================
# FLASK ROUTES
# ============================================================================

@app.route('/')
def index():
    """Serve the main chat UI."""
    return render_template('index.html')


@app.route('/api/chat', methods=['POST'])
def chat():
    """Handle chat messages."""
    try:
        data = request.json
        user_message = data.get('message', '').strip()
        
        if not user_message:
            return jsonify({'error': 'Message cannot be empty'}), 400
        
        # Initialize agent if not already done
        initialize_agent()
        
        # Get response from agent
        response = get_agent_response(agent_executor, user_message)
        
        return jsonify({
            'response': response,
            'status': 'success'
        })
    
    except Exception as e:
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 500


@app.route('/api/clear', methods=['POST'])
def clear_memory():
    """Clear conversation memory."""
    try:
        clear_agent_memory()
        return jsonify({'status': 'success', 'message': 'Memory cleared'})
    except Exception as e:
        return jsonify({'error': str(e), 'status': 'error'}), 500


@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({'status': 'healthy'})

@app.route('/api/history', methods=['GET'])
def get_history():
    global chat_history
    return jsonify({
        "history": [msg.content for msg in chat_history]
    })

# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

if __name__ == '__main__':
    # Check if API key is set
    if not os.getenv("OPENAI_API_KEY"):
        print("WARNING: OPENAI_API_KEY not found in environment variables.")
        print("Please create a .env file with: OPENAI_API_KEY=your_key_here")
    

    app.run(debug=True, port=5000)
