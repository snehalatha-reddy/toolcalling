"""LangChain Tool-Calling Chat Application - All in One File"""

# ============================================================================
# IMPORTS
# ============================================================================
# Flask for backend API handling and rendering pages
from flask import Flask, request, jsonify, render_template

# Allows frontend (React/HTML) to call this backend without CORS issues
from flask_cors import CORS

# LangChain tools, LLM model, prompt templates, and message types
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

# Additional Python utilities
from typing import Any, Dict, List, Optional
import os
import re
import random
import traceback
import socket
import requests
from dotenv import load_dotenv

# Try importing agent creation functions (newer LangChain versions)
# If not found, fallback logic will be used later
create_openai_tools_agent = None
AgentExecutor = None

try:
    from langchain.agents import create_openai_tools_agent, AgentExecutor
except ImportError:
    # LangChain version might be older — try other import paths
    try:
        from langchain.agents import AgentExecutor
    except ImportError:
        try:
            from langchain_core.agents import AgentExecutor
        except ImportError:
            pass  # If all imports fail, fallback manual agent will be used

# Load environment variables from .env file (OpenAI Key)
load_dotenv()

# ============================================================================
# FLASK APP INITIALIZATION
# ============================================================================
app = Flask(__name__)
CORS(app)  # Enable cross-origin requests

# ============================================================================
# LANGCHAIN TOOLS (Custom Tools)
# ============================================================================
# These tools are used when the AI decides a message fits a category.
# LangChain agent will automatically choose the correct tool.


@tool
def positive_prompt_tool(query: str) -> str:
    #positive prompt tool
    """Returns motivational and positive messages."""
    positive_responses = [
        "You're doing great! Keep up the excellent work!",
        "Remember, every challenge is an opportunity to grow. You've got this!",
        "Your efforts are making a difference. Stay positive and keep moving forward!",
        "You have the strength and capability to overcome any obstacle. Believe in yourself!",
        "Today is a new day full of possibilities. Make it count!"
    ]

    query_lower = query.lower()

    # Special cases—custom responses depending on topic
    if "work" in query_lower or "job" in query_lower:
        return "You're making progress in your work! Your dedication will pay off."
    elif "study" in query_lower or "exam" in query_lower:
        return "You're working hard in studies. That effort will bring success!"
    elif "relationship" in query_lower or "friend" in query_lower:
        return "You're capable of building strong and meaningful relationships!"
    else:
        return random.choice(positive_responses)


@tool
def negative_prompt_tool(query: str) -> str:
    #negative prompt tool
    """Handles negative feelings—sadness, stress, anger, anxiety."""
    query_lower = query.lower()

    if "tired" in query_lower or "exhausted" in query_lower:
        return "It's okay to feel tired. Rest and take care of yourself."
    elif "frustrated" in query_lower or "angry" in query_lower:
        return "Frustration is normal. Take a deep breath—this moment will pass."
    elif "sad" in query_lower or "down" in query_lower:
        return "I'm sorry you're feeling sad. Difficult times do not last forever."
    elif "stressed" in query_lower or "anxious" in query_lower:
        return "Stress can feel overwhelming, but you're stronger than you think."
    else:
        return "Your feelings are valid. It's okay to not feel okay sometimes."


@tool
def student_marks_tool(query: str) -> str:
    #student marks tool
    """Handles academic-related queries such as marks, grades, scores."""
    query_lower = query.lower()
    numbers = re.findall(r'\d+', query)  # Extract marks from message

    # If user mentions marks or grades
    if "grade" in query_lower or "mark" in query_lower or "score" in query_lower:
        if numbers:
            score = int(numbers[0])
            # Provide feedback based on mark range
            if score >= 90:
                return f"Excellent! {score} is an outstanding score!"
            elif score >= 80:
                return f"Great job! {score} shows strong performance."
            elif score >= 70:
                return f"Good work! {score} shows progress and improvement opportunity."
            elif score >= 60:
                return f"{score} is a pass, but you can do even better with practice!"
            else:
                return f"{score} indicates you may need extra support. Don’t be discouraged!"
        else:
            return "Share your score and I can help you understand it!"

    # If user asks how to improve marks
    elif "improve" in query_lower or "better" in query_lower:
        return "To improve marks: revise mistakes, practice regularly, ask questions, and study consistently."

    return "Tell me your marks or study questions, and I can help!"


@tool
def suicide_related_tool(query: str) -> str:
    #sucide tool
    """Provides crisis support when suicide or self-harm is mentioned."""
    crisis_resources = """
    I'm really concerned about you. Your life is valuable and important.

    If you're in immediate danger, please call your local emergency number.

    USA: 988 Suicide & Crisis Lifeline  
    International helplines: https://www.iasp.info/resources/Crisis_Centres/

    Talk to someone you trust or a mental health professional. You are not alone.
    """
    return crisis_resources


# ============================================================================
# AGENT MEMORY SETUP
# ============================================================================
# A simple list storing chat history (both user messages & AI replies)
chat_history = []

# Will hold the agent instance after initialization
agent_executor = None


# ============================================================================
# LEGACY MEMORY + AGENT (Compatibility for older LangChain versions)
# ============================================================================
class LegacyConversationBufferMemory:
    """Stores last k messages to maintain conversation context."""
    def __init__(self, memory_key="chat_history", k=10):
        self.memory_key = memory_key
        self.k = k
        self.chat_history = []

    def load_memory_variables(self, inputs):
        """Returns last k messages as a formatted text block."""
        formatted = []
        for item in self.chat_history[-self.k:]:
            role = "Human" if item["role"] == "human" else "Assistant"
            formatted.append(f"{role}: {item['content']}")
        return {self.memory_key: "\n".join(formatted)}

    def save_context(self, inputs, outputs):
        """Saves human + AI messages into memory."""
        user_msg = inputs.get("input", "")
        ai_msg = outputs.get("output", "")

        if user_msg:
            self.chat_history.append({"role": "human", "content": user_msg})
        if ai_msg:
            self.chat_history.append({"role": "assistant", "content": ai_msg})

        # Limit memory size
        if len(self.chat_history) > self.k * 2:
            self.chat_history = self.chat_history[-self.k * 2:]


class LegacyAgentExecutor:
    """Fallback agent executor if modern agent cannot load."""
    def __init__(self, llm, tools, prompt, memory=None):
        self.llm = llm
        self.tools = {t.name: t for t in tools}
        self.prompt = prompt
        self.memory = memory or LegacyConversationBufferMemory()

    def invoke(self, inputs):
        """Processes input → LLM → tool call → final reply."""
        try:
            user_input = inputs.get("input", "")

            # Load previous memory
            memory_vars = self.memory.load_memory_variables(inputs)
            memory_text = memory_vars["chat_history"]

            # Build chat messages for LLM
            messages = [SystemMessage(content=self.prompt.messages[0][1])]
            for line in memory_text.split("\n"):
                if line.startswith("Human:"):
                    messages.append(HumanMessage(content=line.replace("Human:", "").strip()))
                elif line.startswith("Assistant:"):
                    messages.append(AIMessage(content=line.replace("Assistant:", "").strip()))
            messages.append(HumanMessage(content=user_input))

            # Ask model for tool call or direct response
            response = self.llm.invoke(messages)

            # If the model requests a tool
            if hasattr(response, "tool_calls") and response.tool_calls:
                tool_results = []
                for call in response.tool_calls:
                    tool_name = call["name"]
                    tool_args = call["args"]

                    # Execute selected tool
                    result = self.tools[tool_name].invoke(tool_args["query"])
                    tool_results.append(result)

                # Ask model again with tool results
                messages.append(HumanMessage(content="\n".join(tool_results)))
                final = self.llm.invoke(messages)
                output = final.content
            else:
                output = response.content

            # Save to memory
            self.memory.save_context({"input": user_input}, {"output": output})
            return {"output": output}

        except Exception as exc:
            return {"output": f"Error: {exc}"}


# ============================================================================
# AGENT CREATION LOGIC
# ============================================================================
def create_agent():
    """Creates agent with tools, memory, and prompt."""
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.7,
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )

    # List of all available tools
    tools = [
        positive_prompt_tool,
        negative_prompt_tool,
        student_marks_tool,
        suicide_related_tool
    ]

    # System instructions for the agent
    prompt = ChatPromptTemplate.from_messages([
        ("system", """
You are a tool-routing AI assistant.

Tools:
- Use positive_prompt_tool for motivation
- Use negative_prompt_tool for sadness/stress
- Use student_marks_tool for academic questions
- ALWAYS use suicide_related_tool for suicide/self-harm queries

Follow rules and maintain context using chat history.
"""),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad")
    ])

    # Try creating modern agent first
    if create_openai_tools_agent:
        try:
            agent = create_openai_tools_agent(llm, tools, prompt)
            return AgentExecutor(agent=agent, tools=tools, handle_parsing_errors=True)
        except:
            pass

    # If modern agent fails — use fallback SimpleAgentExecutor
    class SimpleAgentExecutor:
        """Simple agent that manually routes tool calls."""
        def __init__(self, llm, tools, prompt):
            self.llm = llm.bind_tools(tools)
            self.tools = {tool.name: tool for tool in tools}
            self.prompt = prompt

        def invoke(self, inputs):
            chat_history = inputs.get("chat_history", [])
            user_input = inputs.get("input", "")

            # Build messages for LLM
            msgs = [SystemMessage(content="You route user queries to correct tools.")]
            for msg in chat_history:
                if isinstance(msg, HumanMessage):
                    msgs.append(HumanMessage(msg.content))
                else:
                    msgs.append(AIMessage(msg.content))

            msgs.append(HumanMessage(user_input))

            # First LLM call: decide tool
            response = self.llm.invoke(msgs)

            # Handle tool calls
            if hasattr(response, "tool_calls") and response.tool_calls:
                results = []
                for call in response.tool_calls:
                    tname = call["name"]
                    targs = call["args"]
                    result = self.tools[tname].invoke(targs["query"])
                    results.append(result)

                msgs.append(HumanMessage("\n".join(results)))
                final = self.llm.invoke(msgs)
                return {"output": final.content}

            return {"output": response.content}

    return SimpleAgentExecutor(llm, tools, prompt)


# ============================================================================
# INTERNET + API KEY CHECKING
# ============================================================================

def check_internet_connection():
    """Checks basic internet connectivity."""
    try:
        socket.create_connection(("8.8.8.8", 53), timeout=3)
        return True
    except:
        return False


def check_openai_reachability():
    """Checks if OpenAI API is reachable."""
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        return {"reachable": False, "error": "API key missing"}

    try:
        r = requests.get(
            "https://api.openai.com/v1/models",
            headers={"Authorization": f"Bearer {key}"},
            timeout=5
        )
        return {"reachable": r.status_code == 200, "status": r.status_code}
    except Exception as e:
        return {"reachable": False, "error": str(e)}


# ============================================================================
# RESPONSE HANDLING
# ============================================================================
def get_agent_response(agent_executor, user_input):
    """Gets model response and stores chat history."""
    global chat_history

    # Check API key
    if not os.getenv("OPENAI_API_KEY"):
        return "⚠️ OpenAI API key missing."

    # Check internet
    if not check_internet_connection():
        return "📡 No internet connection."

    try:
        result = agent_executor.invoke({
            "input": user_input,
            "chat_history": chat_history
        })

        output = result.get("output", "Something went wrong.")

        # Store conversation messages
        chat_history.append(HumanMessage(user_input))
        chat_history.append(AIMessage(output))

        return output
    except Exception as e:
        return f"❌ Error: {e}"


def clear_agent_memory():
    """Clears chat history list."""
    global chat_history
    chat_history = []


def initialize_agent():
    """Creates agent only once."""
    global agent_executor
    if agent_executor is None:
        agent_executor = create_agent()


# ============================================================================
# FLASK ROUTES
# ============================================================================

@app.route('/')
def index():
    """Returns chat UI page."""
    return render_template("index.html")


@app.route('/api/chat', methods=['POST'])
def chat():
    """Handles incoming user messages."""
    try:
        data = request.json
        message = data.get("message", "").strip()

        if not message:
            return jsonify({"error": "Message cannot be empty"}), 400

        initialize_agent()
        response = get_agent_response(agent_executor, message)

        return jsonify({"response": response, "status": "success"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/clear', methods=['POST'])
def clear_memory():
    """Clears chat history for a fresh conversation."""
    clear_agent_memory()
    return jsonify({"status": "success", "message": "Memory cleared"})


@app.route('/api/health', methods=['GET'])
def health():
    
    """Simple health check."""
    return jsonify({"status": "healthy"})


@app.route('/api/history', methods=['GET'])
#it returns the history which we chat
def get_history():
    """Returns stored chat history."""
    return jsonify({
        "history": [msg.content for msg in chat_history]
    })


# ============================================================================
# MAIN APPLICATION START
# ============================================================================
if __name__ == '__main__':
    # Remind user to set API key
    if not os.getenv("OPENAI_API_KEY"):
        print("WARNING: OPENAI_API_KEY not found. Add it in .env file.")

    # Run app on port 5000
    app.run(debug=True, port=5000)

