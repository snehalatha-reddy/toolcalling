
# **LangChain Tool-Calling Chat Application**

This project is a Flask-based chat application that uses **LangChain**, **OpenAI**, and **Tool Calling** to intelligently route user messages to the right tools such as:

* Positive message tool
* Negative emotion tool
* Student marks evaluation tool
* Suicide-related support tool

The app also stores **chat history**, checks **internet connectivity**, and validates the **OpenAI API key**.


## 🚀 **Features**

✔ AI Chat System using OpenAI
✔ LangChain Tools for smart routing
✔ Conversation memory
✔ Automatic selection of correct tool
✔ Flask backend with multiple API endpoints
✔ Postman support
✔ Safety tool for crisis messages
✔ Fully contained in one Python file


## 📦 **Project Structure**

```
project/
│── app.py        → Main application (your all-in-one code)
│── templates/
│     └── index.html  → Frontend UI
│── .env           → Stores OPENAI_API_KEY
│── README.md      → Documentation
```


## ⚙️ **Requirements**

Install dependencies:

```bash
pip install flask flask-cors langchain langchain-openai python-dotenv requests
```

---

## 🔑 **Environment Setup**

Create a `.env` file:

```
OPENAI_API_KEY=your_openai_api_key_here
```

---

## ▶️ **Running the Application**

Start the server:

```bash
python app.py
```

The app runs on:
👉 **[http://localhost:5000](http://localhost:5000)**


## 📌 **Available API Endpoints**

### **1. Chat with AI**

```
POST /api/chat
```

JSON Body:

```json
{
  "message": "Hello"
}
```


### **2. Clear Chat History**

```
POST /api/clear
```


### **3. Server Debug Info**

```
GET /api/debug
```

Shows API key status, internet status, and OpenAI reachability.


### **4. Health Check**

```
GET /api/health
```

---

## 🧠 **Tool Routing Logic**

Your system automatically chooses a tool based on message meaning:

| Tool                 | When Used?                                          |
| -------------------- | --------------------------------------------------- |
| positive_prompt_tool | Motivation, encouragement, positive talks           |
| negative_prompt_tool | Sadness, frustration, anger, stress                 |
| student_marks_tool   | Marks, exams, scores, grades                        |
| suicide_related_tool | Suicide/self-harm keywords (**always prioritized**) |


## 🗃 **Chat Memory**

The app stores previous messages so the AI can continue the conversation naturally.

In Postman, you test memory by sending multiple messages.



# 🏗 **Architecture Diagram (Simple ASCII)**

```
                      ┌──────────────────────────────────┐
                      │            User (UI)             │
                      └───────────────┬──────────────────┘
                                      │  HTTP Request
                                      ▼
                     ┌──────────────────────────────────┐
                     │           Flask Server            │
                     └────────────────┬──────────────────┘
                                      │
                                      ▼
                     ┌──────────────────────────────────┐
                     │        /api/chat Endpoint         │
                     └────────────────┬──────────────────┘
                                      │
                                      ▼
                     ┌──────────────────────────────────┐
                     │     Agent Executor (LangChain)    │
                     └────────────────┬──────────────────┘
                                      │
               ┌──────────────────────┼─────────────────────────┐
               │                      │                         │
               ▼                      ▼                         ▼
      ┌──────────────────┐   ┌──────────────────┐    ┌──────────────────┐
      │  Tool Selector   │→→ │ Chat Memory       │    │ Prompt Template  │
      └──────────────────┘   └──────────────────┘    └──────────────────┘
               │
               ▼
      ┌──────────────────────────────────────────────┐
      │    Tools (positive / negative / marks /      │
      │             suicide support)                 │
      └──────────────────────────────────────────────┘
               │
               ▼
      ┌──────────────────────────────────────────────┐
      │        OpenAI Model (gpt-4o-mini)            │
      └──────────────────────────────────────────────┘
               │
               ▼
      ┌──────────────────────────────────────────────┐
      │             Final AI Response                 │
      └──────────────────────────────────────────────┘
```



