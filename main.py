"""
EdTech AI Assistant Platform - Python Backend
Replaces no-code Assistants API with modern agent workflows
"""

import os
from typing import Optional, Dict, Any, List
from datetime import datetime
from enum import Enum

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import httpx
from openai import OpenAI

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """Central configuration"""
    XANO_BASE_URL = os.getenv("XANO_BASE_URL", "https://your-instance.xano.io/api:xxxxx")
    XANO_API_KEY = os.getenv("XANO_API_KEY", "")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    
    # Models
    DEFAULT_MODEL = "gpt-4o"
    
    # Rate limits
    MAX_MESSAGES_PER_SESSION = 100


# ============================================================================
# DATA MODELS
# ============================================================================

class ChatStatus(str, Enum):
    """Chat status enum matching Xano"""
    IDLE = "idle"
    STARTED = "started"
    FINISHED = "finished"
    BLOCKED = "blocked"
    TEST_FINISHED = "test_finished"


class GradeResult(str, Enum):
    """Grade results"""
    PASS = "pass"
    FAIL = "fail"
    EMPTY = ""


class MessageRole(str, Enum):
    """Message roles"""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


# Pydantic Models for API

class StudentMessage(BaseModel):
    """Student message input"""
    ub_id: int = Field(..., description="User block (chat session) ID")
    content: str = Field(..., description="Student message content")


class AssistantResponse(BaseModel):
    """Assistant response"""
    title: str = "-"
    text: str
    type: str = "interview"  # or "test_finished"
    additional: Optional[Dict[str, Any]] = None


class ChatSession(BaseModel):
    """Chat session data from UB table"""
    id: int
    user_id: int
    block_id: int
    status: ChatStatus
    thread_id: Optional[str] = None
    created_at: int


class Block(BaseModel):
    """Exercise block from blocks table"""
    id: int
    name: str
    int_template_id: int
    eval_template_id: int
    int_instructions: str
    eval_instructions: str
    specifications: Dict[str, Any]
    int_specs: Optional[str] = None
    eval_specs: Optional[str] = None


class Template(BaseModel):
    """Template from template table"""
    id: int
    name: str
    type: str
    instructions: str
    eval_instructions: str
    model: str = "gpt-4o"
    params_structure: List[Dict[str, Any]]
    functions: List[str]


# ============================================================================
# XANO CLIENT
# ============================================================================

class XanoClient:
    """Client for Xano API operations"""
    
    def __init__(self, base_url: str, api_key: str):
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.client = httpx.AsyncClient(
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=30.0
        )
    
    async def get_block(self, block_id: int) -> Dict[str, Any]:
        """Get block by ID"""
        response = await self.client.get(f"{self.base_url}/block/{block_id}")
        response.raise_for_status()
        return response.json()
    
    async def get_template(self, template_id: int) -> Dict[str, Any]:
        """Get template by ID"""
        response = await self.client.get(f"{self.base_url}/template/{template_id}")
        response.raise_for_status()
        return response.json()
    
    async def get_chat_session(self, ub_id: int) -> Dict[str, Any]:
        """Get chat session (UB record)"""
        response = await self.client.get(f"{self.base_url}/ub/{ub_id}")
        response.raise_for_status()
        return response.json()
    
    async def get_messages(self, ub_id: int) -> List[Dict[str, Any]]:
        """Get all messages for a chat session"""
        response = await self.client.get(
            f"{self.base_url}/air",
            params={"ub_id": ub_id}
        )
        response.raise_for_status()
        return response.json()
    
    async def save_message(self, ub_id: int, role: str, content: str) -> Dict[str, Any]:
        """Save a message to air table"""
        data = {
            "ub_id": ub_id,
            "role": role,
            "user_content": {"text": content} if role == "user" else {},
            "ai_content": [{"text": content}] if role == "assistant" else [],
            "created_at": int(datetime.now().timestamp() * 1000)
        }
        response = await self.client.post(f"{self.base_url}/air", json=data)
        response.raise_for_status()
        return response.json()
    
    async def update_chat_status(
        self, 
        ub_id: int, 
        status: ChatStatus,
        grade: Optional[GradeResult] = None
    ) -> Dict[str, Any]:
        """Update chat session status"""
        data = {"status": status.value}
        if grade:
            data["grade"] = grade.value
        
        response = await self.client.patch(
            f"{self.base_url}/ub/{ub_id}",
            json=data
        )
        response.raise_for_status()
        return response.json()
    
    async def run_evaluation(self, ub_id: int) -> Dict[str, Any]:
        """Trigger evaluation for a chat session"""
        response = await self.client.post(
            f"{self.base_url}/evaluate/{ub_id}"
        )
        response.raise_for_status()
        return response.json()


# ============================================================================
# AGENT SYSTEM
# ============================================================================

class BaseAgent:
    """Base class for all agents"""
    
    def __init__(self, openai_client: OpenAI, model: str = Config.DEFAULT_MODEL):
        self.client = openai_client
        self.model = model
    
    def build_system_prompt(
        self, 
        template_instructions: str, 
        specifications: Dict[str, Any]
    ) -> str:
        """Build system prompt from template + specifications"""
        # Combine template instructions with teacher specifications
        prompt = template_instructions
        
        if specifications:
            prompt += "\n\n# Assignment Specifications:\n"
            for key, value in specifications.items():
                prompt += f"\n## {key}:\n{value}\n"
        
        return prompt
    
    async def generate_response(
        self,
        messages: List[Dict[str, str]],
        system_prompt: str,
        functions: Optional[List[Dict]] = None
    ) -> Dict[str, Any]:
        """Generate response using OpenAI"""
        
        all_messages = [
            {"role": "system", "content": system_prompt}
        ] + messages
        
        kwargs = {
            "model": self.model,
            "messages": all_messages,
        }
        
        if functions:
            kwargs["tools"] = [{"type": "function", "function": f} for f in functions]
            kwargs["tool_choice"] = "auto"
        
        response = self.client.chat.completions.create(**kwargs)
        
        # Check if function was called
        message = response.choices[0].message
        
        if message.tool_calls:
            return {
                "type": "function_call",
                "function_name": message.tool_calls[0].function.name,
                "arguments": message.tool_calls[0].function.arguments
            }
        
        return {
            "type": "message",
            "content": message.content
        }


class ExaminationAgent(BaseAgent):
    """
    Examination-style interview agent
    Asks questions, doesn't give hints or answers
    """
    
    FUNCTIONS = [
        {
            "name": "status_grade",
            "description": "Update chat status and assign pass/fail grade",
            "parameters": {
                "type": "object",
                "required": ["status", "grade"],
                "properties": {
                    "status": {
                        "type": "string",
                        "enum": ["idle", "started", "finished", "blocked"],
                        "description": "The current status of the chat session"
                    },
                    "grade": {
                        "type": "string",
                        "enum": ["fail", "pass"],
                        "description": "Evaluation of the chat outcome"
                    }
                },
                "additionalProperties": False
            },
            "strict": True
        }
    ]
    
    async def process_message(
        self,
        student_message: str,
        conversation_history: List[Dict[str, str]],
        template: Template,
        specifications: Dict[str, Any]
    ) -> AssistantResponse:
        """Process student message and generate response"""
        
        # Build system prompt
        system_prompt = self.build_system_prompt(
            template.instructions,
            specifications
        )
        
        # Add student message to history
        messages = conversation_history + [
            {"role": "user", "content": student_message}
        ]
        
        # Generate response
        result = await self.generate_response(
            messages=messages,
            system_prompt=system_prompt,
            functions=self.FUNCTIONS
        )
        
        # Handle function call
        if result["type"] == "function_call":
            import json
            args = json.loads(result["arguments"])
            
            return AssistantResponse(
                title="-",
                text="Thank you for completing the examination.",
                type="test_finished",
                additional={
                    "function": result["function_name"],
                    "status": args.get("status"),
                    "grade": args.get("grade")
                }
            )
        
        # Parse JSON response if needed
        content = result["content"]
        
        # Try to parse as JSON (examination template uses JSON format)
        try:
            import json
            response_data = json.loads(content)
            return AssistantResponse(**response_data)
        except:
            # Fallback to plain text
            return AssistantResponse(
                title="-",
                text=content,
                type="interview"
            )


class EvaluationAgent(BaseAgent):
    """
    Evaluation agent - analyzes chat and grades student
    """
    
    async def evaluate_chat(
        self,
        conversation_history: List[Dict[str, str]],
        template: Template,
        specifications: Dict[str, Any],
        criteria: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Evaluate student performance"""
        
        # Build evaluation prompt
        system_prompt = self.build_system_prompt(
            template.eval_instructions,
            specifications
        )
        
        # Add criteria
        if criteria:
            system_prompt += "\n\n# Evaluation Criteria:\n"
            for criterion in criteria:
                system_prompt += f"\n## {criterion.get('criterion_name')}:\n"
                system_prompt += f"Max points: {criterion.get('max_points')}\n"
                system_prompt += f"Instructions: {criterion.get('grading_instructions')}\n"
        
        # Generate evaluation
        result = await self.generate_response(
            messages=conversation_history,
            system_prompt=system_prompt
        )
        
        return {
            "evaluation": result["content"],
            "timestamp": datetime.now().isoformat()
        }


# ============================================================================
# AGENT FACTORY
# ============================================================================

class AgentFactory:
    """Factory to create appropriate agent based on template type"""
    
    @staticmethod
    def create_chat_agent(template: Template, openai_client: OpenAI) -> BaseAgent:
        """Create chat agent based on template type"""
        
        template_type = template.type
        
        if template_type == "interview":
            # For examination-style
            if "examination" in template.name.lower():
                return ExaminationAgent(openai_client, template.model)
            # Add other types here
            else:
                return ExaminationAgent(openai_client, template.model)
        
        # Default
        return BaseAgent(openai_client, template.model)
    
    @staticmethod
    def create_eval_agent(template: Template, openai_client: OpenAI) -> EvaluationAgent:
        """Create evaluation agent"""
        return EvaluationAgent(openai_client, template.model)


# ============================================================================
# FASTAPI APPLICATION
# ============================================================================

app = FastAPI(
    title="EdTech AI Platform",
    description="AI-powered educational assistant platform",
    version="2.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure properly in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize clients
xano = XanoClient(Config.XANO_BASE_URL, Config.XANO_API_KEY)
openai_client = OpenAI(api_key=Config.OPENAI_API_KEY)


# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.get("/")
async def root():
    """Health check"""
    return {
        "status": "operational",
        "version": "2.0.0",
        "message": "EdTech AI Platform - Python Backend"
    }


@app.post("/chat/message", response_model=AssistantResponse)
async def process_student_message(
    message: StudentMessage,
    background_tasks: BackgroundTasks
):
    """
    Main endpoint: process student message and return AI response
    
    This replaces the old Assistants API flow:
    1. Get chat session (UB)
    2. Get block configuration
    3. Get template
    4. Load conversation history
    5. Process with appropriate agent
    6. Save messages
    7. Return response
    """
    
    try:
        # 1. Get chat session
        session = await xano.get_chat_session(message.ub_id)
        
        # 2. Get block configuration
        block = await xano.get_block(session["block_id"])
        
        # 3. Get template
        template_data = await xano.get_template(block["int_template_id"])
        template = Template(**template_data)
        
        # 4. Load conversation history
        messages_data = await xano.get_messages(message.ub_id)
        conversation_history = [
            {
                "role": msg["role"],
                "content": msg["user_content"]["text"] if msg["role"] == "user" 
                          else msg["ai_content"][0]["text"]
            }
            for msg in messages_data
        ]
        
        # 5. Save student message
        await xano.save_message(message.ub_id, "user", message.content)
        
        # 6. Create appropriate agent
        agent = AgentFactory.create_chat_agent(template, openai_client)
        
        # 7. Process message
        response = await agent.process_message(
            student_message=message.content,
            conversation_history=conversation_history,
            template=template,
            specifications=block.get("specifications", {})
        )
        
        # 8. Save assistant response
        await xano.save_message(message.ub_id, "assistant", response.text)
        
        # 9. Handle status updates if test finished
        if response.type == "test_finished" and response.additional:
            background_tasks.add_task(
                xano.update_chat_status,
                message.ub_id,
                ChatStatus.TEST_FINISHED,
                GradeResult(response.additional.get("grade", ""))
            )
        
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat/{ub_id}/evaluate")
async def evaluate_chat(ub_id: int):
    """
    Run evaluation on completed chat
    """
    
    try:
        # Get chat session and block
        session = await xano.get_chat_session(ub_id)
        block = await xano.get_block(session["block_id"])
        
        # Get evaluation template
        eval_template_data = await xano.get_template(block["eval_template_id"])
        eval_template = Template(**eval_template_data)
        
        # Load conversation
        messages_data = await xano.get_messages(ub_id)
        conversation_history = [
            {
                "role": msg["role"],
                "content": msg["user_content"]["text"] if msg["role"] == "user" 
                          else msg["ai_content"][0]["text"]
            }
            for msg in messages_data
        ]
        
        # Create eval agent
        eval_agent = AgentFactory.create_eval_agent(eval_template, openai_client)
        
        # Run evaluation
        evaluation = await eval_agent.evaluate_chat(
            conversation_history=conversation_history,
            template=eval_template,
            specifications=block.get("specifications", {}),
            criteria=block.get("eval_crit_json", [])
        )
        
        return evaluation
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/templates")
async def list_templates():
    """List all available templates"""
    # This would query Xano for all templates
    return {"message": "Templates endpoint - implement based on Xano structure"}


@app.get("/chat/{ub_id}/history")
async def get_chat_history(ub_id: int):
    """Get conversation history for a chat session"""
    
    try:
        messages = await xano.get_messages(ub_id)
        return {"messages": messages}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)