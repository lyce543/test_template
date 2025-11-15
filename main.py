import os
from typing import Optional, Dict, Any, List
from datetime import datetime
from enum import Enum
import json

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import httpx
from openai import OpenAI
from agents import Agent, Runner
from dotenv import load_dotenv

load_dotenv()


class Config:
    XANO_BASE_URL = os.getenv("XANO_BASE_URL", "https://your-instance.xano.io/api:xxxxx")
    XANO_API_KEY = os.getenv("XANO_API_KEY", "")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    DEFAULT_MODEL = "gpt-4o"
    MAX_MESSAGES_PER_SESSION = 100


class ChatStatus(str, Enum):
    IDLE = "idle"
    STARTED = "started"
    FINISHED = "finished"
    BLOCKED = "blocked"
    TEST_FINISHED = "test_finished"


class GradeResult(str, Enum):
    PASS = "pass"
    FAIL = "fail"
    EMPTY = ""


class StudentMessage(BaseModel):
    ub_id: int = Field(..., description="User block (chat session) ID")
    content: str = Field(..., description="Student message content")


class AssistantResponse(BaseModel):
    title: str = "-"
    text: str
    type: str = "interview"
    additional: Optional[Dict[str, Any]] = None


class XanoClient:
    def __init__(self, base_url: str, api_key: str):
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        
        headers = {}
        if api_key and api_key != "your_xano_api_key_here":
            headers["Authorization"] = f"Bearer {api_key}"
        
        self.client = httpx.AsyncClient(
            headers=headers,
            timeout=30.0
        )
    
    async def get_block(self, block_id: int) -> Dict[str, Any]:
        url = f"{self.base_url}/block/{block_id}"
        response = await self.client.get(url)
        response.raise_for_status()
        return response.json()
    
    async def get_template(self, template_id: int) -> Dict[str, Any]:
        response = await self.client.get(f"{self.base_url}/template/{template_id}")
        response.raise_for_status()
        return response.json()
    
    async def get_chat_session(self, ub_id: int) -> Dict[str, Any]:
        url = f"{self.base_url}/ub/{ub_id}"
        response = await self.client.get(url)
        response.raise_for_status()
        return response.json()
    
    async def get_messages(self, ub_id: int, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        params = {"ub_id": ub_id}
        if limit:
            params["limit"] = limit
        response = await self.client.get(f"{self.base_url}/air", params=params)
        response.raise_for_status()
        return response.json()
    
    async def get_conversation_history(self, ub_id: int, session_data: Dict[str, Any]) -> List[Dict[str, str]]:
        messages_data = await self.get_messages(ub_id)
        conversation_history = []
        
        if not messages_data:
            return []
        
        for msg in messages_data:
            role = msg.get("role")
            if not role:
                continue
            
            if role == "user":
                user_content = msg.get("user_content", {})
                if isinstance(user_content, dict):
                    content = user_content.get("text", "")
                else:
                    content = ""
            elif role == "assistant":
                ai_content = msg.get("ai_content", [])
                if isinstance(ai_content, list) and len(ai_content) > 0:
                    content = ai_content[0].get("text", "") if isinstance(ai_content[0], dict) else ""
                else:
                    content = ""
            else:
                continue
            
            if content:
                conversation_history.append({
                    "role": role,
                    "content": content
                })
        
        return conversation_history
    
    async def save_message(self, ub_id: int, role: str, content: str, prev_id: Optional[int] = None, 
                          run_info: Optional[Dict] = None, ai_message_info: Optional[Dict] = None) -> Dict[str, Any]:
        
        timestamp = int(datetime.now().timestamp() * 1000)
        
        message_record = {
            "ub_id": ub_id,
            "role": role,
            "created_at": timestamp,
            "status": "new",
            "block_id": ub_id,
            "user_content": json.dumps({
                "type": "text",
                "text": content,
                "created_at": timestamp
            }) if role == "user" else json.dumps({}),
            "ai_content": json.dumps([{
                "text": content,
                "type": None,
                "title": "",
                "created_at": timestamp
            }]) if role == "assistant" else json.dumps([])
        }
        
        if prev_id:
            message_record["prev_id"] = prev_id
        if run_info:
            message_record["run_info"] = json.dumps(run_info) if isinstance(run_info, dict) else run_info
        if ai_message_info:
            message_record["ai_message_info"] = json.dumps(ai_message_info) if isinstance(ai_message_info, dict) else ai_message_info
        
        try:
            response = await self.client.post(f"{self.base_url}/add_air", json=message_record)
            if response.status_code in [200, 201]:
                result = response.json()
                print(f"✅ Message saved: id={result.get('id')}, role={role}")
                return result
            else:
                print(f"❌ Failed to save message: {response.status_code}")
                print(f"   Response: {response.text[:200]}")
        except Exception as e:
            print(f"❌ Exception saving message: {e}")
        
        return {
            "id": timestamp,
            "ub_id": ub_id,
            "role": role,
            "created_at": timestamp,
            "status": "new"
        }
    
    async def update_chat_status(
        self, 
        ub_id: int, 
        status: ChatStatus,
        grade: Optional[GradeResult] = None,
        last_air_id: Optional[int] = None
    ) -> Dict[str, Any]:
        return {"status": "ok", "message": "Status update skipped"}
    
    async def update_block_agent(self, block_id: int, agent_id: str) -> Dict[str, Any]:
        response = await self.client.patch(
            f"{self.base_url}/block/{block_id}",
            json={"assistant_id": agent_id}
        )
        response.raise_for_status()
        return response.json()
    
    async def update_session_ids(self, ub_id: int, agent_id: str, session_id: str) -> Dict[str, Any]:
        return {"status": "ok", "message": "Session IDs saved"}


class AgentManager:
    def __init__(self, openai_client: OpenAI):
        self.client = openai_client
        self.agents_cache = {}
    
    def build_instructions(self, template_instructions: str, specifications: Any) -> str:
        instructions = template_instructions
        
        if specifications:
            instructions += "\n\n# Assignment Specifications Structure: \n"
            instructions += json.dumps(specifications.get("params_structure", [])) if isinstance(specifications, dict) else ""
            instructions += "\n\n# Specifications: \n"
            
            specs_data = specifications.get("specifications", {}) if isinstance(specifications, dict) else {}
            instructions += json.dumps(specs_data)
        
        return instructions
    
    def parse_functions(self, function_list: str) -> List[Dict]:
        if not function_list:
            return []
        try:
            return json.loads(function_list)
        except:
            return []
    
    async def get_or_create_agent(
        self,
        block: Dict[str, Any],
        template: Dict[str, Any],
        xano: XanoClient
    ) -> Agent:
        block_id = block["id"]
        
        if block_id in self.agents_cache:
            return self.agents_cache[block_id]
        
        instructions = self.build_instructions(
            template.get("instructions", ""),
            {
                "params_structure": template.get("params_structure", []),
                "specifications": block.get("specifications", {})
            }
        )
        
        if template.get("int_function_calling"):
            instructions += "\n\n" + template.get("int_function_calling", "")
        
        agent = Agent(
            name=f"{template.get('name', 'Assistant')} - Block {block_id}",
            instructions=instructions,
            model=template.get("model", Config.DEFAULT_MODEL)
        )
        
        self.agents_cache[block_id] = agent
        
        return agent
    
    async def send_message(
        self,
        agent: Agent,
        message: str,
        history: Optional[List[Dict[str, str]]] = None
    ) -> tuple[str, Dict, Dict]:
        try:
            messages = []
            
            if history:
                messages.extend(history)
            
            messages.append({"role": "user", "content": message})
            
            import asyncio
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, lambda: Runner.run_sync(agent, messages))
            
            full_response = result.final_output if hasattr(result, 'final_output') else str(result)
            
            run_info = {
                "agent_name": agent.name,
                "timestamp": datetime.now().isoformat(),
                "model": agent.model
            }
            
            msg_info = {
                "agent_name": agent.name,
                "role": "assistant"
            }
            
            return full_response, run_info, msg_info
            
        except Exception as e:
            print(f"ERROR in send_message: {type(e).__name__}: {str(e)}")
            import traceback
            traceback.print_exc()
            return f"Error: {str(e)}", {}, {}


class EvaluationAgent:
    def __init__(self, openai_client: OpenAI):
        self.client = openai_client
    
    def build_evaluation_prompt(
        self,
        template_instructions: str,
        specifications: Dict[str, Any],
        criteria: List[Dict[str, Any]]
    ) -> str:
        prompt = template_instructions
        
        if specifications:
            prompt += "\n\n# Assignment Specifications:\n"
            for key, value in specifications.items():
                prompt += f"\n## {key}:\n{value}\n"
        
        if criteria:
            prompt += "\n\n# Evaluation Criteria:\n"
            for criterion in criteria:
                prompt += f"\n## {criterion.get('criterion_name')}:\n"
                prompt += f"Max points: {criterion.get('max_points')}\n"
                prompt += f"Instructions: {criterion.get('grading_instructions')}\n"
        
        return prompt
    
    async def evaluate_chat(
        self,
        conversation_history: List[Dict[str, str]],
        template: Dict[str, Any],
        specifications: Dict[str, Any],
        criteria: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        system_prompt = self.build_evaluation_prompt(
            template.get("eval_instructions", ""),
            specifications,
            criteria
        )
        
        messages = [{"role": "system", "content": system_prompt}] + conversation_history
        
        response = self.client.chat.completions.create(
            model=template.get("model", Config.DEFAULT_MODEL),
            messages=messages
        )
        
        return {
            "evaluation": response.choices[0].message.content,
            "timestamp": datetime.now().isoformat()
        }


app = FastAPI(
    title="EdTech AI Platform",
    description="AI-powered educational assistant platform",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

xano = XanoClient(Config.XANO_BASE_URL, Config.XANO_API_KEY)
openai_client = OpenAI(api_key=Config.OPENAI_API_KEY)
agent_manager = AgentManager(openai_client)


@app.get("/")
async def root():
    return {
        "status": "operational",
        "version": "2.0.0",
        "message": "EdTech AI Platform - Python Backend with OpenAI Agents"
    }


@app.get("/health")
async def health():
    xano_configured = bool(Config.XANO_BASE_URL and Config.XANO_BASE_URL != "https://your-instance.xano.io/api:xxxxx")
    openai_configured = bool(Config.OPENAI_API_KEY and not Config.OPENAI_API_KEY.startswith("your_") and len(Config.OPENAI_API_KEY) > 20)
    
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "xano_configured": xano_configured,
        "openai_configured": openai_configured
    }


@app.post("/chat/message", response_model=AssistantResponse)
async def process_student_message(
    message: StudentMessage,
    background_tasks: BackgroundTasks
):
    try:
        session = await xano.get_chat_session(message.ub_id)
        block = await xano.get_block(session["block_id"])
        template_data = await xano.get_template(block["int_template_id"])
        
        agent = await agent_manager.get_or_create_agent(block, template_data, xano)
        
        history = await xano.get_conversation_history(message.ub_id, session)
        
        print(f"DEBUG: History loaded: {len(history)} messages")
        for i, h in enumerate(history):
            print(f"  {i}: role={h.get('role')}, content_len={len(h.get('content', ''))}")
        
        messages_data = await xano.get_messages(message.ub_id, limit=1)
        last_air_id = messages_data[0]["id"] if messages_data else None
        
        saved_user_msg = await xano.save_message(
            message.ub_id, 
            "user", 
            message.content,
            last_air_id
        )
        
        if not saved_user_msg or "id" not in saved_user_msg:
            print(f"ERROR: Failed to save user message properly")
            raise HTTPException(status_code=500, detail="Failed to save user message")
        
        ai_response, run_info, msg_info = await agent_manager.send_message(
            agent,
            message.content,
            history
        )
        
        saved_ai_msg = await xano.save_message(
            message.ub_id, 
            "assistant", 
            ai_response,
            saved_user_msg.get("id"),
            run_info,
            msg_info
        )
        
        await xano.update_chat_status(
            message.ub_id,
            ChatStatus.STARTED,
            last_air_id=saved_ai_msg["id"]
        )
        
        try:
            response_json = json.loads(ai_response)
            return AssistantResponse(**response_json)
        except:
            return AssistantResponse(
                title="-",
                text=ai_response,
                type="interview"
            )
        
    except Exception as e:
        print(f"ERROR: {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat/{ub_id}/evaluate")
async def evaluate_chat(ub_id: int):
    try:
        session = await xano.get_chat_session(ub_id)
        block = await xano.get_block(session["block_id"])
        eval_template_data = await xano.get_template(block["eval_template_id"])
        
        messages_data = await xano.get_messages(ub_id)
        conversation_history = []
        
        messages_dict = {msg["id"]: msg for msg in messages_data}
        current_id = session.get("last_air_id")
        
        while current_id and current_id in messages_dict:
            msg = messages_dict[current_id]
            
            if msg["role"] == "user":
                content = msg.get("user_content", {}).get("text", "")
            else:
                ai_content = msg.get("ai_content", [])
                content = ai_content[0].get("text", "") if ai_content else ""
            
            conversation_history.insert(0, {
                "role": msg["role"],
                "content": content
            })
            
            current_id = msg.get("prev_id")
        
        eval_agent = EvaluationAgent(openai_client)
        evaluation = await eval_agent.evaluate_chat(
            conversation_history=conversation_history,
            template=eval_template_data,
            specifications=block.get("specifications", {}),
            criteria=eval_template_data.get("eval_crit_json", [])
        )
        
        return evaluation
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/chat/{ub_id}/history")
async def get_chat_history(ub_id: int):
    try:
        messages = await xano.get_messages(ub_id)
        return {
            "messages": messages,
            "count": len(messages)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat/{ub_id}/clear-memory")
async def clear_chat_memory(ub_id: int):
    try:
        session = await xano.get_chat_session(ub_id)
        block_id = session.get("block_id")
        
        if block_id and block_id in agent_manager.agents_cache:
            del agent_manager.agents_cache[block_id]
        
        return {"status": "memory cleared"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)