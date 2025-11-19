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
from agents import Agent, Runner, function_tool
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
        
        if not messages_data:
            return []
        
        messages_dict = {msg["id"]: msg for msg in messages_data}
        
        root_messages = [msg for msg in messages_data if not msg.get("prev_id") or msg.get("prev_id") == 0]
        
        if not root_messages:
            sorted_messages = sorted(messages_data, key=lambda x: x.get("created_at", 0))
        else:
            sorted_messages = []
            visited = set()
            
            def build_chain(msg_id):
                if msg_id in visited or msg_id not in messages_dict:
                    return
                visited.add(msg_id)
                msg = messages_dict[msg_id]
                sorted_messages.append(msg)
                
                for other_msg in messages_data:
                    if other_msg.get("prev_id") == msg_id and other_msg["id"] not in visited:
                        build_chain(other_msg["id"])
            
            build_chain(root_messages[0]["id"])
        
        conversation_history = []
        for msg in sorted_messages:
            user_content_raw = msg.get("user_content", "{}")
            try:
                if isinstance(user_content_raw, str):
                    user_content = json.loads(user_content_raw)
                else:
                    user_content = user_content_raw
            except:
                user_content = {}
            
            user_text = user_content.get("text", "") if isinstance(user_content, dict) else ""
            
            ai_content_raw = msg.get("ai_content", "[]")
            try:
                if isinstance(ai_content_raw, str):
                    ai_content = json.loads(ai_content_raw)
                else:
                    ai_content = ai_content_raw
            except:
                ai_content = []
            
            ai_text = ""
            if isinstance(ai_content, list) and len(ai_content) > 0:
                if isinstance(ai_content[0], dict):
                    ai_text = ai_content[0].get("text", "")
            
            if user_text:
                conversation_history.append({
                    "role": "user",
                    "content": user_text
                })
            
            if ai_text:
                conversation_history.append({
                    "role": "assistant",
                    "content": ai_text
                })
        
        print(f"📚 Loaded conversation history: {len(conversation_history)} messages")
        return conversation_history
    
    async def save_message_pair(
        self, 
        ub_id: int, 
        user_message: str, 
        ai_response: str,
        prev_id: Optional[int] = None,
        run_info: Optional[Dict] = None,
        ai_message_info: Optional[Dict] = None
    ) -> Dict[str, Any]:
        
        timestamp = int(datetime.now().timestamp() * 1000)
        
        message_record = {
            "ub_id": ub_id,
            "role": "assistant",
            "created_at": timestamp,
            "status": "new",
            "block_id": ub_id,
        }
        
        message_record["user_content"] = json.dumps({
            "type": "text",
            "text": user_message,
            "file_url": "",
            "created_at": timestamp,
            "error_message": "",
            "error_code": "",
            "file": None,
            "images": None
        })
        
        message_record["ai_content"] = json.dumps([{
            "text": ai_response,
            "title": "",
            "created_at": timestamp,
            "type": None,
            "grade": None,
            "additional": ""
        }])
        
        if prev_id and prev_id > 0:
            message_record["prev_id"] = prev_id
        else:
            message_record["prev_id"] = 0
        
        if run_info:
            message_record["run_info"] = json.dumps(run_info) if isinstance(run_info, dict) else run_info
        if ai_message_info:
            message_record["ai_message_info"] = json.dumps(ai_message_info) if isinstance(ai_message_info, dict) else ai_message_info
        
        try:
            response = await self.client.post(f"{self.base_url}/add_air", json=message_record)
            if response.status_code in [200, 201]:
                result = response.json()
                new_msg_id = result.get('id')
                
                print(f"✅ Message pair saved: id={new_msg_id}, user+ai, prev_id={prev_id or 0}")
                
                return result
            else:
                print(f"❌ Failed to save message pair: {response.status_code}")
                print(f"   Response: {response.text[:200]}")
        except Exception as e:
            print(f"❌ Exception saving message pair: {e}")
        
        return {
            "id": timestamp,
            "ub_id": ub_id,
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
        update_data = {
            "ub_id": int(ub_id),
            "status": status.value
        }
        if grade:
            update_data["grade"] = grade.value
        if last_air_id:
            update_data["last_air_id"] = int(last_air_id)
        
        print(f"   📤 Sending to Xano update_ub: {update_data}")
        
        try:
            response = await self.client.post(
                f"{self.base_url}/update_ub",
                json=update_data
            )
            
            print(f"   📥 Xano response status: {response.status_code}")
            
            if response.status_code in [200, 201]:
                print(f"✅ Chat status updated: {status.value}, grade: {grade.value if grade else 'none'}")
                return response.json()
            else:
                error_text = response.text[:500]
                print(f"⚠️  Failed to update chat status: {response.status_code}")
                print(f"   Response: {error_text}")
                return {"status": "ok", "message": "Status update failed but continuing"}
        except Exception as e:
            print(f"⚠️  Exception updating chat status: {e}")
            import traceback
            traceback.print_exc()
            return {"status": "ok", "message": "Status update failed but continuing"}


class AgentManager:
    def __init__(self, openai_client: OpenAI):
        self.client = openai_client
        self.agents_cache = {}
    
    def build_instructions(self, template_instructions: str, specifications: Any) -> str:
        instructions = template_instructions
        
        if specifications:
            instructions += "\n\n# Assignment Specifications Structure: \n"
            instructions += json.dumps(specifications.get("params_structure", []), ensure_ascii=False) if isinstance(specifications, dict) else ""
            instructions += "\n\n# Specifications: \n"
            
            specs_data = specifications.get("specifications", {}) if isinstance(specifications, dict) else {}
            instructions += json.dumps(specs_data, ensure_ascii=False)
        
        return instructions
    
    def parse_tools(self, function_list: List[Dict]) -> List[callable]:
        if not function_list:
            return []
        
        tools = []
        
        for func in function_list:
            if func.get("name") == "status_grade":
                @function_tool
                def status_grade(status: str, grade: str = "") -> dict:
                    """Update the current chat's status and assign a pass/fail grade.
                    
                    Args:
                        status: The current status. Valid values: "idle", "started", "finished", "blocked"
                        grade: The grade. Valid values: "pass", "fail", or empty string ""
                    
                    Returns:
                        dict: Confirmation with status and grade
                    """
                    return {"status": status, "grade": grade, "success": True}
                
                tools.append(status_grade)
            
            elif func.get("name") == "random_numbers":
                @function_tool
                def random_numbers(array_length: int) -> dict:
                    """Get an array of N random numbers.
                    
                    Args:
                        array_length: Length of the array
                    """
                    import random
                    numbers = [random.randint(0, 1000000) for _ in range(array_length)]
                    return {"numbers": numbers, "success": True}
                
                tools.append(random_numbers)
        
        return tools
    
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
        
        tools = self.parse_tools(template.get("function_list", []))
        
        print(f"🔧 Creating agent with {len(tools)} tools: {[t.__name__ if hasattr(t, '__name__') else 'unknown' for t in tools]}")
        
        agent = Agent(
            name=f"{template.get('name', 'Assistant')} - Block {block_id}",
            instructions=instructions,
            model=template.get("model", Config.DEFAULT_MODEL),
            tools=tools if tools else []
        )
        
        self.agents_cache[block_id] = agent
        
        return agent
    
    async def send_message(
        self,
        agent: Agent,
        message: str,
        history: Optional[List[Dict[str, str]]] = None,
        ub_id: Optional[int] = None,
        xano: Optional[XanoClient] = None
    ) -> tuple[str, Dict, Dict, Optional[Dict]]:
        try:
            messages = []
            
            if history:
                messages.extend(history)
            
            messages.append({"role": "user", "content": message})
            
            import asyncio
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, lambda: Runner.run_sync(agent, messages))
            
            full_response = result.final_output if hasattr(result, 'final_output') else str(result)
            
            print(f"🤖 Agent response received")
            
            function_calls = None
            if hasattr(result, 'new_items') and result.new_items:
                print(f"🔧 Processing {len(result.new_items)} new items")
                function_calls = await self.handle_tool_results(result.new_items, ub_id, xano)
            
            run_info = {
                "agent_name": agent.name,
                "timestamp": datetime.now().isoformat(),
                "model": agent.model,
                "function_calls": function_calls
            }
            
            msg_info = {
                "agent_name": agent.name,
                "role": "assistant"
            }
            
            return full_response, run_info, msg_info, function_calls
            
        except Exception as e:
            print(f"ERROR in send_message: {type(e).__name__}: {str(e)}")
            import traceback
            traceback.print_exc()
            return f"Error: {str(e)}", {}, {}, None
    
    async def handle_tool_results(
        self,
        items: List[Any],
        ub_id: Optional[int],
        xano: Optional[XanoClient]
    ) -> Dict[str, Any]:
        results = {}
        
        for i, item in enumerate(items):
            item_type = type(item).__name__
            print(f"   📦 Item {i}: {item_type}")
            
            if item_type == "ToolCallOutputItem":
                func_name = None
                result_value = None
                
                if hasattr(item, 'raw_item'):
                    raw_item = item.raw_item
                    
                    if hasattr(raw_item, 'name'):
                        func_name = raw_item.name
                    elif hasattr(raw_item, 'function') and hasattr(raw_item.function, 'name'):
                        func_name = raw_item.function.name
                    
                    if hasattr(raw_item, 'output'):
                        result_value = raw_item.output
                
                if hasattr(item, 'output'):
                    result_value = item.output
                    
                    if isinstance(result_value, dict):
                        if 'status' in result_value and 'grade' in result_value:
                            func_name = 'status_grade'
                        elif 'numbers' in result_value:
                            func_name = 'random_numbers'
                
                if not func_name and i > 0:
                    for j in range(i-1, -1, -1):
                        if type(items[j]).__name__ == "ToolCallItem":
                            prev_item = items[j]
                            if hasattr(prev_item, 'raw_item'):
                                prev_raw = prev_item.raw_item
                                if hasattr(prev_raw, 'name'):
                                    func_name = prev_raw.name
                                elif hasattr(prev_raw, 'function') and hasattr(prev_raw.function, 'name'):
                                    func_name = prev_raw.function.name
                            break
                
                if not func_name:
                    continue
                
                print(f"   🎯 Tool called: {func_name}")
                print(f"   📤 Output: {result_value}")
                
                if func_name == "status_grade":
                    try:
                        if isinstance(result_value, dict):
                            status = result_value.get("status")
                            grade = result_value.get("grade", "")
                        elif isinstance(result_value, str):
                            try:
                                parsed = json.loads(result_value)
                                status = parsed.get("status")
                                grade = parsed.get("grade", "")
                            except:
                                status = None
                                grade = ""
                        else:
                            status = None
                            grade = ""
                        
                        print(f"   🎯 Parsed status_grade: status={status}, grade={grade}")
                        
                        if ub_id and xano and status:
                            valid_statuses = [s.value for s in ChatStatus]
                            if status not in valid_statuses:
                                print(f"   ⚠️  Invalid status '{status}' for Xano. Valid: {valid_statuses}")
                                print(f"   Using 'finished' instead")
                                status = "finished"
                            
                            chat_status = ChatStatus(status)
                            grade_result = GradeResult(grade) if grade and grade in [g.value for g in GradeResult] else None
                            
                            await xano.update_chat_status(ub_id, chat_status, grade_result)
                        
                        results[func_name] = {
                            "status": status,
                            "grade": grade,
                            "success": True
                        }
                    except Exception as e:
                        print(f"   ❌ Error handling status_grade: {e}")
                        results[func_name] = {"success": False, "error": str(e)}
                
                elif func_name == "random_numbers":
                    results[func_name] = result_value
        
        return results if results else None


class EvaluationAgent:
    def __init__(self, openai_client: OpenAI):
        self.client = openai_client
    
    def build_evaluation_prompt(
        self,
        eval_instructions: str,
        specifications: Any,
        criteria: List[Dict[str, Any]]
    ) -> str:
        prompt = eval_instructions
        
        if specifications:
            prompt += "\n\n# Questions and Key Concepts:\n"
            if isinstance(specifications, list):
                for i, spec in enumerate(specifications, 1):
                    prompt += f"\nQuestion {i}: {spec.get('question', '')}\n"
                    prompt += f"Key Concepts: {spec.get('key_concepts', '')}\n"
            elif isinstance(specifications, dict):
                for key, value in specifications.items():
                    prompt += f"\n{key}: {value}\n"
        
        if criteria:
            prompt += "\n\n# Evaluation Criteria:\n"
            for i, criterion in enumerate(criteria, 1):
                crit_name = criterion.get('criterion_name', f'Criterion {i}')
                max_pts = criterion.get('max_points', 0)
                summary = criterion.get('summary_instructions', '')
                grading = criterion.get('grading_instructions', '')
                
                prompt += f"\n## Criterion {i}"
                if crit_name:
                    prompt += f": {crit_name}"
                prompt += f"\nMax Points: {max_pts}\n"
                if summary:
                    prompt += f"Summary Instructions: {summary}\n"
                if grading:
                    prompt += f"Grading Instructions: {grading}\n"
        
        return prompt
    
    async def evaluate_chat(
        self,
        conversation_history: List[Dict[str, str]],
        eval_instructions: str,
        specifications: Any,
        criteria: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        
        system_prompt = self.build_evaluation_prompt(
            eval_instructions,
            specifications,
            criteria
        )
        
        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(conversation_history)
        messages.append({
            "role": "user",
            "content": "Please evaluate this conversation according to the provided criteria. Provide a detailed assessment with explanation for each criterion and calculate the final grade."
        })
        
        print(f"   📝 Evaluation prompt: {len(system_prompt)} chars")
        print(f"   💬 Conversation: {len(conversation_history)} messages")
        print(f"   📊 Criteria: {len(criteria)}")
        
        response = self.client.chat.completions.create(
            model=Config.DEFAULT_MODEL,
            messages=messages,
            temperature=0.3,
            max_tokens=2000
        )
        
        evaluation_text = response.choices[0].message.content
        
        print(f"   ✅ Evaluation: {len(evaluation_text)} chars")
        
        return {
            "evaluation": evaluation_text,
            "timestamp": datetime.now().isoformat(),
            "conversation_length": len(conversation_history),
            "criteria_count": len(criteria)
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
        
        print(f"📖 History loaded: {len(history)} messages")
        for i, h in enumerate(history[-5:]):
            print(f"  {i}: role={h.get('role')}, content={h.get('content')[:50]}...")
        
        last_air_id = session.get("last_air_id")
        if not last_air_id or last_air_id == 0:
            messages_data = await xano.get_messages(message.ub_id)
            if messages_data:
                messages_data.sort(key=lambda x: x.get("created_at", 0))
                last_air_id = messages_data[-1]["id"]
            else:
                last_air_id = 0
        
        print(f"📍 Using prev_id: {last_air_id}")
        
        ai_response, run_info, msg_info, function_calls = await agent_manager.send_message(
            agent,
            message.content,
            history,
            message.ub_id,
            xano
        )
        
        saved_msg = await xano.save_message_pair(
            message.ub_id, 
            message.content,
            ai_response,
            last_air_id,
            run_info,
            msg_info
        )
        
        if not saved_msg or "id" not in saved_msg:
            print(f"ERROR: Failed to save message pair properly")
            raise HTTPException(status_code=500, detail="Failed to save message")
        
        if not function_calls or "status_grade" not in function_calls:
            await xano.update_chat_status(
                message.ub_id,
                ChatStatus.STARTED,
                last_air_id=saved_msg["id"]
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
        
        eval_instructions = block.get("eval_instructions")
        if not eval_instructions:
            raise HTTPException(status_code=400, detail="No evaluation instructions configured for this block")
        
        conversation_history = await xano.get_conversation_history(ub_id, session)
        
        print(f"📊 Starting evaluation for ub_id={ub_id}")
        
        specifications = block.get("specifications", [])
        if isinstance(specifications, str):
            try:
                specifications = json.loads(specifications)
            except:
                specifications = []
        
        criteria = block.get("eval_crit_json", [])
        if isinstance(criteria, str):
            try:
                criteria = json.loads(criteria)
            except:
                criteria = []
        
        eval_agent = EvaluationAgent(openai_client)
        evaluation = await eval_agent.evaluate_chat(
            conversation_history=conversation_history,
            eval_instructions=eval_instructions,
            specifications=specifications,
            criteria=criteria
        )
        
        return evaluation
        
    except Exception as e:
        print(f"ERROR in evaluate_chat: {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/chat/{ub_id}/history")
async def get_chat_history(ub_id: int):
    try:
        session = await xano.get_chat_session(ub_id)
        messages = await xano.get_messages(ub_id)
        
        print(f"📊 Debug /chat/{ub_id}/history:")
        print(f"   Raw messages from Xano: {len(messages)}")
        
        parsed_messages = []
        for msg in messages:
            user_content_raw = msg.get("user_content", "{}")
            try:
                user_content = json.loads(user_content_raw) if isinstance(user_content_raw, str) else user_content_raw
            except:
                user_content = {}
            
            user_text = user_content.get("text", "") if isinstance(user_content, dict) else ""
            
            ai_content_raw = msg.get("ai_content", "[]")
            try:
                ai_content = json.loads(ai_content_raw) if isinstance(ai_content_raw, str) else ai_content_raw
            except:
                ai_content = []
            
            ai_text = ""
            if isinstance(ai_content, list) and len(ai_content) > 0:
                if isinstance(ai_content[0], dict):
                    ai_text = ai_content[0].get("text", "")
            
            print(f"   Message {msg.get('id')}: user={bool(user_text)}, ai={bool(ai_text)}")
            
            if user_text or ai_text:
                parsed_messages.append({
                    "id": msg.get("id"),
                    "user_message": user_text,
                    "ai_message": ai_text,
                    "created_at": msg.get("created_at"),
                    "prev_id": msg.get("prev_id"),
                    "next_id": msg.get("next_id")
                })
        
        print(f"   Parsed messages: {len(parsed_messages)}")
        
        return {
            "messages": parsed_messages,
            "count": len(parsed_messages),
            "last_air_id": session.get("last_air_id"),
            "ub_id": ub_id,
            "status": session.get("status")
        }
    except Exception as e:
        print(f"ERROR in get_chat_history: {e}")
        import traceback
        traceback.print_exc()
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


@app.get("/debug/xano-endpoints")
async def debug_xano_endpoints():
    return {
        "message": "Xano uses /add_air for creating records only",
        "available_endpoints": {
            "get_list": "GET /air?ub_id={ub_id}",
            "create": "POST /add_air"
        },
        "note": "next_id managed by Xano, prev_id creates the chain"
    }


@app.post("/debug/test-evaluation/{ub_id}")
async def test_evaluation_prompt(ub_id: int):
    try:
        session = await xano.get_chat_session(ub_id)
        block = await xano.get_block(session["block_id"])
        
        eval_instructions = block.get("eval_instructions", "")
        specifications = block.get("specifications", [])
        criteria = block.get("eval_crit_json", [])
        
        if isinstance(specifications, str):
            specifications = json.loads(specifications)
        if isinstance(criteria, str):
            criteria = json.loads(criteria)
        
        eval_agent = EvaluationAgent(openai_client)
        prompt = eval_agent.build_evaluation_prompt(
            eval_instructions,
            specifications,
            criteria
        )
        
        return {
            "eval_instructions_length": len(eval_instructions),
            "specifications": specifications,
            "criteria": criteria,
            "full_prompt": prompt,
            "prompt_length": len(prompt)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)