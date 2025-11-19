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
    # TEST_FINISHED може не підтримуватись в Xano - перевірте базу даних
    # Якщо підтримується, розкоментуйте:
    # TEST_FINISHED = "test_finished"


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
    
    def parse_message_content(self, msg: Dict[str, Any]) -> tuple[str, str]:
        user_content_raw = msg.get("user_content", "{}")
        ai_content_raw = msg.get("ai_content", "[]")
        
        try:
            if isinstance(user_content_raw, str):
                user_content = json.loads(user_content_raw)
            else:
                user_content = user_content_raw
        except:
            user_content = {}
        
        try:
            if isinstance(ai_content_raw, str):
                ai_content = json.loads(ai_content_raw)
            else:
                ai_content = ai_content_raw
        except:
            ai_content = []
        
        user_text = ""
        if isinstance(user_content, dict) and user_content.get("text"):
            user_text = user_content.get("text", "")
        
        ai_text = ""
        if isinstance(ai_content, list) and len(ai_content) > 0:
            if isinstance(ai_content[0], dict):
                ai_text = ai_content[0].get("text", "")
        
        if user_text:
            return "user", user_text
        elif ai_text:
            return "assistant", ai_text
        else:
            return "unknown", ""
    
    async def get_conversation_history(self, ub_id: int, session_data: Dict[str, Any]) -> List[Dict[str, str]]:
        """Отримує повну історію розмови для агента"""
        messages_data = await self.get_messages(ub_id)
        
        if not messages_data:
            return []
        
        # Створюємо словник для швидкого пошуку
        messages_dict = {msg["id"]: msg for msg in messages_data}
        
        # Знаходимо всі повідомлення, які не мають prev_id (початкові)
        root_messages = [msg for msg in messages_data if not msg.get("prev_id")]
        
        if not root_messages:
            # Якщо немає початкових, сортуємо за датою
            sorted_messages = sorted(messages_data, key=lambda x: x.get("created_at", 0))
        else:
            # Будуємо ланцюжок від початкового повідомлення
            sorted_messages = []
            visited = set()
            
            def build_chain(msg_id):
                if msg_id in visited or msg_id not in messages_dict:
                    return
                visited.add(msg_id)
                msg = messages_dict[msg_id]
                sorted_messages.append(msg)
                
                # Знаходимо наступне повідомлення (де prev_id = поточний id)
                for other_msg in messages_data:
                    if other_msg.get("prev_id") == msg_id and other_msg["id"] not in visited:
                        build_chain(other_msg["id"])
            
            # Починаємо з першого root повідомлення
            build_chain(root_messages[0]["id"])
        
        # Конвертуємо в формат для агента
        conversation_history = []
        for msg in sorted_messages:
            # Парсимо user content
            user_content_raw = msg.get("user_content", "{}")
            try:
                if isinstance(user_content_raw, str):
                    user_content = json.loads(user_content_raw)
                else:
                    user_content = user_content_raw
            except:
                user_content = {}
            
            user_text = user_content.get("text", "") if isinstance(user_content, dict) else ""
            
            # Парсимо AI content
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
            
            # Додаємо обидва повідомлення якщо вони є
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
        """Зберігає пару повідомлень (користувач + AI) в один рядок"""
        
        timestamp = int(datetime.now().timestamp() * 1000)
        
        message_record = {
            "ub_id": ub_id,
            "role": "assistant",  # Основна роль - assistant, бо це фінальна відповідь
            "created_at": timestamp,
            "status": "new",
            "block_id": ub_id,
        }
        
        # User content
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
        
        # AI content
        message_record["ai_content"] = json.dumps([{
            "text": ai_response,
            "title": "",
            "created_at": timestamp,
            "type": None,
            "grade": None,
            "additional": ""
        }])
        
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
                print(f"✅ Message pair saved: id={result.get('id')}, user+ai")
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
            "ub_id": int(ub_id),  # Переконуємось що це int
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
            instructions += json.dumps(specifications.get("params_structure", []), ensure_ascii=False) if isinstance(specifications, dict) else ""
            instructions += "\n\n# Specifications: \n"
            
            specs_data = specifications.get("specifications", {}) if isinstance(specifications, dict) else {}
            instructions += json.dumps(specs_data, ensure_ascii=False)
        
        return instructions
    
    def parse_functions(self, function_list: str) -> List[Dict]:
        if not function_list:
            return []
        try:
            return json.loads(function_list)
        except:
            return []
    
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
        tool_calls_map = {}  # Зберігаємо ToolCallItem для пошуку імен
        
        # Спочатку збираємо всі ToolCallItem
        for item in items:
            item_type = type(item).__name__
            if item_type == "ToolCallItem":
                # Зберігаємо ім'я функції з ToolCallItem
                if hasattr(item, 'raw_item'):
                    raw_item = item.raw_item
                    if hasattr(raw_item, 'name'):
                        tool_calls_map[id(item)] = raw_item.name
                    elif hasattr(raw_item, 'function') and hasattr(raw_item.function, 'name'):
                        tool_calls_map[id(item)] = raw_item.function.name
        
        # Тепер обробляємо всі елементи
        for i, item in enumerate(items):
            item_type = type(item).__name__
            print(f"   📦 Item {i}: {item_type}")
            
            # Перевіряємо різні типи елементів
            if item_type == "ToolCallOutputItem":
                # Пробуємо різні способи отримати ім'я функції
                func_name = None
                result_value = None
                
                # Спосіб 1: через agent атрибут
                if hasattr(item, 'agent') and hasattr(item.agent, 'name'):
                    print(f"      Agent name: {item.agent.name}")
                
                # Спосіб 2: детально дивимось raw_item
                if hasattr(item, 'raw_item'):
                    raw_item = item.raw_item
                    print(f"      raw_item type: {type(raw_item).__name__}")
                    print(f"      raw_item attributes: {[attr for attr in dir(raw_item) if not attr.startswith('_')]}")
                    
                    # Пробуємо різні шляхи
                    if hasattr(raw_item, 'name'):
                        func_name = raw_item.name
                        print(f"      Found name in raw_item.name: {func_name}")
                    elif hasattr(raw_item, 'function'):
                        if hasattr(raw_item.function, 'name'):
                            func_name = raw_item.function.name
                            print(f"      Found name in raw_item.function.name: {func_name}")
                    elif hasattr(raw_item, 'tool_call_id'):
                        print(f"      tool_call_id: {raw_item.tool_call_id}")
                    
                    # Якщо є output, спробуємо його розпарсити
                    if hasattr(raw_item, 'output'):
                        result_value = raw_item.output
                        print(f"      raw_item.output: {result_value}")
                
                # Спосіб 3: через output
                if hasattr(item, 'output'):
                    result_value = item.output
                    print(f"      item.output: {result_value}")
                    
                    # Іноді в output є вся інформація
                    if isinstance(result_value, dict):
                        if 'status' in result_value and 'grade' in result_value:
                            func_name = 'status_grade'
                            print(f"      Detected status_grade from output structure")
                        elif 'numbers' in result_value:
                            func_name = 'random_numbers'
                            print(f"      Detected random_numbers from output structure")
                
                # Спосіб 4: пошук в попередніх ToolCallItem
                if not func_name and i > 0:
                    # Шукаємо попередній ToolCallItem
                    for j in range(i-1, -1, -1):
                        if type(items[j]).__name__ == "ToolCallItem":
                            prev_item = items[j]
                            if hasattr(prev_item, 'raw_item'):
                                prev_raw = prev_item.raw_item
                                if hasattr(prev_raw, 'name'):
                                    func_name = prev_raw.name
                                    print(f"      Found name from previous ToolCallItem: {func_name}")
                                elif hasattr(prev_raw, 'function') and hasattr(prev_raw.function, 'name'):
                                    func_name = prev_raw.function.name
                                    print(f"      Found name from previous ToolCallItem function: {func_name}")
                            break
                
                if not func_name:
                    print(f"   ⚠️  Could not extract function name from ToolCallOutputItem")
                    print(f"   Available attributes: {[attr for attr in dir(item) if not attr.startswith('_')]}")
                    continue
                
                print(f"   🎯 Tool called: {func_name}")
                print(f"   📤 Output: {result_value}")
                
                if func_name == "status_grade":
                    try:
                        if isinstance(result_value, dict):
                            status = result_value.get("status")
                            grade = result_value.get("grade", "")
                        elif isinstance(result_value, str):
                            # Якщо результат - рядок, пробуємо розпарсити як JSON
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
                            # Перевіряємо чи статус валідний для Xano
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
                        import traceback
                        traceback.print_exc()
                        results[func_name] = {"success": False, "error": str(e)}
                
                elif func_name == "random_numbers":
                    try:
                        print(f"   🎲 Function call: random_numbers -> {result_value}")
                        results[func_name] = result_value
                    except Exception as e:
                        print(f"   ❌ Error handling random_numbers: {e}")
                        results[func_name] = {"success": False, "error": str(e)}
        
        return results if results else None


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
            # specifications це список dict з questions та key_concepts
            if isinstance(specifications, list):
                for i, spec in enumerate(specifications, 1):
                    prompt += f"\n## Question {i}:\n"
                    prompt += f"Question: {spec.get('question', '')}\n"
                    prompt += f"Key Concepts: {spec.get('key_concepts', '')}\n"
            elif isinstance(specifications, dict):
                for key, value in specifications.items():
                    prompt += f"\n## {key}:\n{value}\n"
        
        if criteria:
            prompt += "\n\n# Evaluation Criteria:\n"
            for i, criterion in enumerate(criteria, 1):
                prompt += f"\n## Criterion {i}:\n"
                prompt += f"Criterion Name: {criterion.get('criterion_name', f'Criterion {i}')}\n"
                prompt += f"Max Points: {criterion.get('max_points', 0)}\n"
                prompt += f"Summary Instructions: {criterion.get('summary_instructions', '')}\n"
                prompt += f"Grading Instructions: {criterion.get('grading_instructions', '')}\n"
        
        return prompt
    
    async def evaluate_chat(
        self,
        conversation_history: List[Dict[str, str]],
        template: Dict[str, Any],
        specifications: Any,  # Може бути list або dict
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
        
        print(f"📖 History loaded: {len(history)} messages")
        for i, h in enumerate(history[-5:]):  # Показуємо останні 5
            print(f"  {i}: role={h.get('role')}, content={h.get('content')[:50]}...")
        
        # Отримуємо prev_id для ланцюжка
        messages_data = await xano.get_messages(message.ub_id, limit=1)
        last_air_id = messages_data[0]["id"] if messages_data else None
        
        # Відправляємо повідомлення агенту
        ai_response, run_info, msg_info, function_calls = await agent_manager.send_message(
            agent,
            message.content,
            history,
            message.ub_id,
            xano
        )
        
        # Зберігаємо пару повідомлень в один рядок
        saved_msg = await xano.save_message_pair(
            message.ub_id, 
            message.content,  # user message
            ai_response,      # ai response
            last_air_id,
            run_info,
            msg_info
        )
        
        if not saved_msg or "id" not in saved_msg:
            print(f"ERROR: Failed to save message pair properly")
            raise HTTPException(status_code=500, detail="Failed to save message")
        
        # Оновлюємо статус якщо функція не викликалась
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
        
        eval_template_id = block.get("eval_template_id")
        if not eval_template_id:
            raise HTTPException(status_code=400, detail="No evaluation template configured for this block")
        
        eval_template_data = await xano.get_template(eval_template_id)
        
        conversation_history = await xano.get_conversation_history(ub_id, session)
        
        print(f"📊 Starting evaluation for ub_id={ub_id}")
        print(f"   Block specifications type: {type(block.get('specifications'))}")
        print(f"   Criteria type: {type(eval_template_data.get('eval_crit_json'))}")
        
        # Specifications можуть бути списком або dict
        specifications = block.get("specifications", [])
        if isinstance(specifications, str):
            try:
                specifications = json.loads(specifications)
            except:
                specifications = []
        
        # Criteria - це список
        criteria = eval_template_data.get("eval_crit_json", [])
        if isinstance(criteria, str):
            try:
                criteria = json.loads(criteria)
            except:
                criteria = []
        
        eval_agent = EvaluationAgent(openai_client)
        evaluation = await eval_agent.evaluate_chat(
            conversation_history=conversation_history,
            template=eval_template_data,
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
        
        # Парсимо повідомлення для зручного відображення
        parsed_messages = []
        for msg in messages:
            # Парсимо user content
            user_content_raw = msg.get("user_content", "{}")
            try:
                user_content = json.loads(user_content_raw) if isinstance(user_content_raw, str) else user_content_raw
            except:
                user_content = {}
            
            user_text = user_content.get("text", "") if isinstance(user_content, dict) else ""
            
            # Парсимо AI content
            ai_content_raw = msg.get("ai_content", "[]")
            try:
                ai_content = json.loads(ai_content_raw) if isinstance(ai_content_raw, str) else ai_content_raw
            except:
                ai_content = []
            
            ai_text = ""
            if isinstance(ai_content, list) and len(ai_content) > 0:
                if isinstance(ai_content[0], dict):
                    ai_text = ai_content[0].get("text", "")
            
            # Створюємо запис для кожної пари
            if user_text or ai_text:
                parsed_messages.append({
                    "id": msg.get("id"),
                    "user_message": user_text,
                    "ai_message": ai_text,
                    "created_at": msg.get("created_at"),
                    "prev_id": msg.get("prev_id")
                })
        
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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)