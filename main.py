import os
from typing import Optional, Dict, Any, List
from datetime import datetime
from enum import Enum
import json
from pathlib import Path

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import httpx
from openai import OpenAI
from agents import Agent, Runner, RunConfig, RunContextWrapper, TResponseInputItem, ModelSettings, function_tool, trace, SQLiteSession
from dotenv import load_dotenv

load_dotenv()


class Config:
    XANO_BASE_URL = os.getenv("XANO_BASE_URL", "https://your-instance.xano.io/api:xxxxx")
    XANO_API_KEY = os.getenv("XANO_API_KEY", "")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    DEFAULT_MODEL = "gpt-4o"
    SESSIONS_DIR = "sessions"


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


class WorkflowState(BaseModel):
    questions: List[Dict[str, str]] = []
    current_question_index: int = 0
    answers: List[Dict[str, Any]] = []
    status: str = "active"
    ub_id: Optional[int] = None


class XanoClient:
    def __init__(self, base_url: str, api_key: str):
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        
        headers = {}
        if api_key and api_key != "your_xano_api_key_here":
            headers["Authorization"] = f"Bearer {api_key}"
        
        self.client = httpx.AsyncClient(headers=headers, timeout=30.0)
    
    async def get_block(self, block_id: int) -> Dict[str, Any]:
        response = await self.client.get(f"{self.base_url}/block/{block_id}")
        response.raise_for_status()
        return response.json()
    
    async def get_template(self, template_id: int) -> Dict[str, Any]:
        response = await self.client.get(f"{self.base_url}/template/{template_id}")
        response.raise_for_status()
        return response.json()
    
    async def get_chat_session(self, ub_id: int) -> Dict[str, Any]:
        response = await self.client.get(f"{self.base_url}/ub/{ub_id}")
        response.raise_for_status()
        return response.json()
    
    async def get_messages(self, ub_id: int, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        params = {"ub_id": ub_id}
        if limit:
            params["limit"] = limit
        response = await self.client.get(f"{self.base_url}/air", params=params)
        response.raise_for_status()
        return response.json()
    
    async def get_conversation_history(self, ub_id: int, session_data: Dict[str, Any]) -> List[TResponseInputItem]:
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
        
        conversation_history: List[TResponseInputItem] = []
        for msg in sorted_messages:
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
            
            if user_text:
                conversation_history.append({
                    "role": "user",
                    "content": [{"type": "input_text", "text": user_text}]
                })
            
            if ai_text:
                conversation_history.append({
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": ai_text}]
                })
        
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
            "user_content": json.dumps({
                "type": "text",
                "text": user_message,
                "file_url": "",
                "created_at": timestamp,
                "error_message": "",
                "error_code": "",
                "file": None,
                "images": None
            }),
            "ai_content": json.dumps([{
                "text": ai_response,
                "title": "",
                "created_at": timestamp,
                "type": None,
                "grade": None,
                "additional": ""
            }]),
            "prev_id": prev_id if prev_id and prev_id > 0 else 0
        }
        
        if run_info:
            message_record["run_info"] = json.dumps(run_info) if isinstance(run_info, dict) else run_info
        if ai_message_info:
            message_record["ai_message_info"] = json.dumps(ai_message_info) if isinstance(ai_message_info, dict) else ai_message_info
        
        try:
            response = await self.client.post(f"{self.base_url}/add_air", json=message_record)
            if response.status_code in [200, 201]:
                return response.json()
        except Exception as e:
            print(f"❌ Exception saving message pair: {e}")
        
        return {"id": timestamp, "ub_id": ub_id, "created_at": timestamp, "status": "new"}
    
    async def update_chat_status(
        self, 
        ub_id: int, 
        status: ChatStatus,
        grade: Optional[GradeResult] = None,
        last_air_id: Optional[int] = None
    ) -> Dict[str, Any]:
        update_data = {"ub_id": int(ub_id), "status": status.value}
        if grade:
            update_data["grade"] = grade.value
        if last_air_id:
            update_data["last_air_id"] = int(last_air_id)
        
        try:
            response = await self.client.post(f"{self.base_url}/update_ub", json=update_data)
            if response.status_code in [200, 201]:
                return response.json()
            return {"status": "ok", "message": "Status update failed but continuing"}
        except Exception as e:
            print(f"⚠️  Exception updating chat status: {e}")
            return {"status": "ok", "message": "Status update failed but continuing"}


class WorkflowContext:
    def __init__(
        self,
        state: WorkflowState,
        template_instructions: str,
        specifications: Any,
        ub_id: int,
        xano: XanoClient
    ):
        self.state = state
        self.template_instructions = template_instructions
        self.specifications = specifications
        self.ub_id = ub_id
        self.xano = xano


class ExaminationWorkflow:
    def __init__(self, openai_api_key: str):
        self.openai_api_key = openai_api_key
        self.sessions: Dict[int, SQLiteSession] = {}
        Path(Config.SESSIONS_DIR).mkdir(parents=True, exist_ok=True)
    
    def get_session(self, ub_id: int) -> SQLiteSession:
        if ub_id not in self.sessions:
            session_path = Path(Config.SESSIONS_DIR) / f"ub_{ub_id}.db"
            self.sessions[ub_id] = SQLiteSession(f"ub_{ub_id}", str(session_path))
        return self.sessions[ub_id]
    
    def create_tools(self, function_list: List[Dict]) -> List[callable]:
        if not function_list:
            return []
        
        tools = []
        
        for func in function_list:
            if func.get("name") == "status_grade":
                @function_tool
                def status_grade(status: str, grade: str = "") -> dict:
                    return {"status": status, "grade": grade, "success": True}
                tools.append(status_grade)
            
            elif func.get("name") == "random_numbers":
                @function_tool
                def random_numbers(array_length: int) -> dict:
                    import random
                    numbers = [random.randint(0, 1000000) for _ in range(array_length)]
                    return {"numbers": numbers, "success": True}
                tools.append(random_numbers)
        
        return tools
    
    def create_interview_agent(
        self,
        context: WorkflowContext,
        tools: List[callable],
        model: str = Config.DEFAULT_MODEL
    ) -> Agent[WorkflowContext]:
        
        def agent_instructions(run_context: RunContextWrapper[WorkflowContext], _agent: Agent):
            ctx = run_context.context
            specs_json = json.dumps(ctx.specifications, ensure_ascii=False)
            
            state_info = f"""
Current workflow state:
- Question {ctx.state.current_question_index + 1} of {len(ctx.state.questions)}
- Status: {ctx.state.status}
"""
            
            return f"""{ctx.template_instructions}

# Assignment Specifications Structure:
{specs_json}

# Specifications:
{specs_json}

{state_info}
"""
        
        agent = Agent[WorkflowContext](
            name="Interview Agent",
            instructions=agent_instructions,
            model=model,
            tools=tools,
            model_settings=ModelSettings(
                temperature=1,
                top_p=1,
                max_tokens=2048,
                store=True
            )
        )
        
        return agent
    
    def create_evaluator_agent(
        self,
        context: WorkflowContext,
        model: str = Config.DEFAULT_MODEL
    ) -> Agent[WorkflowContext]:
        
        def agent_instructions(run_context: RunContextWrapper[WorkflowContext], _agent: Agent):
            ctx = run_context.context
            current_q = ctx.state.questions[ctx.state.current_question_index] if ctx.state.current_question_index < len(ctx.state.questions) else {}
            
            return f"""You are an evaluator for an examination system.

Current question: {current_q.get('question', '')}
Expected key concepts: {current_q.get('key_concepts', '')}

Evaluate the student's answer and determine:
1. Is it correct? (yes/no/partial)
2. Should we ask a follow-up question? (yes/no)
3. Should we move to next question? (yes/no)

Return your evaluation as JSON.
"""
        
        agent = Agent[WorkflowContext](
            name="Evaluator Agent",
            instructions=agent_instructions,
            model=model,
            model_settings=ModelSettings(
                temperature=0.3,
                top_p=1,
                max_tokens=1024,
                store=True
            )
        )
        
        return agent
    
    async def run_workflow(
        self,
        block: Dict[str, Any],
        template: Dict[str, Any],
        user_message: str,
        ub_id: int,
        xano: XanoClient
    ) -> tuple[str, Dict, Dict, Optional[Dict]]:
        
        with trace(f"ExaminationWorkflow-{ub_id}"):
            template_instructions = template.get("instructions", "")
            specifications = block.get("specifications", {})
            if isinstance(specifications, str):
                try:
                    specifications = json.loads(specifications)
                except:
                    specifications = []
            
            state = WorkflowState(
                questions=specifications if isinstance(specifications, list) else [],
                current_question_index=0,
                answers=[],
                status="active",
                ub_id=ub_id
            )
            
            context = WorkflowContext(
                state=state,
                template_instructions=template_instructions,
                specifications=specifications,
                ub_id=ub_id,
                xano=xano
            )
            
            tools = self.create_tools(template.get("function_list", []))
            interview_agent = self.create_interview_agent(context, tools, template.get("model", Config.DEFAULT_MODEL))
            
            session = self.get_session(ub_id)
            
            result = await Runner.run(
                interview_agent,
                user_message,
                session=session,
                context=context,
                run_config=RunConfig(
                    trace_metadata={
                        "__trace_source__": "edtech-platform",
                        "workflow_type": "examination",
                        "block_id": block["id"],
                        "template_id": template["id"],
                        "ub_id": ub_id
                    }
                )
            )
            
            ai_response = result.final_output_as(str)
            
            function_calls = None
            if result.new_items:
                function_calls = await self.handle_tool_results(result.new_items, ub_id, xano)
            
            run_info = {
                "workflow_type": "examination",
                "workflow_name": template.get("name", "Unknown"),
                "timestamp": datetime.now().isoformat(),
                "model": template.get("model", Config.DEFAULT_MODEL),
                "question_index": state.current_question_index,
                "total_questions": len(state.questions),
                "function_calls": function_calls
            }
            
            msg_info = {
                "workflow_name": template.get("name", "Unknown"),
                "role": "assistant",
                "workflow_state": state.model_dump()
            }
            
            return ai_response, run_info, msg_info, function_calls
    
    async def handle_tool_results(
        self,
        items: List[Any],
        ub_id: Optional[int],
        xano: Optional[XanoClient]
    ) -> Dict[str, Any]:
        results = {}
        
        for item in items:
            if hasattr(item, 'type') and item.type == "tool_call_output":
                func_name = None
                result_value = None
                
                if hasattr(item, 'name'):
                    func_name = item.name
                
                if hasattr(item, 'output'):
                    result_value = item.output
                
                if not func_name:
                    continue
                
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
                        
                        if ub_id and xano and status:
                            valid_statuses = [s.value for s in ChatStatus]
                            if status not in valid_statuses:
                                status = "finished"
                            
                            chat_status = ChatStatus(status)
                            grade_result = GradeResult(grade) if grade and grade in [g.value for g in GradeResult] else None
                            await xano.update_chat_status(ub_id, chat_status, grade_result)
                        
                        results[func_name] = {"status": status, "grade": grade, "success": True}
                    except Exception as e:
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
        ub_id: int,
        xano: XanoClient,
        eval_instructions: str,
        specifications: Any,
        criteria: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        session_data = await xano.get_chat_session(ub_id)
        conversation_history = await xano.get_conversation_history(ub_id, session_data)
        
        system_prompt = self.build_evaluation_prompt(eval_instructions, specifications, criteria)
        
        messages = [{"role": "system", "content": system_prompt}]
        
        for msg in conversation_history:
            role = msg.get("role", "user")
            content_list = msg.get("content", [])
            
            text_content = ""
            for content_item in content_list:
                if isinstance(content_item, dict):
                    text_content += content_item.get("text", "")
            
            if text_content:
                messages.append({"role": role, "content": text_content})
        
        messages.append({
            "role": "user",
            "content": "Please evaluate this conversation according to the provided criteria. Provide a detailed assessment with explanation for each criterion and calculate the final grade."
        })
        
        response = self.client.chat.completions.create(
            model=Config.DEFAULT_MODEL,
            messages=messages,
            temperature=0.3,
            max_tokens=2000
        )
        
        evaluation_text = response.choices[0].message.content
        
        return {
            "evaluation": evaluation_text,
            "timestamp": datetime.now().isoformat(),
            "conversation_length": len(conversation_history),
            "criteria_count": len(criteria)
        }


app = FastAPI(title="EdTech AI Platform", description="AI-powered educational assistant platform", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

xano = XanoClient(Config.XANO_BASE_URL, Config.XANO_API_KEY)
openai_client = OpenAI(api_key=Config.OPENAI_API_KEY)
workflow = ExaminationWorkflow(Config.OPENAI_API_KEY)


@app.on_event("startup")
async def startup_event():
    Path(Config.SESSIONS_DIR).mkdir(parents=True, exist_ok=True)


@app.get("/")
async def root():
    return {"status": "operational", "version": "2.0.0", "message": "EdTech AI Platform - Workflows with OpenAI Agent SDK"}


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
async def process_student_message(message: StudentMessage, background_tasks: BackgroundTasks):
    try:
        session = await xano.get_chat_session(message.ub_id)
        block = await xano.get_block(session["block_id"])
        template_data = await xano.get_template(block["int_template_id"])
        
        last_air_id = session.get("last_air_id")
        if not last_air_id or last_air_id == 0:
            messages_data = await xano.get_messages(message.ub_id)
            if messages_data:
                messages_data.sort(key=lambda x: x.get("created_at", 0))
                last_air_id = messages_data[-1]["id"]
            else:
                last_air_id = 0
        
        ai_response, run_info, msg_info, function_calls = await workflow.run_workflow(
            block, template_data, message.content, message.ub_id, xano
        )
        
        saved_msg = await xano.save_message_pair(
            message.ub_id, message.content, ai_response, last_air_id, run_info, msg_info
        )
        
        if not saved_msg or "id" not in saved_msg:
            raise HTTPException(status_code=500, detail="Failed to save message")
        
        if not function_calls or "status_grade" not in function_calls:
            await xano.update_chat_status(message.ub_id, ChatStatus.STARTED, last_air_id=saved_msg["id"])
        
        try:
            response_json = json.loads(ai_response)
            return AssistantResponse(**response_json)
        except:
            return AssistantResponse(title="-", text=ai_response, type="interview")
        
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
            ub_id=ub_id,
            xano=xano,
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
            
            if user_text or ai_text:
                parsed_messages.append({
                    "id": msg.get("id"),
                    "user_message": user_text,
                    "ai_message": ai_text,
                    "created_at": msg.get("created_at"),
                    "prev_id": msg.get("prev_id"),
                    "next_id": msg.get("next_id")
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
        if ub_id in workflow.sessions:
            del workflow.sessions[ub_id]
        
        session_path = Path(Config.SESSIONS_DIR) / f"ub_{ub_id}.db"
        if session_path.exists():
            session_path.unlink()
        
        return {"status": "memory cleared"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    Path(Config.SESSIONS_DIR).mkdir(parents=True, exist_ok=True)
    uvicorn.run(app, host="0.0.0.0", port=8000)