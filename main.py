import os
from typing import Optional, Dict, Any, List
from datetime import datetime
from enum import Enum
import json

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import httpx
from agents import Agent, Runner, RunConfig, RunContextWrapper, ModelSettings, trace
from dotenv import load_dotenv

load_dotenv()


class Config:
    XANO_BASE_URL = os.getenv("XANO_BASE_URL", "")
    XANO_API_KEY = os.getenv("XANO_API_KEY", "")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    DEFAULT_MODEL = "gpt-4o"


class ChatStatus(str, Enum):
    IDLE = "idle"
    STARTED = "started"
    FINISHED = "finished"
    BLOCKED = "blocked"


class StudentMessage(BaseModel):
    ub_id: int
    content: str


class AssistantResponse(BaseModel):
    title: str = "-"
    text: str
    type: str = "interview"


class WorkflowState(BaseModel):
    ub_id: int
    block_id: int
    current_question_index: int = 0
    questions: List[Dict[str, str]] = []
    answers: List[Dict[str, Any]] = []
    follow_up_count: int = 0
    max_follow_ups: int = 3
    status: str = "active"


class XanoClient:
    def __init__(self, base_url: str, api_key: str):
        self.base_url = base_url.rstrip('/')
        headers = {}
        if api_key:
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
    
    async def get_workflow_state(self, ub_id: int) -> Optional[WorkflowState]:
        try:
            response = await self.client.get(f"{self.base_url}/get_workflow_state/{ub_id}")
            print(f"[DEBUG] get_workflow_state status: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                print(f"[DEBUG] workflow_state raw data: {data}")

                if isinstance(data, dict) and 'error' in data:
                    print(f"[DEBUG] No state found in Xano")
                    return None
                
                if not data:
                    return None
                
                state_data = data if isinstance(data, dict) else (data[0] if isinstance(data, list) and data else None)
                
                if not state_data:
                    return None
                
                if 'questions' in state_data and isinstance(state_data['questions'], str):
                    state_data['questions'] = json.loads(state_data['questions'])
                if 'answers' in state_data and isinstance(state_data['answers'], str):
                    state_data['answers'] = json.loads(state_data['answers'])
                
                print(f"[DEBUG] parsed state: q_index={state_data.get('current_question_index')}, follow_ups={state_data.get('follow_up_count')}")
                return WorkflowState(**state_data)
            
            return None
        except Exception as e:
            print(f"[ERROR] get_workflow_state: {e}")
            return None
    
    async def save_workflow_state(self, state: WorkflowState):
        try:
            # Xano endpoint сам визначить чи робити Add чи Edit через Conditional
            data = {
                "ub_id": state.ub_id,
                "block_id": state.block_id,
                "current_question_index": state.current_question_index,
                "questions": state.questions,  # JSON object
                "answers": state.answers,      # JSON object
                "follow_up_count": state.follow_up_count,
                "status": state.status
            }
            
            print(f"[DEBUG] Saving state: ub_id={data['ub_id']}, q_index={data['current_question_index']}, follow_ups={data['follow_up_count']}")
            
            response = await self.client.post(f"{self.base_url}/save_workflow_state", json=data)
            print(f"[DEBUG] save_workflow_state status: {response.status_code}")
            print(f"[DEBUG] save_workflow_state response: {response.text}")
            
            if response.status_code in [200, 201]:
                result = response.json()
                print(f"[DEBUG] State saved successfully")
                return result
            else:
                print(f"[ERROR] Failed to save state: {response.text}")
                return None
        except Exception as e:
            print(f"[ERROR] save_workflow_state: {e}")
            import traceback
            traceback.print_exc()
            return None
        
    async def save_message_pair(self, ub_id: int, user_message: str, ai_response: str, prev_id: Optional[int] = None) -> Dict[str, Any]:
        timestamp = int(datetime.now().timestamp() * 1000)
        message_record = {
            "ub_id": ub_id,
            "created_at": timestamp,
            "status": "new",
            "user_content": json.dumps({"type": "text", "text": user_message, "created_at": timestamp}),
            "ai_content": json.dumps([{"text": ai_response, "title": "", "created_at": timestamp}]),
            "prev_id": prev_id if prev_id else 0
        }
        response = await self.client.post(f"{self.base_url}/air", json=message_record)
        return response.json() if response.status_code in [200, 201] else {"id": timestamp}
    
    async def update_chat_status(self, ub_id: int, status: ChatStatus, grade: Optional[str] = None):
        update_data = {"ub_id": int(ub_id), "status": status.value}
        if grade:
            update_data["grade"] = grade
        try:
            await self.client.post(f"{self.base_url}/ub/update", json=update_data)
        except Exception as e:
            print(f"[ERROR] Status update: {e}")


class WorkflowContext:
    def __init__(self, state: WorkflowState):
        self.state = state


class ExaminationWorkflow:
    def __init__(self, openai_api_key: str):
        self.openai_api_key = openai_api_key
    
    def create_interviewer_agent(self, context: WorkflowContext, model: str) -> Agent[WorkflowContext]:
        def agent_instructions(run_context: RunContextWrapper[WorkflowContext], _agent: Agent):
            ctx = run_context.context
            current_q = ctx.state.questions[ctx.state.current_question_index]
            
            return f"""You are an examiner conducting an oral exam.

Current question: {current_q['question']}
Expected key concepts: {current_q['key_concepts']}

Rules:
- Ask the question clearly in Ukrainian
- Do NOT reveal the key concepts
- Do NOT give correct answers
- If answer is incomplete, ask open-ended follow-up (max {ctx.state.max_follow_ups})
- Be neutral and professional

Ask the question now."""
        
        return Agent[WorkflowContext](
            name="Interviewer",
            instructions=agent_instructions,
            model=model,
            model_settings=ModelSettings(temperature=1, top_p=1, max_tokens=1024)
        )
    
    def create_evaluator_agent(self, context: WorkflowContext, model: str) -> Agent[WorkflowContext]:
        def agent_instructions(run_context: RunContextWrapper[WorkflowContext], _agent: Agent):
            ctx = run_context.context
            current_q = ctx.state.questions[ctx.state.current_question_index]
            last_answer = ctx.state.answers[-1] if ctx.state.answers else {}
            
            return f"""You are an evaluator.

Question: {current_q['question']}
Key concepts: {current_q['key_concepts']}
Student answer: {last_answer.get('answer', '')}

Evaluate:
1. Does answer contain ALL key concepts? (yes/no)
2. If no, which concepts are missing?
3. Should we ask follow-up or move next?

Return JSON:
{{
  "complete": true/false,
  "missing_concepts": [...],
  "needs_clarification": true/false
}}"""
        
        class EvalOutput(BaseModel):
            complete: bool
            missing_concepts: List[str]
            needs_clarification: bool
        
        return Agent[WorkflowContext](
            name="Evaluator",
            instructions=agent_instructions,
            model=model,
            output_type=EvalOutput,
            model_settings=ModelSettings(temperature=0.3, max_tokens=512)
        )
    
    async def run_workflow(self, block: Dict, template: Dict, user_message: str, ub_id: int, xano: XanoClient) -> str:
        with trace(f"Workflow-{ub_id}"):
            specifications = block.get("specifications", [])
            if isinstance(specifications, str):
                specifications = json.loads(specifications)
            
            state = await xano.get_workflow_state(ub_id)
            
            if not state:
                print(f"[DEBUG] Creating new workflow state for ub_id={ub_id}")
                state = WorkflowState(
                    ub_id=ub_id,
                    block_id=block["id"],
                    questions=specifications,
                    current_question_index=0,
                    answers=[],
                    follow_up_count=0,
                    status="active"
                )
                await xano.save_workflow_state(state)
                
                context = WorkflowContext(state=state)
                interviewer = self.create_interviewer_agent(context, template.get("model", Config.DEFAULT_MODEL))
                result = await Runner.run(interviewer, "", context=context)
                response = result.final_output_as(str)
                
                state.answers.append({
                    "question_index": state.current_question_index,
                    "answer": "",
                    "timestamp": datetime.now().isoformat(),
                    "evaluation": {}
                })
                await xano.save_workflow_state(state)
                return response
            
            print(f"[DEBUG] Loaded existing state: q_index={state.current_question_index}, follow_ups={state.follow_up_count}")
            
            context = WorkflowContext(state=state)
            
            if state.answers and not state.answers[-1].get('answer'):
                state.answers[-1]['answer'] = user_message
                state.answers[-1]['timestamp'] = datetime.now().isoformat()
            else:
                state.answers.append({
                    "question_index": state.current_question_index,
                    "answer": user_message,
                    "timestamp": datetime.now().isoformat(),
                    "evaluation": {}
                })
            
            evaluator = self.create_evaluator_agent(context, template.get("model", Config.DEFAULT_MODEL))
            eval_result = await Runner.run(evaluator, "", context=context)
            evaluation = eval_result.final_output.model_dump()
            
            print(f"[DEBUG] Evaluation result: {evaluation}")
            
            state.answers[-1]['evaluation'] = evaluation
            
            if evaluation['complete']:
                print(f"[DEBUG] Answer complete, moving to next question")
                state.current_question_index += 1
                state.follow_up_count = 0
                
                if state.current_question_index >= len(state.questions):
                    state.status = "finished"
                    await xano.save_workflow_state(state)
                    await xano.update_chat_status(ub_id, ChatStatus.FINISHED)
                    return "Вітаю! Ви відповіли на всі питання. Іспит завершено."
                
                await xano.save_workflow_state(state)
                
                interviewer = self.create_interviewer_agent(context, template.get("model", Config.DEFAULT_MODEL))
                result = await Runner.run(interviewer, "", context=context)
                response = result.final_output_as(str)
                
                state.answers.append({
                    "question_index": state.current_question_index,
                    "answer": "",
                    "timestamp": datetime.now().isoformat(),
                    "evaluation": {}
                })
                await xano.save_workflow_state(state)
                return response
            
            else:
                if state.follow_up_count >= state.max_follow_ups:
                    print(f"[DEBUG] Max follow-ups reached, moving to next question")
                    state.current_question_index += 1
                    state.follow_up_count = 0
                    
                    if state.current_question_index >= len(state.questions):
                        state.status = "finished"
                        await xano.save_workflow_state(state)
                        await xano.update_chat_status(ub_id, ChatStatus.FINISHED)
                        return "Іспит завершено."
                    
                    await xano.save_workflow_state(state)
                    
                    interviewer = self.create_interviewer_agent(context, template.get("model", Config.DEFAULT_MODEL))
                    result = await Runner.run(interviewer, "", context=context)
                    response = result.final_output_as(str)
                    
                    state.answers.append({
                        "question_index": state.current_question_index,
                        "answer": "",
                        "timestamp": datetime.now().isoformat(),
                        "evaluation": {}
                    })
                    await xano.save_workflow_state(state)
                    return response
                
                else:
                    print(f"[DEBUG] Answer incomplete, asking follow-up ({state.follow_up_count + 1}/{state.max_follow_ups})")
                    state.follow_up_count += 1
                    await xano.save_workflow_state(state)
                    
                    interviewer = self.create_interviewer_agent(context, template.get("model", Config.DEFAULT_MODEL))
                    result = await Runner.run(interviewer, f"Student answer was incomplete. Ask a follow-up question to clarify.", context=context)
                    response = result.final_output_as(str)
                    
                    state.answers.append({
                        "question_index": state.current_question_index,
                        "answer": "",
                        "timestamp": datetime.now().isoformat(),
                        "evaluation": {}
                    })
                    await xano.save_workflow_state(state)
                    return response


app = FastAPI(title="EdTech AI Platform", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

xano = XanoClient(Config.XANO_BASE_URL, Config.XANO_API_KEY)
workflow = ExaminationWorkflow(Config.OPENAI_API_KEY)


@app.get("/")
async def root():
    return {"status": "operational", "version": "2.0.0"}


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "xano_configured": bool(Config.XANO_BASE_URL),
        "openai_configured": bool(Config.OPENAI_API_KEY)
    }


@app.post("/chat/message", response_model=AssistantResponse)
async def process_student_message(message: StudentMessage):
    try:
        session = await xano.get_chat_session(message.ub_id)
        block = await xano.get_block(session["block_id"])
        template_data = await xano.get_template(block["int_template_id"])
        
        ai_response = await workflow.run_workflow(block, template_data, message.content, message.ub_id, xano)
        
        messages_data = await xano.get_messages(message.ub_id)
        last_air_id = messages_data[-1]["id"] if messages_data else 0
        
        await xano.save_message_pair(message.ub_id, message.content, ai_response, last_air_id)
        
        return AssistantResponse(title="-", text=ai_response, type="interview")
        
    except Exception as e:
        print(f"[ERROR] process_student_message: {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/chat/{ub_id}/state")
async def get_workflow_state_endpoint(ub_id: int):
    try:
        state = await xano.get_workflow_state(ub_id)
        if state:
            return state.model_dump()
        else:
            raise HTTPException(status_code=404, detail="Workflow state not found")
    except HTTPException:
        raise
    except Exception as e:
        print(f"[ERROR] get_workflow_state endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/chat/{ub_id}/history")
async def get_chat_history(ub_id: int):
    try:
        messages = await xano.get_messages(ub_id)
        return {"messages": messages, "count": len(messages)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)