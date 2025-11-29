import os
from typing import Optional, Dict, Any, List
from datetime import datetime
from enum import Enum
import json

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import httpx
from agents import Agent, Runner, RunConfig, RunContextWrapper, ModelSettings, function_tool, trace
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
            if response.status_code == 200:
                data = response.json()
                if data and not data.get('error'):
                    data['questions'] = json.loads(data['questions']) if isinstance(data.get('questions'), str) else data.get('questions', [])
                    data['answers'] = json.loads(data['answers']) if isinstance(data.get('answers'), str) else data.get('answers', [])
                    return WorkflowState(**data)
        except Exception as e:
            print(f"Error loading workflow state: {e}")
        return None
    
    async def save_workflow_state(self, state: WorkflowState):
        data = {
            "ub_id": state.ub_id,
            "block_id": state.block_id,
            "current_question_index": state.current_question_index,
            "questions": json.dumps(state.questions, ensure_ascii=False),
            "answers": json.dumps(state.answers, ensure_ascii=False),
            "follow_up_count": state.follow_up_count,
            "max_follow_ups": state.max_follow_ups,
            "status": state.status
        }
        response = await self.client.post(f"{self.base_url}/save_workflow_state", json=data)
        return response.json() if response.status_code in [200, 201] else None
    
    async def get_messages(self, ub_id: int) -> List[Dict[str, Any]]:
        response = await self.client.get(f"{self.base_url}/air", params={"ub_id": ub_id})
        response.raise_for_status()
        return response.json()
    
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
        response = await self.client.post(f"{self.base_url}/add_air", json=message_record)
        return response.json() if response.status_code in [200, 201] else {"id": timestamp}
    
    async def update_chat_status(self, ub_id: int, status: Optional[ChatStatus] = None, grade: Optional[str] = None, last_air_id: Optional[int] = None):
        update_data = {"ub_id": int(ub_id)}
        
        if status:
            update_data["status"] = status.value
        if grade is not None:
            update_data["grade"] = grade
        if last_air_id:
            update_data["last_air_id"] = int(last_air_id)
        
        try:
            response = await self.client.post(f"{self.base_url}/update_ub", json=update_data)
            if response.status_code in [200, 201]:
                return response.json()
        except Exception as e:
            print(f"Status update error: {e}")
        return None


class WorkflowContext:
    def __init__(self, state: WorkflowState):
        self.state = state


class EvaluationContext:
    def __init__(
        self,
        workflow_state: WorkflowState,
        eval_instructions: str,
        criteria: List[Dict[str, Any]]
    ):
        self.workflow_state = workflow_state
        self.eval_instructions = eval_instructions
        self.criteria = criteria


class ExaminationWorkflow:
    def __init__(self, openai_api_key: str):
        self.openai_api_key = openai_api_key
    
    def create_interviewer_agent(self, context: WorkflowContext, model: str) -> Agent[WorkflowContext]:
        def agent_instructions(run_context: RunContextWrapper[WorkflowContext], _agent: Agent):
            ctx = run_context.context
            
            if ctx.state.current_question_index >= len(ctx.state.questions):
                return "The exam is complete. Thank the student."
            
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
            
            if state.status == "finished":
                return "Іспит вже завершено."
            
            if state.current_question_index >= len(state.questions):
                state.status = "finished"
                await xano.save_workflow_state(state)
                await xano.update_chat_status(ub_id, status=ChatStatus.FINISHED)
                return "Вітаю! Ви відповіли на всі питання. Іспит завершено."
            
            context = WorkflowContext(state=state)
            
            if not state.answers or state.answers[-1].get('evaluation', {}).get('complete', False):
                interviewer = self.create_interviewer_agent(context, template.get("model", Config.DEFAULT_MODEL))
                result = await Runner.run(interviewer, "", context=context)
                response = result.final_output_as(str)
                
                state.answers.append({
                    "question_index": state.current_question_index,
                    "answer": "",
                    "timestamp": datetime.now().isoformat(),
                    "evaluation": {}
                })
                state.follow_up_count = 0
                await xano.save_workflow_state(state)
                return response
            
            state.answers[-1]['answer'] = user_message
            state.answers[-1]['timestamp'] = datetime.now().isoformat()
            
            evaluator = self.create_evaluator_agent(context, template.get("model", Config.DEFAULT_MODEL))
            eval_result = await Runner.run(evaluator, "", context=context)
            evaluation = eval_result.final_output.model_dump()
            
            state.answers[-1]['evaluation'] = evaluation
            
            if evaluation['complete']:
                state.current_question_index += 1
                state.follow_up_count = 0
                
                if state.current_question_index >= len(state.questions):
                    state.status = "finished"
                    await xano.save_workflow_state(state)
                    await xano.update_chat_status(ub_id, status=ChatStatus.FINISHED)
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
                    state.current_question_index += 1
                    state.follow_up_count = 0
                    
                    if state.current_question_index >= len(state.questions):
                        state.status = "finished"
                        await xano.save_workflow_state(state)
                        await xano.update_chat_status(ub_id, status=ChatStatus.FINISHED)
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
                    state.follow_up_count += 1
                    await xano.save_workflow_state(state)
                    
                    interviewer = self.create_interviewer_agent(context, template.get("model", Config.DEFAULT_MODEL))
                    result = await Runner.run(interviewer, f"Student answer was incomplete. Ask a follow-up question to clarify.", context=context)
                    return result.final_output_as(str)


class EvaluationWorkflow:
    def __init__(self, openai_api_key: str):
        self.openai_api_key = openai_api_key
    
    def create_evaluation_agent(self, context: EvaluationContext, model: str) -> Agent[EvaluationContext]:
        def agent_instructions(run_context: RunContextWrapper[EvaluationContext], _agent: Agent):
            ctx = run_context.context

            criteria_text = ""
            for i, crit in enumerate(ctx.criteria):
                criteria_text += f"\n## Criterion {i+1}"
                if crit.get('criterion_name'):
                    criteria_text += f": {crit['criterion_name']}"
                criteria_text += f"\nMax Points: {crit.get('max_points', 0)}\n"
                if crit.get('summary_instructions'):
                    criteria_text += f"Summary Instructions: {crit['summary_instructions']}\n"
                if crit.get('grading_instructions'):
                    criteria_text += f"Grading Instructions: {crit['grading_instructions']}\n"
                criteria_text += "\n"

            conversation_text = ""
            for i, ans in enumerate(ctx.workflow_state.answers):
                q_index = ans.get('question_index', i)
                
                if q_index < len(ctx.workflow_state.questions):
                    question = ctx.workflow_state.questions[q_index]
                    
                    conversation_text += f"\n{'='*60}\n"
                    conversation_text += f"Exchange {i+1}:\n"
                    conversation_text += f"{'='*60}\n\n"
                    conversation_text += f"**Question:** {question.get('question', 'N/A')}\n"
                    conversation_text += f"**Expected key concepts:** {question.get('key_concepts', 'N/A')}\n\n"
                    conversation_text += f"**Student answer:** {ans.get('answer', 'No answer provided')}\n\n"
                    
                    evaluation = ans.get('evaluation', {})
                    if evaluation:
                        conversation_text += f"**Workflow evaluation:**\n"
                        conversation_text += f"  - Answer was complete: {evaluation.get('complete', False)}\n"
                        if evaluation.get('missing_concepts'):
                            conversation_text += f"  - Missing concepts: {', '.join(evaluation.get('missing_concepts', []))}\n"
                        if evaluation.get('needs_clarification'):
                            conversation_text += f"  - Needed clarification: {evaluation.get('needs_clarification', False)}\n"
                    
                    conversation_text += "\n"
            
            return f"""You are an evaluation assistant for an educational platform.

{ctx.eval_instructions}

# Conversation History
{conversation_text}

# Evaluation Criteria
{criteria_text}

# Your Task

Evaluate the student's performance according to the provided criteria.

For each criterion:
1. Review the student's answers and the workflow evaluation results
2. Assess how well they met the criterion
3. Assign a grade (0 to max_points for that criterion)
4. Provide clear reasoning

Format your response as:

# Evaluation Report

## Criterion 1: [Name]
**Assessment:** [Detailed assessment based on answers]
**Grade:** X/Y points
**Reasoning:** [Why this grade was assigned]

## Criterion 2: [Name]
**Assessment:** [Detailed assessment]
**Grade:** X/Y points
**Reasoning:** [Explanation]

# Summary
**Total Score:** X/Y points
**Overall Performance:** [Brief summary]
**Recommendations:** [Optional suggestions for improvement]"""
        
        return Agent[EvaluationContext](
            name="EvaluationAgent",
            instructions=agent_instructions,
            model=model,
            model_settings=ModelSettings(temperature=0.3, max_tokens=2048)
        )
    
    async def run_evaluation(
        self,
        ub_id: int,
        workflow_state: WorkflowState,
        eval_instructions: str,
        criteria: List[Dict[str, Any]],
        model: str = Config.DEFAULT_MODEL
    ) -> str:
        with trace(f"Evaluation-{ub_id}"):
            context = EvaluationContext(
                workflow_state=workflow_state,
                eval_instructions=eval_instructions,
                criteria=criteria
            )
            
            agent = self.create_evaluation_agent(context, model)
            
            result = await Runner.run(
                agent,
                "",
                context=context,
                run_config=RunConfig(
                    trace_metadata={
                        "__trace_source__": "edtech-evaluation",
                        "ub_id": ub_id
                    }
                )
            )
            
            return result.final_output_as(str)


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
evaluation_workflow = EvaluationWorkflow(Config.OPENAI_API_KEY)


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
        print(f"ERROR: {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat/{ub_id}/evaluate")
async def evaluate_chat(ub_id: int):
    try:
        session = await xano.get_chat_session(ub_id)
        
        if session.get('grade'):
            return {
                "evaluation": session['grade'],
                "timestamp": datetime.now().isoformat(),
                "conversation_length": 0,
                "criteria_count": 0,
                "cached": True
            }
        
        block = await xano.get_block(session["block_id"])
        
        eval_instructions = block.get("eval_instructions")
        if not eval_instructions:
            raise HTTPException(status_code=400, detail="No evaluation instructions configured")
        
        workflow_state = await xano.get_workflow_state(ub_id)
        if not workflow_state:
            raise HTTPException(status_code=404, detail="No workflow state found")
        
        criteria = block.get("eval_crit_json", [])
        if isinstance(criteria, str):
            try:
                criteria = json.loads(criteria)
            except:
                criteria = []
        
        evaluation_text = await evaluation_workflow.run_evaluation(
            ub_id=ub_id,
            workflow_state=workflow_state,
            eval_instructions=eval_instructions,
            criteria=criteria
        )
        
        await xano.update_chat_status(ub_id, grade=evaluation_text)
        
        return {
            "evaluation": evaluation_text,
            "timestamp": datetime.now().isoformat(),
            "conversation_length": len(workflow_state.answers),
            "criteria_count": len(criteria),
            "cached": False
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/chat/{ub_id}/state")
async def get_workflow_state(ub_id: int):
    try:
        state = await xano.get_workflow_state(ub_id)
        if state:
            return state.model_dump()
        else:
            raise HTTPException(status_code=404, detail="Workflow state not found")
    except HTTPException:
        raise
    except Exception as e:
        print(f"ERROR: {e}")
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