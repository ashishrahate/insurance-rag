"""LLM-as-judge service (Phase 5): POST /judge scores a generated answer for
Faithfulness or Answer Relevance.

Runs as its own process on its own port, deliberately separate from
src/api/main.py -- the whole point is that which model judges
(JUDGE_LLM_PROVIDER/JUDGE_LLM_MODEL, config/settings.py) is independently
configurable from which model generates answers (LLM_PROVIDER/LLM_MODEL),
without either touching the other.

Deliberately light, not hardened: this is a batch eval tool called by
src/evaluation/run_full_eval.py (via judge_client.py), not user-facing
production traffic. One try/except at the call site is the client's whole
resilience story -- no retries, no circuit breaker, no queue. A down judge
fails that one question's score, not the run.

No dedicated DB: every /judge call is logged via the existing
src/observability/logger.py:log_run() -> logs/runs.jsonl, same append-only
mechanism every other op in this project already uses.

Run: uvicorn src.judge_service.main:app --port 8100
(requires scripts/start.sh's Ollama up first)
"""
import json

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from src.judge_service.prompts import build_answer_relevance_messages, build_faithfulness_messages
from src.judge_service.schemas import JudgeRequest, JudgeResponse, LLMJudgeReply
from src.observability.logger import get_logger, log_run, new_correlation_id
from src.providers import get_judge_provider

_log = get_logger("judge_service")

app = FastAPI(title="insurance-rag judge service")


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={"error": "internal_error", "detail": str(exc)},
    )


@app.get("/healthcheck")
def healthcheck():
    return {"status": "ok"}


def _parse_reply(content: str) -> LLMJudgeReply:
    return LLMJudgeReply.model_validate(json.loads(content))


def _score(messages: list[dict]) -> LLMJudgeReply:
    provider = get_judge_provider()
    resp = provider.chat(messages, json_mode=True)
    raw_content = resp["content"].strip()
    try:
        return _parse_reply(raw_content)
    except (json.JSONDecodeError, ValidationError):
        # One corrective retry, same pattern as generate.py's JSON-mode path.
        retry_messages = messages + [
            {"role": "assistant", "content": raw_content},
            {"role": "user", "content": (
                'That reply was not valid JSON of the shape '
                '{"score": <float 0.0-1.0>, "rationale": "..."}. '
                "Reply again with ONLY that JSON object, nothing else."
            )},
        ]
        resp = provider.chat(retry_messages, json_mode=True)
        raw_content = resp["content"].strip()
        return _parse_reply(raw_content)  # let a second failure raise -> 500


@app.post("/judge", response_model=JudgeResponse)
def judge(req: JudgeRequest):
    cid = new_correlation_id()

    if req.criterion == "faithfulness":
        messages = build_faithfulness_messages(req.question, req.answer, req.context)
    else:
        messages = build_answer_relevance_messages(req.question, req.answer)

    try:
        reply = _score(messages)
    except (json.JSONDecodeError, ValidationError) as e:
        raise HTTPException(status_code=502, detail=f"judge returned malformed output: {e}")

    log_run({
        "op": "judge_score",
        "correlation_id": cid,
        "criterion": req.criterion,
        "question": req.question,
        "score": reply.score,
        "rationale": reply.rationale,
    })

    return JudgeResponse(score=reply.score, rationale=reply.rationale)
