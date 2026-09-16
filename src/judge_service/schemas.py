"""Pydantic request/response models for the judge service."""
from typing import Literal

from pydantic import BaseModel, Field


class JudgeRequest(BaseModel):
    question: str
    answer: str
    # Faithfulness needs the retrieved passages the answer was built from;
    # answer_relevance judges the question/answer pair alone, so this is
    # empty for that criterion (see prompts.py).
    context: list[str] = []
    criterion: Literal["faithfulness", "answer_relevance"]


class JudgeResponse(BaseModel):
    score: float = Field(ge=0.0, le=1.0)
    rationale: str


# What the LLM's JSON-mode reply is validated against before being trusted --
# same pattern as generate.py's LLMAnswer/_parse_json_answer.
class LLMJudgeReply(BaseModel):
    score: float = Field(ge=0.0, le=1.0)
    rationale: str
