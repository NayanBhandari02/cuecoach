from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field

Topic = Literal[
    "rules",
    "stance_bridge",
    "aiming",
    "cue_ball_control",
    "break",
    "safety",
    "strategy",
    "drills",
    "equipment",
    "misc",
]


SkillLevel = Literal["beginner", "intermediate", "advanced"]


class Chunk(BaseModel):
    chunk_id: str = Field(..., description="Stable unique id for this chunk")
    doc_id: str = Field(..., description="Stable id for the source document")
    source: str = Field(..., description="Human-readable source name (e.g., DrDave, WPA)")
    title: str = Field(..., description="Document title")
    section: Optional[str] = Field(None, description="Section heading if available")
    url: Optional[str] = Field(None, description="Canonical URL if known")

    topic: Topic = "misc"
    skill_level: SkillLevel = "beginner"

    text: str = Field(..., description="Chunk text used for embeddings and retrieval")
