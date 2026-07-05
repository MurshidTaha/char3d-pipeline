from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field, computed_field
import uuid
import time


class OrganismType(str, Enum):
    human = "human"
    animal = "animal"
    fantasy_creature = "fantasy_creature"
    stylized_humanoid = "stylized_humanoid"
    robot_mechanical = "robot_mechanical"


class StyleMode(str, Enum):
    photoreal = "photoreal"
    stylized = "stylized"
    anime = "anime"


class BackendChoice(str, Enum):
    charactergen = "charactergen"
    triposr = "triposr"
    instantmesh = "instantmesh"
    trellis = "trellis"


class PoseHint(str, Enum):
    auto = "auto"
    a_pose = "a_pose"
    t_pose = "t_pose"


class JobStatus(str, Enum):
    queued = "queued"
    running_generation = "running_generation"
    running_rigging = "running_rigging"
    running_upscale = "running_upscale"
    packaging = "packaging"
    complete = "complete"
    failed = "failed"


# Rough, evenly-spaced progress percentage per stage — good enough for a
# frontend progress bar. Not meant to be precisely time-weighted (generation
# takes far longer than packaging in practice), just monotonically increasing
# so the bar never jumps backward and always reaches 100 on completion.
STAGE_PROGRESS: dict[str, int] = {
    "queued": 0,
    "running_generation": 15,
    "running_rigging": 55,
    "running_upscale": 75,
    "packaging": 90,
    "complete": 100,
    "failed": 100,
}


class GenerationRequest(BaseModel):
    character_name: str
    character_description: str
    organism_type: OrganismType
    style_mode: StyleMode
    pose_hint: PoseHint = PoseHint.auto
    backend_override: Optional[BackendChoice] = None
    quick_preview: bool = False
    # reference_image is handled as a separate multipart file upload in the endpoint,
    # not part of this JSON body


class BatchGenerationRequest(BaseModel):
    """Batch mode = N jobs queued sequentially (T4-sharing constraint, per spec)."""
    jobs: list[GenerationRequest]


class Job(BaseModel):
    job_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    request: GenerationRequest
    status: JobStatus = JobStatus.queued
    backend_used: Optional[str] = None
    routing_reason: Optional[str] = None
    created_at: float = Field(default_factory=time.time)
    updated_at: float = Field(default_factory=time.time)
    error: Optional[str] = None
    output_zip_path: Optional[str] = None
    output_glb_path: Optional[str] = None
    reference_image_path: Optional[str] = None
    cache_hit: bool = False
    log: list[str] = Field(default_factory=list)

    def touch(self):
        self.updated_at = time.time()

    def log_line(self, message: str) -> None:
        """Append a timestamped, human-readable status line for the frontend
        log panel, and bump updated_at at the same time."""
        stamp = time.strftime("%H:%M:%S", time.localtime())
        self.log.append(f"[{stamp}] {message}")
        self.touch()

    @computed_field
    @property
    def progress_pct(self) -> int:
        return STAGE_PROGRESS.get(self.status.value if hasattr(self.status, "value") else self.status, 0)
