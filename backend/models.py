from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field
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
    reference_image_path: Optional[str] = None
    cache_hit: bool = False

    def touch(self):
        self.updated_at = time.time()
