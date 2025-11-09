import time
from typing import Any

from pydantic import BaseModel, Field


# Image API protocol models
class ImageResponseData(BaseModel):
    b64_json: str | None = None
    url: str | None = None
    revised_prompt: str | None = None


class ImageResponse(BaseModel):
    created: int = Field(default_factory=lambda: int(time.time()))
    data: list[ImageResponseData]


class ImageGenerationsRequest(BaseModel):
    prompt: str
    model: str | None = None
    n: int | None = 1
    quality: str | None = "auto"
    response_format: str | None = "url"  # url | b64_json
    size: str | None = "1024x1024"  # e.g., 1024x1024
    style: str | None = "vivid"
    background: str | None = "auto"  # transparent | opaque | auto
    output_format: str | None = None  # png | jpeg | webp
    user: str | None = None


# Video API protocol models
class VideoResponse(BaseModel):
    id: str
    object: str = "video"
    model: str = "sora-2"
    status: str = "queued"
    progress: int = 0
    created_at: int = Field(default_factory=lambda: int(time.time()))
    size: str = "720x1280"
    seconds: str = "4"
    quality: str = "standard"
    remixed_from_video_id: str | None = None
    completed_at: int | None = None
    expires_at: int | None = None
    error: dict[str, Any] | None = None


class VideoGenerationsRequest(BaseModel):
    prompt: str
    input_reference: str | None = None
    model: str | None = None
    seconds: int | None = 4
    size: str | None = "720x1280"
    fps: int | None = None
    num_frames: int | None = None


class VideoListResponse(BaseModel):
    data: list[VideoResponse]
    object: str = "list"


class VideoRemixRequest(BaseModel):
    prompt: str
