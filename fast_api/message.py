from pydantic import BaseModel, Field


class Message(BaseModel):
    body: str = Field(min_length=1)
