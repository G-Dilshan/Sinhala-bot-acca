from pydantic import BaseModel

class RequestBody(BaseModel):
    question: str