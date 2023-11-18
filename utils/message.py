from enum import Enum
from typing import Dict, Optional
from pydantic import BaseModel


class Task(Enum):
    HAERAE = "haerae"
    CUSTOM = "custom"
    # TODO : More tasks Korean and English


class Action(Enum):
    pass


# Server to Client
class ServerAction(Action):
    ERROR = "ERROR"

    START = "START"
    END = "END"

    # Test data
    INPUT = "INPUT"
    ACK = "ACK"

    RUN_ACK = "RUN_ACK"

    NOT_INIT = "NOT_INIT"
    EMPTY = "EMPTY"

    # Report ack
    ANALYZE_ACK = "ANALYZE_ACK"
    SAMPLE_ACK = "SAMPLE_ACK"


class TaskPayload(BaseModel):
    task: Task
    eval_key: str
    upload: bool
    model_tag: str


# Client to Server
class ClientAction(Action):
    ERROR = "ERROR"

    RUN = "RUN"
    ANALYZE = "ANALYZE"
    SAMPLES = "SAMPLES"

    UPLOAD_ONLY = "UPLOAD_ONLY"

    # Model output
    OUTPUT = "OUTPUT"


class Identifier(BaseModel):
    task: Task
    uid: str

    class Config:
        frozen = True


class ServerMessage(BaseModel):
    action: ServerAction
    id: Optional[Identifier] = None
    data: Optional[Dict] = None


class ClientMessage(BaseModel):
    action: ClientAction
    id: Optional[Identifier] = None
    data: Optional[Dict] = None
