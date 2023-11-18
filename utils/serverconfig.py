from typing import Optional

from pydantic import BaseModel


class ServerConfig(BaseModel):
    pass


# TODO: Copy lm_eval?
class RunConfig(ServerConfig):
    num_fewshot: Optional[int] = 0
    limit: Optional[int] = None

    class Config:
        frozen = True