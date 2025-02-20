from typing import Optional

from pydantic import BaseModel


class HellowRequest(BaseModel):
    names: list[str]