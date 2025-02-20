from typing import Annotated, List

# import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware

from src.config import logger, public_or_local
from src.models.schemes import HellowRequest
from src.utils.greeting import hellow_names

if public_or_local == 'LOCAL':
    url = 'http://localhost'
else:
    url = 'http://11.11.11.11'

origins = [
    url
]

app = FastAPI(docs_url="/template_fast_api/v1/", openapi_url='/template_fast_api/v1/openapi.json')
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



@app.post("/template_fast_api/v1/greetings")
async def inputation(body: Annotated[
    HellowRequest, Body(
        example={"names": ['Sasha', 'Nikita', 'Kristina']})]):
    try:
        names = body.names
        if names:
            res = hellow_names(names)
            return res
        else:
            logger.error("Something happened during creation of the search table")
            raise HTTPException(
                status_code=400,
                detail="Bad Request",
                headers={"X-Error": "Something happened during creation of the search table"},
            )
    except Exception as ApplicationError:
        logger.error(ApplicationError.__repr__())
        raise HTTPException(
            status_code=400,
            detail="Unknown Error",
            headers={"X-Error": f"{ApplicationError.__repr__()}"},
        )


@app.get("/")
def read_root():
    return {"message": "Welcome to the indicators System API"}


if __name__ == "__main__":
    port = 7070
    uvicorn.run(app, host="0.0.0.0", port=port)
