from fastapi import FastAPI
from fastapi import Body, Form, Request, FastAPI
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

app = FastAPI(
    title="Nelpos Giver",
    description="Get a real quote said by Nelpos himself.",
    servers=[
        {
            "url": "https://occurrence-signup-keeps-creator.trycloudflare.com",
        },
    ],
)

templates = Jinja2Templates(directory="templates")


@app.get("/privacy", response_class=HTMLResponse)
async def get_privacy_policy(request: Request):
    return templates.TemplateResponse("privacy.html", {"request": request})


class Quote(BaseModel):
    quote: str = Field(
        description="The quote that Nelpos said.",
    )
    year: int = Field(
        description="The year when Nelpos said the quote.",
    )


@app.get(
    "/quote",
    summary="Returns a random quote by Nelpos",
    description="Upon receiving a GET request this endpoint will return a real quiote said by Nelpos herself.",
    response_description="A Quote object that contains the quote said by Nelpos and the date when the quote was said.",
    response_model=Quote,
)
def get_quote():
    return {
        "quote": "Life is short so eat it all.",
        "year": 1950,
    }
