from typing import Any, Dict
from fastapi import Body, Form, Request, FastAPI
from pydantic import BaseModel, Field
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request
from langchain_pinecone import PineconeVectorStore
from langchain.embeddings import OpenAIEmbeddings
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import pinecone
import os

load_dotenv()


app = FastAPI(
    title="ChefGPT. The best provider of indian Recipes in the world",
    description="Give ChefGPT the name of an ingredient and it will give you multiple recipes to use that ingredient on in return.",
    servers=[{"url": "https://occurrence-signup-keeps-creator.trycloudflare.com"}],
)

embeddings = OpenAIEmbeddings()
vector_store = PineconeVectorStore.from_existing_index("recipes", embeddings)


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


class Document(BaseModel):
    page_content: str


@app.get(
    "/recipes",
    summary="Returns a list of recipes.",
    description="Upon receiving an ingredient, this endpoint will return a list of recipes that contain that ingredient.",
    response_description="A Document object that contains the recipe and preparation instructions",
    response_model=list[Document],
    openapi_extra={
        "x-openai-isConsequential": False,
    },
)
def get_recipe(ingredient: str):
    docs = vector_store.similarity_search(ingredient)
    return docs


user_token_db = {"ABCDEF": "nico"}


@app.get("/authorize", response_class=HTMLResponse, include_in_schema=False)
def handle_authorize(client_id: str, redirect_uri: str, state: str):
    return f"""
    <html>
        <head>
            <title>Nicolacus Maximus Log In</title>
        </head>
        <body>
            <h1>Log Into Nicolacus Maximus</h1>
            <a href="{redirect_uri}?code=ABCDEF&state={state}">Authorize Nicolacus Maximus GPT</a>
        </body>
    </html>
    """


@app.post(
    "/token",
    include_in_schema=False,
)
def handle_token(code=Form(...)):
    print(code)
    return {"access_token": user_token_db[code]}
