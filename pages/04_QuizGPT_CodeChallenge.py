import streamlit as st
import json
from langchain.retrievers import WikipediaRetriever
from langchain.document_loaders import UnstructuredFileLoader
from langchain.storage import LocalFileStore
from langchain.text_splitter import CharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.callbacks import StreamingStdOutCallbackHandler
from langchain.schema import BaseOutputParser

st.set_page_config(
    page_title="QuizGPT_Code_Challenge",
    page_icon="👍",
)

st.title("QuizGPT_Code_Challenge")

function = {
    "name": "create_quiz",
    "description": "function that takes a list of questions and answers and returns a quiz",
    "parameters": {
        "type": "object",
        "properties": {
            "questions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "question": {
                            "type": "string",
                        },
                        "answers": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "answer": {
                                        "type": "string",
                                    },
                                    "correct": {
                                        "type": "boolean",
                                    },
                                },
                                "required": ["answer", "correct"],
                            },
                        },
                    },
                    "required": ["question", "answers"],
                },
            }
        },
        "required": ["questions"],
    },
}


llm = ChatOpenAI(
    temperature=0.1,
    model="gpt-3.5-turbo-1106",
    streaming=True,
    callbacks=[StreamingStdOutCallbackHandler()],
).bind(
    function_call={
        "name": "create_quiz",
    },
    functions=[
        function,
    ],
)


def format_docs(docs):
    return "\n\n".join(document.page_content for document in docs)


prompt = PromptTemplate.from_template("Make a quiz about {keyword}")


@st.cache_data(show_spinner="Loading file...")
def split_file(file):
    file_content = file.read()
    file_path = f"./.cache/quiz_files/{file.name}"
    with open(file_path, "wb") as f:
        f.write(file_content)

    cache_dir = LocalFileStore(f"./.cache/embeddings/{file.name}")

    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator="\n",
        chunk_size=600,
        chunk_overlap=100,
    )
    loader = UnstructuredFileLoader(file_path)
    docs = loader.load_and_split(text_splitter=splitter)
    return docs


@st.cache_data(show_spinner="Making quiz...")
def run_quiz_chain(_docs, topic):
    chain = prompt | llm
    return chain.invoke({"keyword": topic})


@st.cache_data(show_spinner="Search Wikipidia...")
def wiki_search(term):
    retriever = WikipediaRetriever(top_k_results=5)
    docs = retriever.get_relevant_documents(term)
    return docs


with st.sidebar:
    docs = None
    choice = st.selectbox(
        "Choose what you want to use.",
        (
            "File",
            "Wikipedia Article",
        ),
    )
    if choice == "File":
        file = st.file_uploader(
            "Upload a .docx, .txt or .pdf file",
            type=["pdf", "txt", "docx"],
        )
        if file:
            docs = split_file(file)
    else:
        topic = st.text_input("Search Wikipidia...")
        if topic:
            docs = wiki_search(topic)
    on = st.toggle("Show the answer")

if not docs:
    st.markdown(
        """
Welcome to QuizGPT.

I sill make a quiz from Wikipedia articles or files you upload to test your knowledge and help you study.

Get started by uploading a file or searching on Wikipedia in the sidebar.
"""
    )
else:
    response = run_quiz_chain(docs, topic if topic else file.name)
    response = json.loads(response.additional_kwargs["function_call"]["arguments"])

    with st.form("question_form"):
        for question in response["questions"]:
            st.write(question["question"])
            value = st.radio(
                "Select an option.",
                [answer["answer"] for answer in question["answers"]],
                index=None,
                key=question["question"],
            )
            if {"answer": value, "correct": True} in question["answers"]:
                st.success("Correct")
            elif value is not None:
                if on:
                    st.error(
                        "Wrong.. The Answer is {0}".format(
                            [
                                answer["answer"]
                                for answer in question["answers"]
                                if answer["correct"]
                            ][0]
                        )
                    )
                else:
                    st.error("Wrong")

        button = st.form_submit_button("Submit")
