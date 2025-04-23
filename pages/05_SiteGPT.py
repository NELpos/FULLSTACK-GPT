from langchain.document_loaders import SitemapLoader
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain.callbacks import StreamingStdOutCallbackHandler
from langchain.callbacks.base import BaseCallbackHandler
from langchain.chains import LLMChain
from langchain.memory import ConversationSummaryBufferMemory
from operator import itemgetter
import json

from langchain.schema import BaseOutputParser


class JsonOutputParser(BaseOutputParser):
    def parse(self, text):
        text = text.replace("```", "").replace("json", "")
        return json.loads(text)


output_parser = JsonOutputParser()


class ChatCallbackHandler(BaseCallbackHandler):
    message = ""

    def on_llm_start(self, *args, **kwargs):
        self.message_box = st.empty()

    def on_llm_end(self, *args, **kwargs):
        save_message(self.message, "ai")

    def on_llm_new_token(self, token, *args, **kwargs):
        self.message += token
        self.message_box.markdown(self.message)


llm = ChatOpenAI(
    temperature=0.1,
    streaming=True,
    callbacks=[
        ChatCallbackHandler(),
    ],
)

function = {
    "name": "find_similar_question",
    "description": "A function that finds if there was a similar question in the history and returns an answer if found.",
    "parameters": {
        "type": "object",
        "properties": {
            "question": {
                "type": "string",
            },
            "history": {
                "type": "string",
            },
        },
    },
    "required": ["question", "history"],
}

cache_llm = ChatOpenAI(
    temperature=0.1,
    model="gpt-3.5-turbo-1106",
    streaming=True,
    callbacks=[StreamingStdOutCallbackHandler()],
).bind(
    functions=[
        function,
    ],
    function_call={"name": "find_similar_question"},
)

memory = ConversationSummaryBufferMemory(
    llm=llm, max_token_limit=180, memory_key="chat_history", return_messages=True
)

splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=1000,
    chunk_overlap=200,
)

cache_prompt = ChatPromptTemplate.from_template(
    """
    You are an AI secretary. You recorded your previous question in {history}.
If there is a question similar to the question asked by the user in the history, please output the question and answer as follows.

If you find a similar question, please answer in the form below.

    Example Output:

    ```json
    {{
        "question": "What is the capital of France?",
        "answer": "Paris is the capital of France.",
        "isNew": false
    }}
    ```

If you don't have a similar question, please answer in the form below.

    Example Output:

    ```json
    {{
        "question": "What is the capital of France?",
        "answer": "No similar question found.",
        "isNew": true
    }}
    ```
"""
)


def cache_questions(inputs):
    import pydash as py_

    question = inputs["question"]
    # runnable = RunnablePassthrough.assign(
    #    chat_history=RunnableLambda(memory.load_memory_variables)
    #    | itemgetter("chat_history")
    # )

    # cache_chain = runnable | cache_prompt | cache_llm

    # result = cache_chain.invoke({"question": question, "question_list"}).additional_kwargs[
    #    "function_call"
    # ]["arguments"]

    print(st.session_state["question_list"])

    condensed = "\n\n".join(item for item in st.session_state["question_list"])

    cache_chain = cache_prompt | llm | output_parser
    result = cache_chain.invoke({"question": question, "history": condensed})

    return result


answers_prompt = ChatPromptTemplate.from_template(
    """
    Using ONLY the following context answer the user's question. If you can't just say you don't know, don't make anything up.
                                                  
    Context: {context}
                                
    Your turn!

    Question: {question}
"""
)


def get_answers(inputs):
    docs = inputs["docs"]
    question = inputs["question"]
    answers_chain = answers_prompt | llm
    # answers = []
    # for doc in docs:
    #     result = answers_chain.invoke(
    #         {"question": question, "context": doc.page_content}
    #     )
    #     answers.append(result.content)
    return {
        "question": question,
        "answers": [
            {
                "answer": answers_chain.invoke(
                    {"question": question, "context": doc.page_content}
                ).content,
                "source": doc.metadata["source"],
                "date": doc.metadata["lastmod"],
            }
            for doc in docs
        ],
    }


choose_promt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            Use ONLY the following pre-existing answers to answer the user's question.

            Use the answers that have the highest score (more helpful) and favor the most recent ones.

            Cite sources and return the sources of the answers as they are, do not change them.

            Answers: {answers}
            """,
        ),
        ("human", "{question}"),
    ]
)


def choose_answer(inputs):
    answers = inputs["answers"]
    question = inputs["question"]
    choose_chain = choose_promt | llm
    condensed = "\n\n".join(
        f"{answer['answer']}\nSource: {answer['source']}\nDate:{answer['date']}\n"
        for answer in answers
    )
    return choose_chain.invoke({"question": question, "answers": condensed})


def save_qna(question, answer):
    st.session_state["question_list"].append(f"Question: {question}\nAnswer: {answer}")


def save_message(message, role):
    # Save Question List
    st.session_state["chat_history"].append({"message": message, "role": role})


def send_message(message, role, save=True):
    with st.chat_message(role):
        st.markdown(message)

    if save:
        save_message(message, role)


def save_memory(message, response):
    memory.save_context({"inputs": message}, {"output": response})


def parse_page(soup):
    header = soup.find("header")
    footer = soup.find("footer")
    if header:
        header.decompose()
    if footer:
        footer.decompose()
    return (
        str(soup.get_text())
        .replace("\n", " ")
        .replace("\t", " ")
        .replace("\xa0", " ")
        .replace("  ", " ")
    )


@st.cache_data(show_spinner="Loading website...")
def load_website(url):
    loader = SitemapLoader(
        url,
        filter_urls=[
            r"^(.*\/products\/.*)$",
        ],
        parsing_function=parse_page,
    )
    loader.requests_per_second = 1
    docs = loader.load_and_split(text_splitter=splitter)
    vector_store = FAISS.from_documents(docs, OpenAIEmbeddings())
    return vector_store.as_retriever()


st.set_page_config(
    page_title="SiteGPT",
    page_icon="👍",
)

st.markdown(
    """
    # SiteGPT
            
    Ask questions about the content of a website.
            
    Start by writing the URL of the website on the sidebar.
"""
)

if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

if "question_list" not in st.session_state:
    st.session_state["question_list"] = []


def paint_history():
    for message in st.session_state["chat_history"]:
        send_message(
            message["message"],
            message["role"],
            save=False,
        )


with st.sidebar:
    url = st.text_input("Wirte down a URL", placeholder="https://exmaple.com")

if url:
    if ".xml" not in url:
        with st.sidebar:
            st.error("Please write down a Sitemap URL")
    else:
        paint_history()
        retriever = load_website(url)
        send_message("I'm ready! Ask away!", "ai", save=False)

        message = st.chat_input("Ask a question about the website...")
        if message:
            send_message(message, "human")

            with st.chat_message("ai"):
                result = None
                isNew = True

                if len(st.session_state["question_list"]) > 0:
                    cache_chain = {
                        "question": RunnablePassthrough(),
                    } | RunnableLambda(cache_questions)

                    result = cache_chain.invoke(message)
                    isNew = result.get("isNew", True)

                if isNew:
                    chain = (
                        {
                            "docs": retriever,
                            "question": RunnablePassthrough(),
                        }
                        | RunnableLambda(get_answers)
                        | RunnableLambda(choose_answer)
                    )
                    result = chain.invoke(message)
                    save_qna(message, result.content)

                    st.write(result.content)
                else:
                    st.write(result.get("answer", ""))
