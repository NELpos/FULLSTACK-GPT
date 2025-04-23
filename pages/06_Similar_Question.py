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


cache_prompt = ChatPromptTemplate.from_template(
    """
    
    Use ONLY the following context to find out if there are similar questions and tell me if there are similar questions or not.
    If you can't just say you don't know, don't make anything up.

    If you found a similar question in the context, return it as true, if not, return it as false.

    Context: {context}

    Examples:

    Input Question : How far is the earth and the moon?
    Similar Question: How far away is the moon?
    similiar : true

    Input Question : How far is the earth and the mars?                                          
    Similar Question: How far away is the sun?
    Answer: false

    Question: {question}
    """
)


def cache_questions(inputs):
    import pydash as py_

    question = inputs["question"]
    condensed = "\n\n".join(item for item in st.session_state["question_list"])

    print(condensed)

    cache_chain = cache_prompt | llm
    result = cache_chain.invoke({"question": question, "context": condensed})

    return result


answer_prompt = ChatPromptTemplate.from_template(
    """
    당신은 사용자의 질의를 답변해주는 AI 비서입니다.
    {question}
    """
)


def save_qna(question, answer):
    st.session_state["question_list"].append(f"Question: {question}\nAnswer: {answer}")


def get_answer(inputs):
    question = inputs["question"]
    answers_chain = answer_prompt | llm
    save_qna(question, answers_chain.invoke({"question": question}).content)
    return {"question": question}


def save_message(message, role):
    # Save Question List
    st.session_state["chat_history"].append({"message": message, "role": role})


def send_message(message, role, save=True):
    with st.chat_message(role):
        st.markdown(message)

    if save:
        save_message(message, role)


st.set_page_config(
    page_title="Similar Question GPT",
    page_icon="👍",
)

st.markdown(
    """
    # Similar Question GPT
            
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


paint_history()

message = st.chat_input("Ask a question about the website...")
if message:
    send_message(message, "human")

    with st.chat_message("ai"):
        result = None
        isNew = True

        question_chain = {
            "question": RunnablePassthrough(),
        } | RunnableLambda(cache_questions)

        result = question_chain.invoke(message)

        st.write("재검색중입니다.")

        chain = {
            "question": RunnablePassthrough(),
        } | RunnableLambda(get_answer)

        result = chain.invoke(message)

        # if isNew:
        #     chain = (
        #         {
        #             "docs": retriever,
        #             "question": RunnablePassthrough(),
        #         }
        #         | RunnableLambda(get_answers)
        #         | RunnableLambda(choose_answer)
        #     )
        #     result = chain.invoke(message)
        #     save_qna(message, result.content)

        #     st.write(result.content)
        # else:
        #     st.write(result.get("answer", ""))
