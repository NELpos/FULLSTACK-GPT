import streamlit as st
import subprocess
import math
from pydub import AudioSegment
import glob
import openai
import os
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import StrOutputParser
from langchain.vectorstores.faiss import FAISS
from langchain.embeddings import CacheBackedEmbeddings, OpenAIEmbeddings
from langchain.storage import LocalFileStore
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough

llm = ChatOpenAI(
    temperature=0.1,
    streaming=True,
)

has_transcript = os.path.exists("./.cache/podcast.txt")


@st.cache_data()
def embed_file(file_path):
    cache_dir = LocalFileStore(f"./.cache/embeddings/meeting")
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=800,
        chunk_overlap=100,
    )
    loader = TextLoader(file_path)
    docs = loader.load_and_split(text_splitter=splitter)
    embeddings = OpenAIEmbeddings()
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(embeddings, cache_dir)
    vectorstore = FAISS.from_documents(docs, cached_embeddings)
    retriever = vectorstore.as_retriever()
    return retriever


@st.cache_data()
def transcribe_chunks(chunk_folder, destination):
    if has_transcript:
        return
    files = glob.glob(f"{chunk_folder}/*.mp3")
    files.sort()
    for file in files:
        with open(file, "rb") as audio_file, open(destination, "a") as f:
            transcript = openai.Audio.transcribe("whisper-1", audio_file)
            f.write(transcript["text"])


@st.cache_data()
def extract_audio_from_video(video_path):
    if has_transcript:
        return
    audio_path = video_path.replace(".mp4", ".mp3")
    command = [
        "ffmpeg",
        "-y",
        "-i",
        video_path,
        "-vn",
        audio_path,
    ]
    subprocess.run(command)


@st.cache_data()
def cut_audio_in_chunks(audio_path, chunk_size, chunks_folder):
    if has_transcript:
        return
    track = AudioSegment.from_mp3(audio_path)
    chunk_len = chunk_size * 60 * 1000
    chunks = math.ceil(len(track) / chunk_len)
    for i in range(chunks):
        start = i * chunk_len
        end_time = (i + 1) * chunk_len
        track[start:end_time].export(f"{chunks_folder}/chunk_{i}.mp3", format="mp3")


refine_prompt = ChatPromptTemplate.from_template(
    """
    Your job is to produce a final summary.
    We have provided an existing summary up to a certain point: {existing_summary}
    We have the opportunity to refine the existing summary (only if needed) with some more context below.
    ------------
    {context}
    ------------
    Given the new context, refine the original summary.
    If the context isn't useful, RETURN the original summary.
    """
)

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
    answers_chain = answers_prompt | llm | StrOutputParser()

    return {
        "answers": [
            answers_chain.invoke({"question": question, "context": doc.page_content})
            for doc in docs
        ]
    }


def summary_answer(inputs):
    answers = inputs["answers"]
    refine_chain = refine_prompt | llm | StrOutputParser()
    summary = answers[0]
    for answer in answers[1:]:
        summary = refine_chain.invoke({"existing_summary": summary, "context": answer})

    return summary


st.set_page_config(page_title="Meeting GPT", page_icon="👍")

st.markdown(
    """
# MeetingGPT
            
Welcome to MeetingGPT, upload a video and I will give you a transcript, a summary and a chat bot to ask any questions about it.

Get started by uploading a video file in the sidebar.
"""
)

with st.sidebar:
    video = st.file_uploader("Video", type=["mp4", "avi", ",mkv", "mov"])

if video:
    chunks_folder = "./.cache/chunks"
    with st.status("Loading video...") as status:
        video_content = video.read()
        video_path = f"./.cache/{video.name}"
        audio_path = video_path.replace(".mp4", ".mp3")
        transcript_path = video_path.replace(".mp4", ".txt")
        with open(video_path, "wb") as f:
            f.write(video_content)

        status.update(label="Extracting audio...")
        extract_audio_from_video(video_path)
        status.update(label="Cutting audio segments...")
        cut_audio_in_chunks(audio_path, 10, "./.cache/chunks")
        status.update(label="Transcribing Audio...")
        transcribe_chunks(chunks_folder, transcript_path)

    trascript_tab, summary_tab, qa_tab = st.tabs(["Transcript", "Summary", "Q&A"])

    with trascript_tab:
        with open(transcript_path, "r") as f:
            st.write(f.read())

    with summary_tab:
        start = st.button("Generate summary")

        if start:
            loader = TextLoader("./.cache/podcast.txt")

            splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
                chunk_size=800, chunk_overlap=100
            )

            docs = loader.load_and_split(text_splitter=splitter)

            first_summary_prompt = ChatPromptTemplate.from_template(
                """
                                                                    Write a concise summary of the following:
                                                                    "{text}"
                                                                    CONCISE SUMMARY
            """
            )

            first_summary_chain = first_summary_prompt | llm | StrOutputParser()

            summary = first_summary_chain.invoke({"text": docs[0].page_content})

            refine_chain = refine_prompt | llm | StrOutputParser()

            with st.status("Summarizing....") as status:
                for i, doc in enumerate(docs[1:]):
                    status.update(label=f"Processing document {i+1}/{len(docs)-1}")
                    summary = refine_chain.invoke(
                        {"existing_summary": summary, "context": doc.page_content}
                    )
                st.write(summary)
            st.write(summary)

    with qa_tab:
        retriever = embed_file(transcript_path)
        # docs = retriever.invoke("do they talk about marcus aurelius?")

    message = st.chat_input("Ask a question about the video...")

    if message:
        with st.chat_message("human"):
            st.markdown(message)

        with st.chat_message("ai"):
            chain = (
                {
                    "docs": retriever,
                    "question": RunnablePassthrough(),
                }
                | RunnableLambda(get_answers)
                | RunnableLambda(summary_answer)
            )
            result = chain.invoke(message)

            st.write(result)
