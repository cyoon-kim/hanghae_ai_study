# app.py

import streamlit as st
from yt_dlp import YoutubeDL
import json, os, requests, logging
from dotenv import load_dotenv
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.schema import Document
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
import yaml
import re

# 로그 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ★ 환경 설정
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

st.set_page_config(page_title="🎥 YouTube RAG QA", layout="wide")
st.title("🎥 YouAsk - 유튜브 자막 기반 RAG QA")


def fetch_video_info(video_url):
    ydl_opts = {
        "writesubtitles": True,
        "writeautomaticsub": True,
        "subtitleslangs": ["ko"],
        "skip_download": True
    }
    with YoutubeDL(ydl_opts) as ydl:
        return ydl.extract_info(video_url, download=False)


def parse_subtitles(subtitle_url):
    captions = []
    try:
        res = requests.get(subtitle_url)
        content_type = res.headers.get("Content-Type", "")
        logger.info(f"자막 Content-Type: {content_type}")
        logger.info(f"자막 URL: {subtitle_url}")

        if "application/json" in content_type:
            sub_json = res.json()
            for event in sub_json.get("events", []):
                if "segs" in event:
                    text = "".join([seg.get("utf8", "") for seg in event["segs"]]).strip()
                    if text and text != "\n":
                        captions.append({
                            "start": event["tStartMs"] / 1000,
                            "end": (event["tStartMs"] + event.get("dDurationMs", 0)) / 1000,
                            "text": text
                        })

        elif "text/vtt" in content_type or res.text.strip().startswith("WEBVTT"):
            vtt_text = res.text
            blocks = vtt_text.strip().split("\n\n")
            for block in blocks:
                lines = block.strip().split("\n")
                if len(lines) >= 2 and "-->" in lines[0]:
                    timecode = lines[0]
                    text = " ".join(lines[1:]).strip()
                    if text:
                        start, end = [t.strip() for t in timecode.split("-->")]
                        captions.append({
                            "start": start,
                            "end": end,
                            "text": text
                        })

        else:
            logger.warning("지원하지 않는 자막 형식입니다.")

    except Exception as e:
        logger.exception("자막 파싱 실패")

    return captions


def chunk_captions(captions, chunk_size=20, overlap=5):
    chunks = []
    i = 0
    while i < len(captions):
        group = captions[i:i + chunk_size]
        if not group:
            break
        text = " ".join([c["text"] for c in group])
        chunks.append({
            "text": text,
            "start": group[0]["start"],
            "end": group[-1]["end"]
        })
        i += chunk_size - overlap
    return chunks


def build_vector_db(docs, persist_dir):
    embedding = OpenAIEmbeddings()
    vectordb = Chroma.from_documents(docs, embedding, persist_directory=persist_dir)
    vectordb.persist()
    return vectordb

def load_prompt(version="v1", prompt_name="qa_prompt", path="prompts.yaml"):
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if prompt_name not in data:
        raise KeyError(f"'{prompt_name}' 항목이 '{path}'에 없습니다.")

    versioned_prompt = data[prompt_name].get(version)
    if versioned_prompt is None:
        raise ValueError(f"프롬프트 버전 '{version}'을 찾을 수 없습니다.")

    return versioned_prompt

def generate_summary(chunks):
    logger.info("요약 생성 시작")
    llm = ChatOpenAI(temperature=0.3, model_name="gpt-4o")
    full_text = "\n".join([chunk["text"] for chunk in chunks])

    prompt_template = load_prompt(version="v1", prompt_name="summary_prompt")
    prompt = prompt_template.format(full_text=full_text)

    response = llm.invoke(prompt)
    logger.info("요약 생성 완료")
    return response



# ★ 1. 유튜브 링크 입력
video_url = st.text_input("유튜브 영상 URL을 입력해주세요:")

if video_url:
    video_id = video_url.split("v=")[-1][:11]
    persist_dir = f"./chroma_db/{video_id}"

    if "vectordb" not in st.session_state:
        if os.path.exists(persist_dir):
            embedding = OpenAIEmbeddings()
            vectordb = Chroma(persist_directory=persist_dir, embedding_function=embedding)
            st.session_state.vectordb = vectordb

            with open(f"{persist_dir}/meta.json", "r", encoding="utf-8") as f:
                meta = json.load(f)
            st.session_state.meta = meta
            st.session_state.title = meta["title"]
            st.session_state.thumbnail = meta["thumbnail_url"]
            st.session_state.description = meta["description"]
            st.session_state.summary = meta.get("summary")
            st.session_state.video_id = meta["video_id"]

        else:
            with st.spinner("자막 및 메타데이터 처리 중..."):
                info = fetch_video_info(video_url)

                title = info["title"]
                thumbnail = info["thumbnail"]
                description = info.get("description", "")
                subtitle_formats = info.get("automatic_captions", {}).get("ko", [])

                if not subtitle_formats:
                    st.error("❌ 자막이 제공되지 않습니다.")
                    st.stop()

                subtitle_url = subtitle_formats[0]["url"]
                captions = parse_subtitles(subtitle_url)

                if not captions:
                    st.error("❌ 자막을 불러오지 못했습니다.")
                    st.stop()

                chunks = chunk_captions(captions)
                docs = [
                    Document(page_content=chunk["text"], metadata={"start": chunk["start"], "end": chunk["end"]})
                    for chunk in chunks
                ]

                vectordb = build_vector_db(docs, persist_dir)
                summary = generate_summary(chunks)
                if hasattr(summary, "content"):
                    summary = summary.content

                meta = {
                    "video_id": info["id"],
                    "title": title,
                    "description": description,
                    "uploader": info.get("uploader"),
                    "upload_date": info.get("upload_date"),
                    "duration": info["duration"],
                    "view_count": info["view_count"],
                    "thumbnail_url": thumbnail,
                    "summary": summary
                }

                with open(f"{persist_dir}/meta.json", "w", encoding="utf-8") as f:
                    json.dump(meta, f, ensure_ascii=False, indent=2)

                st.session_state.meta = meta
                st.session_state.vectordb = vectordb
                st.session_state.title = title
                st.session_state.thumbnail = thumbnail
                st.session_state.description = description
                st.session_state.summary = summary
                st.session_state.video_id = info["id"]

# ★ 2. 영상 정보 + 질문 UI
if "vectordb" in st.session_state:
    col1, col2 = st.columns([1, 3])
    with col1:
        st.image(st.session_state.thumbnail, width=180)
    with col2:
        st.subheader(st.session_state.title)
        st.write(st.session_state.description)
        if st.session_state.summary:
            st.markdown("**📄 요약:**")
            st.write(st.session_state.summary)

    query = st.text_input("질문을 입력해주세요:")

    if st.button("질문하기") and query:
        with st.spinner("LLM에서 답변 생성 중..."):
            llm = ChatOpenAI(temperature=0.2, model_name="gpt-4o-mini")
            retriever = st.session_state.vectordb.as_retriever(search_kwargs={"k": 3})
            top_docs = retriever.get_relevant_documents(query)

            meta_text = "\n".join([f"{k}: {v}" for k, v in st.session_state.meta.items() if k not in ["summary"]])
            meta_doc = Document(page_content=f"[meta]\n{meta_text}", metadata={"type": "meta"})
            summary_doc = Document(page_content=f"[summary]\n{st.session_state.summary}", metadata={"type": "summary"})
            numbered_retrievals = [Document(page_content=f"[retriever{i+1}]\n{doc.page_content}", metadata=doc.metadata) for i, doc in enumerate(top_docs)]

            all_contexts = [meta_doc, summary_doc] + numbered_retrievals
            full_context = "\n\n".join([doc.page_content for doc in all_contexts])
            
            prompt_template = load_prompt(version="v1", prompt_name="qa_prompt")
            prompt = prompt_template.format(full_context=full_context, query=query)

            logger.info(f"prompt: {prompt}")
            answer = llm.invoke(prompt)
            answer_text = answer.content.strip()

            logger.info(f"llm_answer: {answer_text}")
            match = re.search(r"출처:\s*(\[(.*?)\])", answer_text)
            source_tag = match.group(1) if match else None

            if source_tag:
                answer_text = re.sub(rf"\n?출처:\s*{re.escape(source_tag)}\s*", "", answer_text).strip()

            st.markdown("### 💡 답변")
            st.write(answer_text)

            st.markdown("### 🕒 참조된 구간")
            if source_tag and source_tag.startswith("[retriever"):
                idx = int(source_tag[-2]) - 1  # [retriever2] → 1
                doc = top_docs[idx]
                start = float(doc.metadata["start"])
                end = float(doc.metadata["end"])
                link = f"https://www.youtube.com/watch?v={st.session_state.video_id}&t={int(start)}s"
                st.markdown(f"⏱️ 출처: {source_tag} → [{start:.2f} ~ {end:.2f}초]({link})", unsafe_allow_html=True)
            elif source_tag:
                st.markdown(f"ℹ️ 출처: {source_tag} 기반입니다.")
            else:
                st.markdown("⚠️ 출처 정보를 찾을 수 없습니다.")