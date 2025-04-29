from dotenv import load_dotenv
import os
import base64
import streamlit as st

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

# 환경변수 로드
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# 모델 준비
model = ChatOpenAI(model="gpt-4o-mini", api_key=openai_api_key)

# Streamlit UI
st.title("Image QA Bot")

# 이미지 업로더
images = st.file_uploader(
    "질문하고 싶은 이미지를 올려주세요!",
    type=['png', 'jpg', 'jpeg'],
    accept_multiple_files=True
)

# 질문 입력창
user_question = st.text_input("사진에 대해 질문을 입력하세요")

submit = st.button("질문하기")

# 이미지 미리보기 (가로로 정렬)
if images:
    st.subheader("업로드된 이미지 미리보기")
    cols = st.columns(len(images))
    for idx, img in enumerate(images):
        with cols[idx]:
            st.image(img, width=200)

# 버튼 클릭 시 로직
if submit:
    if not images:
        st.error("❗ 이미지를 한 장 이상 업로드해주세요.")
    elif not user_question:
        st.error("❗ 질문을 입력해주세요.")
    else:
        img_payloads = []
        for image in images:
            encoded_img = base64.b64encode(image.read()).decode("utf-8")
            img_payloads.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{encoded_img}"},
            })

        with st.spinner("답변 생성 중..."):  # 로딩 표시
            with st.chat_message("assistant"):
                message = HumanMessage(
                    content=[
                        {"type": "text", "text": user_question},
                        *img_payloads
                    ],
                )
                result = model.invoke([message])
                response = result.content
                st.markdown(response)
