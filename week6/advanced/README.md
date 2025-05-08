# 🎥 유튜브 자막 기반 RAG QA

유튜브 영상의 자막을 활용하여 질문에 대한 정확한 답변을 제공하는 RAG 기반의 QA 웹앱입니다.  
`LangChain`, `OpenAI`, `ChromaDB`, `Streamlit` 등을 활용하여 영상 내용을 요약하고, 관련 정보를 근거와 함께 제공합니다.   

---

## ✨ 주요 기능

- 🎞️ 유튜브 영상 링크 입력 시 자동 자막 추출
- 🧠 자막을 기반으로 벡터 임베딩 및 DB 저장 (Chroma + OpenAI Embedding)
- 🧾 영상 요약 생성
- ❓ 자유 질문에 대해 RAG을 활용한 응답 생성
- 🔍 답변의 **출처 문서 식별** 및 관련 유튜브 구간 링크 제공
- 📦 `meta.json`을 통한 메타데이터 및 요약 캐싱

---

## 🏗️ 프로젝트 구조

```
📁 chroma_db/
    └─ {video_id}/
         ├─ index...
         └─ meta.json

📄 prompts.yaml    # 프롬프트 템플릿 관리
📄 app.py          # Streamlit 실행 파일
```

---

## ⚙️ 사용 기술

| 구성 요소 | 설명 |
|-----------|------|
| Streamlit | UI 구성 및 실행 |
| yt-dlp    | 유튜브 메타데이터 및 자막 추출 |
| LangChain | 문서 임베딩 + QA 체인 구성 |
| OpenAI    | `gpt-4o`, `gpt-4o-mini` 사용 |
| Chroma    | 벡터 DB로 자막 chunk 저장 및 검색 |
| YAML      | 프롬프트 템플릿 관리 (`prompts.yaml`) |

---

## ✅ 출처 추적 방식

- 모델은 응답 마지막 줄에 `출처: [meta]` 형식으로 주요 근거를 명시
- 이를 파싱하여 UI에 관련 구간(시간 범위) 및 유튜브 링크 출력
