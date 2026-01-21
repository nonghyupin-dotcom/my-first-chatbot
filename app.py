import streamlit as st
import tempfile
import os
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_community.embeddings import HuggingFaceEmbeddings

# 1. 기본 설정
st.set_page_config(page_title="나만의 AI 용어 챗봇", page_icon="📘")
st.title("📘 AI 용어 100선 챗봇")

# --- Secrets에서 키 가져오기 ---
if "GOOGLE_API_KEY" in st.secrets:
    google_api_key = st.secrets["GOOGLE_API_KEY"].strip()
else:
    st.error("Secrets에 API 키가 없습니다.")
    st.stop()

# 2. 사이드바
with st.sidebar:
    st.header("📂 문서 업로드")
    uploaded_file = st.file_uploader("PDF 파일 선택", type="pdf")
    
    if google_api_key.startswith("AIza"):
        st.caption("✅ Gemini LLM 연결 대기 중")

# 3. 메인 로직
if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file_path = tmp_file.name

    # (1) 문서 학습 (로컬 CPU 사용)
    with st.spinner("AI가 문서를 분석 중입니다... (서버에서 직접 처리)"):
        try:
            loader = PyMuPDFLoader(tmp_file_path)
            pages = loader.load()
            
            # 무료 로컬 임베딩
            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            
            vectorstore = FAISS.from_documents(pages, embeddings)
            st.success(f"✅ {len(pages)}페이지 문서 학습 완료! 질문해보세요.")
        except Exception as e:
            st.error(f"문서 처리 중 오류: {e}")
            st.stop()

    # (2) LLM 연결 (여기가 수정된 부분입니다!)
    # gemini-pro -> gemini-1.5-flash (최신 모델로 변경)
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=google_api_key)
    
    query = st.chat_input("궁금한 용어를 물어보세요!")
    
    if query:
        with st.chat_message("user"):
            st.write(query)
        
        # RAG 체인 가동
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vectorstore.as_retriever(search_kwargs={"k": 2}),
        )
        
        with st.spinner("답변 생성 중..."):
            try:
                result = qa_chain.invoke(query)
                with st.chat_message("assistant"):
                    st.write(result['result'])
            except Exception as e:
                st.error(f"오류 발생: {e}")

elif not uploaded_file:
    st.info("👈 왼쪽에서 PDF 파일을 업로드하면 챗봇이 시작됩니다.")
