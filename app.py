import streamlit as st
import tempfile
import os
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA

# 1. 페이지 제목 설정
st.set_page_config(page_title="나만의 AI 용어 챗봇", page_icon="📘")
st.title("📘 AI 용어 100선 챗봇")
st.write("구글 Gemini를 활용한 무료 RAG 서비스입니다.")

# 2. 사이드바: API 키 입력 & 파일 업로드
with st.sidebar:
    st.header("설정")
    google_api_key = st.text_input("Google API Key를 입력하세요", type="password")
    
    st.markdown("---")
    st.write("학습할 PDF 파일을 업로드하세요.")
    uploaded_file = st.file_uploader("PDF 파일 선택", type="pdf")

# 3. 메인 로직
if uploaded_file is not None and google_api_key:
    # (1) 임시 파일로 저장 (Streamlit은 파일을 바로 읽을 수 없어서 저장해야 함)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file_path = tmp_file.name

    # (2) 데이터 로드 및 벡터 DB 생성 (한 번만 실행되도록 캐싱)
    @st.cache_resource
    def process_pdf(file_path):
        loader = PyMuPDFLoader(file_path)
        pages = loader.load()
        
        # 구글의 무료 임베딩 모델 사용
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=google_api_key)
        vectorstore = FAISS.from_documents(pages, embeddings)
        return vectorstore

    # 처리 중 표시
    with st.spinner("문서를 분석 중입니다..."):
        vectorstore = process_pdf(tmp_file_path)
        st.success("문서 학습 완료!")

    # (3) Gemini LLM 연결
    llm = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=google_api_key)
    
    # (4) 질문 및 답변
    query = st.chat_input("궁금한 용어를 물어보세요!")
    
    if query:
        # 사용자의 질문을 화면에 표시
        with st.chat_message("user"):
            st.write(query)

        # RAG 체인 실행
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vectorstore.as_retriever(search_kwargs={"k": 2}),
        )
        
        # 답변 생성
        with st.spinner("답변을 생성 중입니다..."):
            result = qa_chain.invoke(query)
            
        # AI의 답변을 화면에 표시
        with st.chat_message("assistant"):
            st.write(result['result'])

elif not google_api_key:
    st.warning("왼쪽 사이드바에 Google API Key를 입력해주세요!")
elif not uploaded_file:
    st.info("PDF 파일을 업로드하면 챗봇이 시작됩니다.")
