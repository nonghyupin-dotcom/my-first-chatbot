import streamlit as st
import tempfile
import os
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA

# 1. 기본 설정
st.set_page_config(page_title="나만의 AI 용어 챗봇", page_icon="📘")
st.title("📘 AI 용어 100선 챗봇")

# --- [핵심] Secrets에서 키를 가져옵니다 ---
if "GOOGLE_API_KEY" in st.secrets:
    google_api_key = st.secrets["GOOGLE_API_KEY"]
else:
    st.error("설정(Settings) > Secrets 메뉴에 API 키를 저장해주세요!")
    st.stop()

# 2. 사이드바
with st.sidebar:
    st.header("📂 문서 업로드")
    st.write("학습할 PDF 파일을 선택해주세요.")
    uploaded_file = st.file_uploader("PDF 파일 선택", type="pdf")
    
    if google_api_key.startswith("AIza"):
        st.success("✅ 서버와 정상적으로 연결되었습니다!")

# 3. 메인 로직
if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file_path = tmp_file.name

    # 문서 학습
    with st.spinner("AI가 문서를 읽고 있습니다..."):
        try:
            loader = PyMuPDFLoader(tmp_file_path)
            pages = loader.load()
            embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=google_api_key)
            vectorstore = FAISS.from_documents(pages, embeddings)
            st.success("문서 학습 완료! 질문해보세요.")
        except Exception as e:
            st.error(f"오류 발생: {e}")
            st.stop()

    # 질문 답변
    llm = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=google_api_key)
    query = st.chat_input("궁금한 용어를 물어보세요!")
    
    if query:
        with st.chat_message("user"):
            st.write(query)
        
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vectorstore.as_retriever(search_kwargs={"k": 2}),
        )
        
        with st.spinner("답변 생성 중..."):
            result = qa_chain.invoke(query)
            with st.chat_message("assistant"):
                st.write(result['result'])

elif not uploaded_file:
    st.info("👈 왼쪽에서 PDF 파일을 업로드하면 챗봇이 시작됩니다.")
