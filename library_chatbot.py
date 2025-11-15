import os
import streamlit as st
import nest_asyncio

nest_asyncio.apply()

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain_community.chat_message_histories.streamlit import StreamlitChatMessageHistory

__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
from langchain_chroma import Chroma



# ============================
# 1. API Key 설정
# ============================
try:
    os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
except:
    st.error("⚠️ GOOGLE_API_KEY를 Streamlit Secrets에 설정해주세요!")
    st.stop()




# ============================
# 2. PDF 로드 함수
# ============================
@st.cache_resource
def load_and_split_pdf(file_path):
    loader = PyPDFLoader(file_path)
    return loader.load_and_split()



# ============================
# 3. 임베딩 + Vector DB 구축
# ============================
@st.cache_resource
def create_vector_store(_docs):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    split_docs = text_splitter.split_documents(_docs)

    st.info(f"📄 {len(split_docs)}개의 청크로 분할했습니다.")

    persist_directory = "./chroma_db_marine_biodegradable"

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

    vectorstore = Chroma.from_documents(
        split_docs,
        embeddings,
        persist_directory=persist_directory
    )

    st.success("🌊 해양 생분해 신소재 Vector DB 구축 완료!")
    return vectorstore


@st.cache_resource
def get_vectorstore(_docs):
    persist_directory = "./chroma_db_marine_biodegradable"

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

    if os.path.exists(persist_directory):
        return Chroma(
            persist_directory=persist_directory,
            embedding_function=embeddings
        )
    else:
        return create_vector_store(_docs)




# ============================
# 4. RAG 구성 요소 초기화
# ============================
@st.cache_resource
def initialize_components(selected_model):

    # 👉 이 PDF 경로만 교체하면 됨 (예: PHA, PLA, 해양 미생물 기반 생분해 연구 PDF)
    file_path = r"/mnt/data/Review_of_recent_advances_in_the_biodegradability_.pdf"


    pages = load_and_split_pdf(file_path)
    vectorstore = get_vectorstore(pages)
    retriever = vectorstore.as_retriever()

    # 🔵 질문 재구성 프롬프트
    contextualize_q_system_prompt = """
    Reformulate the user’s question into a standalone question 
    using the conversation history only for context. Do NOT answer.
    """

    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("history"),
        ("human", "{input}"),
    ])

    # 🔵 생분해 신소재 Q&A 프롬프트
    qa_system_prompt = """
    당신은 해양 플라스틱 분해 신소재(PHA, PLA, 미생물 기반 폴리머 등)에 대한 정보를 제공하는 AI 조교입니다.
    아래 제공된 연구자료와 문맥을 기반으로 답변하세요.
    정보를 모르면 모른다고 답하고, 추측하지 않습니다.
    답변은 한국어 + 존댓말 + 이모지 조합을 유지하세요.

    {context}
    """

    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", qa_system_prompt),
        MessagesPlaceholder("history"),
        ("human", "{input}"),
    ])

    llm = ChatGoogleGenerativeAI(
        model=selected_model,
        temperature=0.6,
        convert_system_message_to_human=True
    )

    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )

    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

    rag_chain = create_retrieval_chain(
        history_aware_retriever,
        question_answer_chain
    )

    return rag_chain




# ============================
# 5. Streamlit UI
# ============================
st.header("🌊 해양 플라스틱 분해 신소재 RAG 챗봇")

if not os.path.exists("./chroma_db_marine_biodegradable"):
    st.info("🔄 첫 실행: PDF 임베딩 생성 중입니다...")

option = st.selectbox(
    "Select Gemini Model",
    ("gemini-2.0-flash-exp", "gemini-2.5-flash", "gemini-2.0-flash-lite"),
    index=0
)

with st.spinner("🔧 연구자료 로딩 및 모델 초기화 중..."):
    rag_chain = initialize_components(option)

st.success("✅ 챗봇 준비 완료!")




# ============================
# 6. 대화 히스토리 및 RAG
# ============================
chat_history = StreamlitChatMessageHistory(key="chat_messages")

conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    lambda session_id: chat_history,
    input_messages_key="input",
    history_messages_key="history",
    output_messages_key="answer",
)




# ============================
# 7. 기존 히스토리 출력
# ============================
for msg in chat_history.messages:
    st.chat_message(msg.type).write(msg.content)




# ============================
# 8. 유저 질문 처리
# ============================
if prompt := st.chat_input("해양 플라스틱 분해 신소재에 대해 궁금한 점을 입력하세요! 🌱"):
    st.chat_message("human").write(prompt)

    with st.chat_message("ai"):
        with st.spinner("🔍 자료 검색 및 분석 중..."):
            config = {"configurable": {"session_id": "any"}}

            response = conversational_rag_chain.invoke(
                {"input": prompt},
                config
            )

            st.write(response["answer"])

            with st.expander("📄 참고 문서 보기"):
                for doc in response["context"]:
                    st.markdown(doc.metadata['source'], help=doc.page_content)
SS
