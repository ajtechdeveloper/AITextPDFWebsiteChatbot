import streamlit as st
from langchain_community.document_loaders import PyPDFLoader, UnstructuredURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_community.llms import HuggingFaceHub
from langchain.prompts import PromptTemplate
from langchain.schema import Document
import tempfile
import os

# Set up Hugging Face credentials
# To be used when running locally
# Also .env file to be created at root folder level
# with token: HUGGINGFACEHUB_API_TOKEN = <Your HuggingFace Hub API Token>
# load_dotenv(find_dotenv())
# To be used when deploying to Streamlit Cloud
hf_token = st.secrets["HUGGINGFACE_TOKEN"]["token"]
os.environ["HUGGINGFACEHUB_API_TOKEN"] = hf_token

# Initialize session state
if 'conversation' not in st.session_state:
    st.session_state.conversation = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'input_type' not in st.session_state:
    st.session_state.input_type = None
if 'user_question' not in st.session_state:
    st.session_state.user_question = ""

st.set_page_config(page_title="AI PDF, Website, or Text Chatbot", page_icon=":)")
st.header("AI PDF, Website, or Text Chatbot")
st.markdown(
    "This App uses AI to train a chatbot to reply to questions from any of the input: text, PDF or website link.")

st.markdown("Please Note: When a new input type is selected, the previous chat history will be cleared.")
# Input type selection
input_type = st.radio("Choose input type:", ("PDF", "Website URL", "Text"))

# Clear chat when input type changes
if input_type != st.session_state.input_type:
    st.session_state.conversation = None
    st.session_state.chat_history = []
    st.session_state.input_type = input_type

if input_type == "PDF":
    uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")
elif input_type == "Website URL":
    url_input = st.text_input("Enter website URL:")
else:  # Text input
    text_input = st.text_area("Enter your text here:")

if st.button("Process Input"):
    # Clear previous conversation and chat history
    st.session_state.conversation = None
    st.session_state.chat_history = []

    with st.spinner("Processing input..."):
        if input_type == "PDF" and uploaded_file is not None:
            with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_file_path = tmp_file.name

            loader = PyPDFLoader(tmp_file_path)
            documents = loader.load()
            os.unlink(tmp_file_path)
        elif input_type == "Website URL" and url_input:
            loader = UnstructuredURLLoader([url_input])
            documents = loader.load()
        elif input_type == "Text" and text_input:
            documents = [Document(page_content=text_input)]
        else:
            st.error("Please provide input based on your selected input type.")
            st.stop()

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        texts = text_splitter.split_documents(documents)

        embeddings = HuggingFaceEmbeddings()
        vectorstore = FAISS.from_documents(texts, embeddings)

        llm = HuggingFaceHub(repo_id="google/flan-t5-base", model_kwargs={"temperature": 0.5, "max_length": 512})

        prompt_template = """
        If the question is asked outside the input context, just say "Sorry, I do not know the answer"
        and
        use the following pieces of context to answer the question at the end. If you don't know the answer, just say "Sorry, I do not know the answer" - don't try to make up an answer.

        {context}

        Question: {question}
        Answer:"""
        PROMPT = PromptTemplate(
            template=prompt_template, input_variables=["context", "question"]
        )

        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

        st.session_state.conversation = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vectorstore.as_retriever(),
            memory=memory,
            combine_docs_chain_kwargs={"prompt": PROMPT}
        )

    st.success("Input processed successfully!")

# Chat interface
if st.session_state.conversation:
    user_question = st.text_input("Ask a question about the input provided:", key="user_question", value=st.session_state.user_question)
    if st.button("Ask"):
        if user_question:
            with st.spinner("Generating response..."):
                response = st.session_state.conversation({"question": user_question})
                st.session_state.chat_history.append(("Answer", response['answer']))
                st.session_state.chat_history.append(("Question", user_question))
            # Clear the question input
            st.session_state.user_question = ""
            st.rerun()

# Display chat history
if st.session_state.chat_history:
    st.subheader("Chat History")
    for role, message in reversed(st.session_state.chat_history):
        st.write(f"{role}: {message}")
