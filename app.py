import streamlit as st
from langchain_community.document_loaders import PyPDFLoader, UnstructuredURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
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
if 'question_key' not in st.session_state:
    st.session_state.question_key = 0

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
        try:
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

            embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

            vectorstore = FAISS.from_documents(texts, embeddings)

            llm = HuggingFaceHub(
                repo_id="google/flan-t5-large",  # Changed to a more stable model
                model_kwargs={
                    "temperature": 0.5,
                    "max_length": 512,
                    "task": "text2text-generation"
                }
            )

            prompt_template = """
                Use the following context to answer the question. If the question cannot be answered using only the provided context, respond with "I cannot answer this question based on the provided context."

                Context: {context}

                Question: {question}
                Answer:"""

            PROMPT = PromptTemplate(
                template=prompt_template,
                input_variables=["context", "question"]
            )

            memory = ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True,
                output_key="answer"
            )

            st.session_state.conversation = ConversationalRetrievalChain.from_llm(
                llm=llm,
                retriever=vectorstore.as_retriever(),
                memory=memory,
                combine_docs_chain_kwargs={"prompt": PROMPT},
                return_source_documents=True
            )

            st.success("Input processed successfully!")

        except Exception as e:
            st.error(f"An error occurred during processing: {str(e)}")
            st.stop()

# Chat interface
if st.session_state.conversation:
    user_question = st.text_input("Ask a question about the input provided:",
                                  key=f"user_question_{st.session_state.question_key}")
    if st.button("Ask"):
        if user_question:
            with st.spinner("Generating response..."):
                try:
                    response = st.session_state.conversation({"question": user_question})
                    st.session_state.chat_history.append(("Question", user_question))
                    st.session_state.chat_history.append(("Answer", response['answer']))
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")
                    st.stop()
            # Increment the question key to reset the input field
            st.session_state.question_key += 1
            st.rerun()

# Display chat history
if st.session_state.chat_history:
    st.subheader("Chat History")
    for role, message in reversed(st.session_state.chat_history):
        if role == "Question":
            st.markdown(f"**{role}:** {message}")
        else:
            st.markdown(f"_{role}:_ {message}")