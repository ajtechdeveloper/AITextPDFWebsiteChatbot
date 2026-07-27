import os
import tempfile
import streamlit as st

from langchain_community.document_loaders import (
    PyPDFLoader,
    UnstructuredURLLoader,
)

from langchain_text_splitters import RecursiveCharacterTextSplitter

from huggingface_hub import InferenceClient
from langchain_core.runnables import RunnableLambda
from langchain_community.vectorstores import FAISS

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain

from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory


# -----------------------------
# Hugging Face setup
# -----------------------------

hf_token = st.secrets["HUGGINGFACE_TOKEN"]["token"]

os.environ["HUGGINGFACEHUB_API_TOKEN"] = hf_token


# -----------------------------
# Streamlit configuration
# -----------------------------

st.set_page_config(
    page_title="AI PDF Website Text Chatbot",
    page_icon="🤖"
)

st.header("AI PDF, Website, or Text Chatbot")

st.write(
    "Upload a PDF, provide a website URL, or paste text "
    "and chat with an AI assistant."
)


# -----------------------------
# Session state
# -----------------------------

if "conversation" not in st.session_state:
    st.session_state.conversation = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "input_type" not in st.session_state:
    st.session_state.input_type = None


if "message_store" not in st.session_state:
    st.session_state.message_store = {}


# -----------------------------
# LLM
# -----------------------------

def initialize_llm():

    client = InferenceClient(
        model="google/flan-t5-large",
        token=hf_token
    )

    def generate(prompt):

        response = client.text2text_generation(
            prompt,
            max_new_tokens=512,
            temperature=0.7
        )

        return response.generated_text

    return RunnableLambda(generate)


# -----------------------------
# Chat history
# -----------------------------

def get_session_history(session_id):

    if session_id not in st.session_state.message_store:
        st.session_state.message_store[session_id] = (
            InMemoryChatMessageHistory()
        )

    return st.session_state.message_store[session_id]


# -----------------------------
# Build RAG chain
# -----------------------------

def create_chatbot(documents):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.split_documents(documents)


    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


    vectorstore = FAISS.from_documents(
        chunks,
        embeddings
    )


    retriever = vectorstore.as_retriever(
        search_kwargs={
            "k": 4
        }
    )


    llm = initialize_llm()


    system_prompt = """
You are a helpful assistant.

Answer questions only from the provided context.

If the answer is not available in the context,
say:

"Sorry, I cannot answer this question based on the provided context."

Context:

{context}
"""


    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                system_prompt
            ),

            MessagesPlaceholder(
                variable_name="chat_history"
            ),

            (
                "human",
                "{input}"
            ),
        ]
    )


    document_chain = create_stuff_documents_chain(
        llm,
        prompt
    )


    retrieval_chain = create_retrieval_chain(
        retriever,
        document_chain
    )


    chatbot = RunnableWithMessageHistory(
        retrieval_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )


    return chatbot



# -----------------------------
# Input selection
# -----------------------------

input_type = st.radio(
    "Choose input type:",
    [
        "PDF",
        "Website URL",
        "Text"
    ]
)


if input_type != st.session_state.input_type:

    st.session_state.conversation = None
    st.session_state.chat_history = []

    st.session_state.input_type = input_type



if input_type == "PDF":

    uploaded_file = st.file_uploader(
        "Upload PDF",
        type="pdf"
    )


elif input_type == "Website URL":

    url_input = st.text_input(
        "Website URL"
    )


else:

    text_input = st.text_area(
        "Enter text"
    )



# -----------------------------
# Process input
# -----------------------------

if st.button("Process Input"):

    with st.spinner("Processing..."):

        try:

            documents = []


            if input_type == "PDF":

                if uploaded_file is None:
                    st.error("Upload a PDF first.")
                    st.stop()


                with tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=".pdf"
                ) as tmp:

                    tmp.write(
                        uploaded_file.read()
                    )

                    path = tmp.name


                loader = PyPDFLoader(path)

                documents = loader.load()

                os.unlink(path)



            elif input_type == "Website URL":

                loader = UnstructuredURLLoader(
                    [url_input]
                )

                documents = loader.load()



            else:

                documents = [
                    Document(
                        page_content=text_input
                    )
                ]



            st.session_state.conversation = create_chatbot(
                documents
            )


            st.success(
                "Ready! Ask questions."
            )


        except Exception as e:

            st.error(
                f"Processing failed: {e}"
            )



# -----------------------------
# Chat
# -----------------------------

if st.session_state.conversation:


    question = st.text_input(
        "Ask a question"
    )


    if st.button("Ask"):


        with st.spinner("Thinking..."):

            try:

                response = (
                    st.session_state.conversation.invoke(
                        {
                            "input": question
                        },
                        config={
                            "configurable": {
                                "session_id": "default"
                            }
                        }
                    )
                )


                answer = response["answer"]


                st.session_state.chat_history.append(
                    (
                        question,
                        answer
                    )
                )


            except Exception as e:

                st.error(
                    f"Error: {e}"
                )



# -----------------------------
# History
# -----------------------------

if st.session_state.chat_history:

    st.subheader(
        "Chat History"
    )

    for q, a in reversed(
        st.session_state.chat_history
    ):

        st.markdown(
            f"**Q:** {q}"
        )

        st.markdown(
            f"**A:** {a}"
        )

        st.divider()
