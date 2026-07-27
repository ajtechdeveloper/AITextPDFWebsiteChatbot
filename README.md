# AITextPDFWebsiteChatbot
AI chatbot that answers to questions from the input text, PDF or website link

Architecture:

```
Streamlit
    |
    ↓
Document loaders (PDF / URL / Text)
    |
    ↓
Text splitter
    |
    ↓
FAISS vector store
    |
    ↓
Retriever
    |
    ↓
LangChain RAG chain
    |
    ↓
OpenAI-compatible Hugging Face Router API
    |
    ↓
google/gemma-4-31B-it:cerebras
```
