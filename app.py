import streamlit as st
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate

# Page configuration
st.set_page_config(page_title="Chat with PDF", layout="wide")
st.title("PDF Chat Application")

# Initialize session states
if "conversation" not in st.session_state:
    st.session_state.conversation = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "processComplete" not in st.session_state:
    st.session_state.processComplete = None

# Custom prompt template
custom_template = """
You are a helpful AI assistant that answers questions based solely on the provided PDF content.
If the question cannot be answered using the information from the PDF, respond with "I don't know. This information is not in the PDF."

Context: {context}
Chat History: {chat_history}
Question: {question}

Please provide an answer based only on the context provided above. If the information isn't in the context, say "I don't know. This information is not in the PDF."

Answer:"""

CUSTOM_PROMPT = PromptTemplate(
    input_variables=["context", "chat_history", "question"],
    template=custom_template
)

def process_pdf(uploaded_file, openai_api_key):
    # Load PDF
    loader = PyPDFLoader(uploaded_file)
    documents = loader.load()
    
    # Split documents into chunks
    text_splitter = CharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separator="\n",
    )
    chunks = text_splitter.split_documents(documents)
    
    # Create embeddings and vectorstore
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    vectorstore = FAISS.from_documents(chunks, embeddings)
    
    # Create conversation chain
    llm = ChatOpenAI(
        temperature=0.1,  # Lower temperature for more focused answers
        model_name='gpt-3.5-turbo',
        openai_api_key=openai_api_key
    )
    
    memory = ConversationBufferMemory(
        memory_key='chat_history',
        return_messages=True
    )
    
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(
            search_kwargs={"k": 3}  # Retrieve top 3 most relevant chunks
        ),
        memory=memory,
        combine_docs_chain_kwargs={'prompt': CUSTOM_PROMPT},
        
    )
    
    return conversation_chain

# Sidebar for API key and PDF upload
with st.sidebar:
    st.header("Configuration")
    openai_api_key = st.text_input("Enter your OpenAI API key:", type="password")
    uploaded_file = st.file_uploader("Upload your PDF file:", type=['pdf'])
    
    # Add a button to reset the chat
    if st.button("Reset Chat"):
        st.session_state.conversation = None
        st.session_state.chat_history = []
        st.session_state.processComplete = None
        st.experimental_rerun()
    
    if uploaded_file and openai_api_key and not st.session_state.processComplete:
        with st.spinner("Processing PDF..."):
            # Save uploaded file temporarily
            with open("temp.pdf", "wb") as f:
                f.write(uploaded_file.getvalue())
            
            st.session_state.conversation = process_pdf("temp.pdf", openai_api_key)
            st.session_state.processComplete = True

# Main chat interface
if st.session_state.conversation:
    # Chat input
    user_question = st.chat_input("Ask a question about your PDF:")
    
    if user_question:
        try:
            response = st.session_state.conversation({"question": user_question})
            st.session_state.chat_history.append((user_question, response["answer"]))
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.session_state.chat_history.append((user_question, "I encountered an error while processing your question."))

    # Display chat history
    for question, answer in st.session_state.chat_history:
        with st.chat_message("user"):
            st.write(question)
        with st.chat_message("assistant"):
            st.write(answer)

else:
    st.info("Please upload a PDF file and enter your OpenAI API key in the sidebar to start chatting!")