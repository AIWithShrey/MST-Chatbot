import streamlit as st
import os
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import LlamaCpp
from langchain.prompts import PromptTemplate
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.agents import AgentExecutor, create_react_agent
import tempfile

os.environ['TAVILY_API_KEY'] = "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"

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

if 'search_results' not in st.session_state:
    st.session_state.search_results = {}

def format_chat_history(history):
    formatted_history = ""
    for exchange in history:
        formatted_history += f'User: {exchange["Human"]}\nChatACM: {exchange["ChatACM"]}\n'
    return formatted_history

tools = [TavilySearchResults(max_results=3)]

def format_latest_search_result(query):
    # Fetch the latest result for the given query
    if query in st.session_state.search_results:
        latest_result = st.session_state.search_results[query][-1]  # Get the last search result for this query
        formatted_result = f"Latest search result: {latest_result}"  # Format the result
        return formatted_result
    return "No web search result found."

# Initialize Llama model
model_path = "models/gemma-7b-it/gemma-7b-it-q8_0.gguf"
#model = Llama(model_path=model_path, chat_format="chatml", n_gpu_layers=-1, n_batch=2048)
model = LlamaCpp(model_path=model_path,
                 #chat_format="conv", 
                 n_gpu_layers=-1, 
                 n_batch=4096,
                 n_ctx=4096, 
                 temperature=0.5)

template = """
ChatACM is a large language model trained by ACM AI at Missouri University of Science and Technology.

ChatACM is designed to be able to assist students of Missouri University Science and Technology (MS&T) with a wide range of tasks, from answering simple questions to providing in-depth explanations and discussions on a wide range of topics. As a language model, ChatACM is able to generate human-like text based on the input it receives, allowing it to engage in natural-sounding conversations and provide responses that are coherent and relevant to the topic at hand.

ChatACM is constantly learning and improving, and its capabilities are constantly evolving. It is able to process and understand large amounts of text, and can use this knowledge to provide accurate and informative responses to a wide range of questions. Additionally, ChatACM is able to generate its own text based on the input it receives, allowing it to engage in discussions and provide explanations and descriptions on a wide range of topics.

Overall, ChatACM is a powerful tool that can help with a wide range of tasks and provide valuable insights and information on a wide range of topics. Whether you need help with a specific question or just want to have a conversation about a particular topic, ChatACM is here to assist.

TOOLS:
------

ChatACM has access to the following tools:

{tools}

To use a tool, please use the following format:

```
Thought: Do I need to use a tool? Yes
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
```

When you have a response to say to the Human, or if you do not need to use a tool, you MUST use the format:

```
Thought: Do I need to use a tool? No
Final Answer: [your response here]
```

Begin!

Previous conversation history:
{chat_history}

New input: {input}
{agent_scratchpad}
"""
prompt = PromptTemplate(input_variables=['agent_scratchpad', 'input', 'tool_names', 'tools'], template=template)

# Streamlit app setup
st.set_page_config(layout="wide", page_title="ChatACM v2.0")
st.title("ChatACM v2.0")
st.text("S&T's very own chatbot. Proudly presented by ACM AI.")
st.text("This is currently a work in progress. Your patience is much appreciated.")
st.text("ChatACM can now browse the web for you! Simply ask your question and it will respond with up-to-date information.")
st.caption("Questions? Reach out to Shreyas Mocherla: ")
st.caption("Email - vmgng@umsystem.edu")
st.caption("ACM AI Discord - https://discord.gg/rvkW7HM58k")
col1, col2 = st.columns([0.9, 0.6])
with col1:
    user_input = st.text_input("You:", "")
with col2:
    uploaded_file = st.file_uploader("Choose a document", type=['pdf', 'docx', 'txt'])

# Function to get a response from the model
def get_response(message):
    chat_history = format_chat_history(st.session_state.chat_history)
    agent = create_react_agent(model, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, handle_parsing_errors=True)
    response = agent_executor.invoke({"input": message, "chat_history": chat_history})
    return response['output']

# Function to load and chunk document data
def load_document(file):
    # Create a temporary file to save the uploaded file's content
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.name)[1]) as tmp:
        tmp.write(file.getvalue())  # Write the content of the uploaded file to the temporary file
        tmp_path = tmp.name  # Store the path of the temporary file

    # Now use tmp_path with the appropriate loader
    if tmp_path.endswith('.pdf'):
        loader = PyPDFLoader(tmp_path)
    elif tmp_path.endswith('.docx'):
        loader = Docx2txtLoader(tmp_path)
    elif tmp_path.endswith('.txt'):
        loader = TextLoader(tmp_path)
    else:
        st.error('Document format is not supported!')
        return None

    data = loader.load()
    os.unlink(tmp_path)  # Clean up the temporary file
    return data

def chunk_data(data, chunk_size=512):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=2)
    chunks = text_splitter.split_documents(data)
    return chunks

# Function to insert or fetch embeddings
def insert_or_fetch_embeddings(chunks):
    embeddings_model = GPT4AllEmbeddings()
    vector_store = FAISS.from_documents(chunks, embeddings_model)
    return vector_store

# Function to get response with document
def get_response_with_doc(user_input, uploaded_file):
    data = load_document(uploaded_file)
    if data:
        chunks = chunk_data(data)
        #index_name = "llm-app"  # Define a consistent index name
        vector_store = insert_or_fetch_embeddings(chunks)
        retriever = vector_store.as_retriever(search_type='similarity', search_kwargs={'k': 3})
        chain = RetrievalQA.from_chain_type(llm=model, chain_type="stuff", retriever=retriever)
        answer = chain.invoke(user_input)
        return answer['result']
    return "Could not process document."

def handle_response(user_input):
    if user_input:
        response = get_response(user_input)
        # Update chat history
        st.session_state.chat_history.append({"Human": user_input, "ChatACM": response})
        return response
    return ""

# Display bot response based on Human input and/or uploaded file
if uploaded_file is not None and user_input:
    bot_response = get_response_with_doc(user_input, uploaded_file)
else:
    bot_response = handle_response(user_input) if user_input else ""

st.text_area("ChatACM:", value=bot_response, height=200, key="bot_response_area")
