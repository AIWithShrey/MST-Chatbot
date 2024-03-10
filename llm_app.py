import streamlit as st
import os
import tempfile
import replicate
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import LlamaCpp
from langchain.prompts import PromptTemplate
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.agents import AgentExecutor, create_react_agent, Tool
from langchain import hub


st.set_page_config(layout="wide", page_title="ChatACM v2.0")

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

if 'search_results' not in st.session_state:
    st.session_state.search_results = {}

def format_chat_history(history):
    formatted_history = ""
    for exchange in history:
        formatted_history += f'User: {exchange["Human"]}\nChatACM: {exchange["ChatACM"]}\n'
    return formatted_history

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
                 chat_format="chatml", 
                 n_gpu_layers=-1, 
                 n_batch=4096,
                 n_ctx=5096, 
                 temperature=0)


def generate_image(input=""):
    output = replicate.run(
                "stability-ai/sdxl:39ed52f2a78e934b3ba6e2a89f5b1c712de7dfea535525255b1aa35c5565e08b",
                input={
                    "width": 768,
                    "height": 768,
                    "prompt": f"{input}, cinematic, dramatic",
                    "refine": "expert_ensemble_refiner",
                    "scheduler": "K_EULER",
                    "lora_scale": 0.6,
                    "num_outputs": 1,
                    "guidance_scale": 7.5,
                    "apply_watermark": False,
                    "high_noise_frac": 0.8,
                    "negative_prompt": "",
                    "prompt_strength": 0.8,
                    "num_inference_steps": 25
                }
    )

    return output

image_generator_tool = Tool(name="image_generator",
                            func=generate_image,
                            description="Generates an image based on the input prompt.")

prompt = hub.pull("hwchase17/react-chat")

template = """
You are ChatACM, a virtual assistant designed to help students at Missouri University of Science and Technology (MS&T) with a wide range of tasks, from answering simple questions to providing in-depth explanations and discussions on a wide range of topics. As a language model.
You are able to generate human-like text based on the input you receive, allowing you to engage in 
natural-sounding conversations and provide responses that are coherent and relevant to the topic at hand.

Always provide accurate and informative responses to a wide range of questions. 
Additionally, you are able to generate your own text based on the input you receive, 
allowing you to engage in discussions and provide explanations and descriptions on a wide range of topics.

You do not answer questions about personal information, such as social security numbers,
credit card numbers, or other sensitive information. You also do not provide medical, legal, or financial advice.

You will not respond to any questions that are inappropriate or offensive. You are friendly, helpful,
and you are here to assist students with any questions they may have.

If you do not find the answer you are looking for, you can use the search tool to find more information.
Keep your answers clear and concise, and provide as much information as possible to help students understand the topic.

Input: {input}

"""
prompt_template = PromptTemplate.from_template(template=template)

# Streamlit app setup

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

tools = [TavilySearchResults(max_results=1), image_generator_tool]
chat_history = st.session_state.chat_history#format_chat_history(st.session_state.chat_history)
agent = create_react_agent(model, tools=tools, prompt=prompt)
agent_executor = AgentExecutor(agent=agent, 
                                   tools=tools, 
                                   handle_parsing_errors=True,
                                   verbose=True)

# Function to get a response from the model
def get_response(message):
    response = agent_executor.invoke({"input": prompt_template.format(input=message), "chat_history": chat_history})
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
        # Assuming the response from image generator includes a URL
        if 'generate' in user_input.lower() and isinstance(response, str) and "http" in response:
            st.session_state.chat_history.append({"Human": user_input, "ChatACM": response})
            # Display the generated image
            url_start = response.find("http")
            url_end = response.find("'", url_start)
            image_url = response[url_start:url_end]
            st.markdown(f"![Alt Text]({image_url})")
            return response
        else:
            # Update chat history for text responses
            st.session_state.chat_history.append({"Human": user_input, "ChatACM": response})
            return response
    return ""

# Display bot response based on Human input and/or uploaded file
if uploaded_file is not None and user_input:
    bot_response = get_response_with_doc(user_input, uploaded_file)
else:
    bot_response = handle_response(user_input) if user_input else ""

st.text_area("ChatACM:", value=bot_response, height=200, key="bot_response_area")