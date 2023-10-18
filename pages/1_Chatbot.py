import streamlit as st
import os
import time
import datetime
import pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain.memory import ConversationBufferWindowMemory
from langchain import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains.llm import LLMChain
from langchain.chat_models import ChatOpenAI
#from langchain.tools import DuckDuckGoSearchRun
from langchain.callbacks import get_openai_callback
from langchain.text_splitter import RecursiveCharacterTextSplitter
#from langchain.callbacks import StreamlitCallbackHandler
# from langchain.embeddings import HuggingFaceInstructEmbeddings
import anthropic
from langchain.chat_models import ChatAnthropic
# from langchain.embeddings import HuggingFaceInferenceAPIEmbeddings


# Setting up Streamlit page configuration
st.set_page_config(
    layout="centered",  
    initial_sidebar_state="expanded"
)

# Getting the OpenAI API key from Streamlit Secrets
anthropic_api_key = st.secrets.secrets.ANTHROPIC_API_KEY #OPENAI_API_KEY
os.environ["ANTHROPIC_API_KEY"] = anthropic_api_key
openai_api_key = st.secrets.secrets.OPENAI_API_KEY #OPENAI_API_KEY
os.environ["OPENAI_API_KEY"] = openai_api_key

# inference_api_key = st.secrets.secrets.INFERENCE_API_KEY

# Getting the Pinecone API key and environment from Streamlit Secrets
PINECONE_API_KEY = st.secrets.secrets.PINECONE_API_KEY
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
PINECONE_ENV = st.secrets.secrets.PINECONE_ENV
os.environ["PINECONE_ENV"] = PINECONE_ENV
# Initialize Pinecone with API key and environment
pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)

param1 = True
#@st.cache_data
def select_index():
    if param1:
        pinecone_index_list = pinecone.list_indexes()
    return pinecone_index_list

# Set the text field for embeddings
text_field = "text"
# Create OpenAI embeddings
embeddings = OpenAIEmbeddings(model = 'text-embedding-ada-002')
# embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
# embeddings = HuggingFaceInferenceAPIEmbeddings(
#     api_key=inference_api_key,
#     model_name= "hkunlp/instructor-xl" #"sentence-transformers/all-MiniLM-l6-v2"
# )
MODEL_OPTIONS = ["claude-2", "claude-instant-1"]
model_name = st.sidebar.selectbox(label="Select Model", options=MODEL_OPTIONS)
# lang_options = ["English", "German", "French", "Chinese", "Italian", "Japanese", "Arabic", "Hindi", "Turkish", "Urdu", "Russian", "Georgian"]
# lang_dic = {"English":"\nAnswer in English", "German":"\nAnswer in German", "French":"\nAnswer in French", "Chinese":"\nAnswer in Chinese", "Italian":"\nAnswer in Italian", "Japanese":"\nAnswer in Japanese", "Arabic":"\nAnswer in Arabic", "Hindi":"\nAnswer in Hindi", "Turkish":"\nAnswer in Turkish", "Urdu":"\nAnswer in Urdu", "Russian":"\nAnswer in Russian language", "Georgian":"\nAnswer in Georgian language"}
# language = st.sidebar.selectbox(label="Select Language", options=lang_options)

@st.cache_resource
def ret(pinecone_index):
    if pinecone_index != "":
        # load a Pinecone index
        index = pinecone.Index(pinecone_index)
        time.sleep(5)
        db = Pinecone(index, embeddings.embed_query, text_field)
    return db

@st.cache_resource
def init_memory():
    return ConversationBufferWindowMemory(
                                        k=5, 
                                        memory_key="chat_history", 
                                        return_messages=True,
                                        verbose=True)
memory = init_memory()

# pt = lang_dic[language]

pinecone_index_list = select_index()
pinecone_index = st.sidebar.selectbox(label="Select Index", options = pinecone_index_list )
TEMPERATURE_MIN_VALUE = 0.0
TEMPERATURE_MAX_VALUE = 1.0
TEMPERATURE_DEFAULT_VALUE = 0.9
TEMPERATURE_STEP = 0.01
top_p = st.sidebar.slider("Top_P", 0.0, 1.0, 0.7, 0.1)
top_k = st.sidebar.slider("Top_K", 0, 100, 55, 1)
temperature = st.sidebar.slider(
            label="Temperature",
            min_value=TEMPERATURE_MIN_VALUE,
            max_value=TEMPERATURE_MAX_VALUE,
            value=TEMPERATURE_DEFAULT_VALUE,
            step=TEMPERATURE_STEP,)



# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

templat_1 = """You are conversational AI assistant and responsible to answer user queries in a conversational manner. 

You always provide useful information & details available in the given sources with long and detailed answer.

Answer the Follow Up Input by user according to the query and what is asked or said by user in follow up input.

"""

templat_2 = """
\nChat History:
{chat_history}
Follow Up Input: {question}

Use the following context to answer questions if necessary or according to follow up input question.
"""
prompt_opt = st.sidebar.selectbox(label="Select Prompt Option", options = ["Use Default Prompt", "Use Custom Prompt"])

def prom(prompt_opt):
    if prompt_opt == "Use Default Prompt":
        templat = templat_1 + templat_2
        #st.sidebar.write(templat_1 + templat_2)
        return templat
    elif prompt_opt == "Use Custom Prompt":
        u_input = st.sidebar.text_area("Write your prompt here: ", "", placeholder=templat_1)
        templat = u_input + templat_2
        return templat

# chatGPT_template = """Assistant is a large language model trained by OpenAI.

# Assistant is designed to be able to assist with a wide range of tasks, from answering simple questions to providing in-depth explanations and discussions on a wide range of topics. As a language model, Assistant is able to generate human-like text based on the input it receives, allowing it to engage in natural-sounding conversations and provide responses that are coherent and relevant to the topic at hand.

# Assistant is constantly learning and improving, and its capabilities are constantly evolving. It is able to process and understand large amounts of text, and can use this knowledge to provide accurate and informative responses to a wide range of questions. Additionally, Assistant is able to generate its own text based on the input it receives, allowing it to engage in discussions and provide explanations and descriptions on a wide range of topics.

# Overall, Assistant is a powerful tool that can help with a wide range of tasks and provide valuable insights and information on a wide range of topics. Whether you need help with a specific question or just want to have a conversation about a particular topic, Assistant is here to assist.

# {history}
# Human: {human_input}
# Assistant:"""

# chatGPT_prompt = PromptTemplate(input_variables=["history", "human_input"], template=chatGPT_template)

# chatgpt_chain = LLMChain(
#     llm=OpenAI(temperature=0),
#     prompt=chatGPT_prompt,
#     verbose=True,
#     memory=ConversationBufferWindowMemory(k=2),
# )
# quest_prompt = """Generate a standalone question which is based on the new question plus the chat history. 
# Chat History:
# {chat_history}
# Just create the standalone question without commentary. New question: {question}"""
# q_prompt = PromptTemplate(input_variables=["chat_history", "question"], template=quest_prompt)
# quest_gpt = LLMChain(
#     llm=ChatAnthropic(model=model_name),
#     prompt=q_prompt,
#     verbose=True
# )

# template = template + pt
# @st.cache_resource
templat = prom(prompt_opt)
st.sidebar.write(templat)
def chat(pinecone_index, query):

    db = ret(pinecone_index)
    # search = DuckDuckGoSearchRun()
    retriever=db.as_retriever()
    

    # @st.cache_resource
    # def agent_meth(query, pt):

    #quest = quest_gpt.predict(question=query, chat_history=st.session_state.messages)

    #web_res = search.run(quest)
    doc_res = db.similarity_search(query, k=2)
    result_string = ' '.join(stri.page_content for stri in doc_res)
    #output = chatgpt_chain.predict(human_input=quest)
    contex = "\nSource: " + result_string +"\nAssistant:"
    templ = templat + contex
    promptt = PromptTemplate(input_variables=["chat_history", "question"], template=templ)
    agent = LLMChain(
        llm=ChatAnthropic(model=model_name, temperature=temperature, top_p=top_p, top_k=top_k, max_tokens_to_sample=99999, streaming=True),
        prompt=promptt,
        verbose=True,
        memory=memory
                                                
    )
        
        
    return agent, contex, result_string#, quest
    

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input():
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content":prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        # st_callback = StreamlitCallbackHandler(st.container(),
        #                                     #    expand_new_thoughts=True, 
        #                                     collapse_completed_thoughts=True)

        
        #st.sidebar.write("standalone question: ", quest)
        with st.spinner("Thinking..."):
            #with get_openai_callback() as cb:
            agent, contex, result_string = chat(pinecone_index, prompt)
            response = agent.predict(question=prompt, chat_history = st.session_state.messages)#,callbacks=[st_callback])#, callbacks=[st_callback])#.run(prompt, callbacks=[st_callback])
            st.write(response)
            st.session_state.chat_history.append((prompt, response))
            st.session_state.messages.append({"role": "assistant", "content": response})

        # st.sidebar.header("Total Token Usage:")
        # st.sidebar.write(f"""
        #         <div style="text-align: left;">
        #             <h3>   {cb.total_tokens}</h3>
        #         </div> """, unsafe_allow_html=True)
        # st.sidebar.write("Information Processing: ", "---")
        # st.sidebar.header(":red[Web Results:] ")
        # st.sidebar.write(web_res)
        # st.sidebar.write("---")
        # st.sidebar.header(":red[Database Results:] ")
        # st.sidebar.write(result_string)
        # st.sidebar.write("---")
        # st.sidebar.header(":red[ChatGPT Results:] ")
        # st.sidebar.write(output)

# if pinecone_index != "":
#     #chat(pinecone_index)
#     #st.sidebar.write(st.session_state.messages)
#     #don_check = st.sidebar.button("Download Conversation")
#     con_check = st.sidebar.button("Upload Conversation to loaded Index")
    
#     text = []
#     for item in st.session_state.messages:
#         text.append(f"Role: {item['role']}, Content: {item['content']}\n")
#     #st.sidebar.write(text)
#     if con_check:
#         text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
#         docs = text_splitter.create_documents(text)
#         st.sidebar.info('Initializing Conversation Uploading to DB...')
#         time.sleep(11)
#         # Upload documents to the Pinecone index
#         vector_store = Pinecone.from_documents(docs, embeddings, index_name=pinecone_index)
        
#         # Display success message
#         st.sidebar.success("Conversation Uploaded Successfully!")
    
#     text = '\n'.join(text)
#     # Provide download link for text file
#     st.sidebar.download_button(
#         label="Download Conversation",
#         data=text,
#         file_name=f"Conversation_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.txt",
#         mime="text/plain"
#     )


# Nucleus sampling is a technique used in large language models to control the randomness and diversity of generated text. It works by sampling from only the most likely tokens in the model’s predicted distribution.

# The key parameters are:

# Temperature: Controls randomness, higher values increase diversity.

# Top-p (nucleus): The cumulative probability cutoff for token selection. Lower values mean sampling from a smaller, more top-weighted nucleus.

# Top-k: Sample from the k most likely next tokens at each step. Lower k focuses on higher probability tokens.

# In general:

# Higher temperature will make outputs more random and diverse.

# Lower top-p values reduce diversity and focus on more probable tokens.

# Lower top-k also concentrates sampling on the highest probability tokens for each step.

# So temperature increases variety, while top-p and top-k reduce variety and focus samples on the model’s top predictions. You have to balance diversity and relevance when tuning these parameters for different applications.
