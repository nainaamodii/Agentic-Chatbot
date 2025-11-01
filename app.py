import streamlit as st
from backend import bot
from langchain_core.messages import HumanMessage, AIMessage
import uuid

# st.title("Agentic Chatbot")

# ********************************** utility functions *******************************
def generate_thread_id():
    return uuid.uuid4()

def reset_chat():
    thread_id = generate_thread_id()
    st.session_state.messages = []
    add_thread(thread_id)
    st.session_state.thread_id = thread_id
    
def add_thread(thread_id):
    if thread_id not in st.session_state.chat_threads:
        st.session_state.chat_threads.append(thread_id)

def load_conversation(thread_id):
    state = bot.get_state(config={'configurable': {'thread_id': thread_id}})
    return state.values.get('messages', [])


# **************************************** Session Setup ****************************
if "messages" not in st.session_state:
    st.session_state.messages = []

if 'thread_id' not in st.session_state:
    st.session_state.thread_id = generate_thread_id()
    
if 'chat_threads' not in st.session_state:
    st.session_state.chat_threads = []
    
add_thread(st.session_state.thread_id)

    
# ********************************Sidebar UI**********************************
st.sidebar.title("Agentic Chatbot")

if st.sidebar.button('New Chat'):
    reset_chat()
    
st.sidebar.header("Chats")

for thread_id in st.session_state.chat_threads[::-1]:
    if st.sidebar.button(str(thread_id)):
        st.session_state.thread_id = thread_id
        messages = load_conversation(thread_id)
        
        # st.session_state.messages : {'role': 'user'/'assistant', 'content': '...'}
        # messages : list of BaseMessage
        
        temp_messages = []
        for msg in messages:
            if isinstance(msg, HumanMessage) and (not temp_messages or msg.content != temp_messages[-1]["content"]):
                temp_messages.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AIMessage) and (not temp_messages or msg.content != temp_messages[-1]["content"]):
                temp_messages.append({"role": "assistant", "content": msg.content})

          
        st.session_state.messages = temp_messages      
                
  
# ******************************* Main UI **********************************

if len(st.session_state.messages) == 0:
    with st.chat_message("assistant"):
        st.markdown("Hi! How can I assist you today?")

    
# load chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"]) 
    
prompt = st.chat_input("Ask Anything?")

if prompt:
    
    # display prompt in chat message container
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
        
    CONFIG={'configurable': {'thread_id': st.session_state['thread_id']}}  #persistence
        
    # call chatbot agent   
    with st.chat_message("assistant"): 
        response_container = st.empty()
        full_response = ""
        
        # Stream the response and capture the content
        for message_chunk, metadata in bot.stream(
            {'messages':[HumanMessage(content=prompt)]},
            config=CONFIG,
            stream_mode="messages"
        ):
            if hasattr(message_chunk, 'content') and message_chunk.content:
                if message_chunk.content==full_response:
                    continue
                                
                full_response += message_chunk.content
                response_container.markdown(full_response + "▌")
        
        # Display final response without cursor
        response_container.markdown(full_response)
    
    # Store the assistant's response in session state
    st.session_state.messages.append({"role": "assistant", "content": full_response})