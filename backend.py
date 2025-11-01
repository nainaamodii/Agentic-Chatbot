# Imports
from langgraph.graph import StateGraph , START , END
from typing import TypedDict, Annotated
from langchain_core.messages import HumanMessage , BaseMessage, AIMessage
from langchain_groq import ChatGroq
from langgraph.graph.message import add_messages
from dotenv import load_dotenv
import os
from langgraph.checkpoint.memory import MemorySaver

load_dotenv()

# Define the state structure
class state(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages] #list of messages

# Initialize the LLM
llm=ChatGroq(model_name="llama-3.1-8b-instant",temperature=0.9)
def chat_node(state:state) -> state:
    
    # take user query
    messages=state.get('messages', [])
    # sent to llm
    response = llm.invoke(messages)
    # strore response
    return {'messages': messages + [AIMessage(content=response.content)]}

# build Graph
graph=StateGraph(state)

# add nodes
graph.add_node('chat_node',chat_node)

# add edges
graph.add_edge(START,'chat_node')
graph.add_edge('chat_node',END)

# add memory saver for persistence
checkpointer=MemorySaver()

# create graph object
bot=graph.compile(checkpointer=checkpointer)

# test
config1={"configurable": {"thread_id":"1"}}
stream =bot.stream(
    {'messages':[HumanMessage(content="Hello, can you help me with my robotics project?")]},
    stream_mode="messages",
    config=config1
)

for message_chunk, metadata in stream:
    print(message_chunk.content , end=" " , flush=True)


