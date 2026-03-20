from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory


# 1. LLM
llm = ChatOllama(model="llama3:8b")

# 2. Prompt
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a friendly assistant."),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{question}"),
    ]
)

chain = prompt | llm

# 3. Memory store (per session)
store = {}


def get_session_history(session_id: str):
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


# 4. Wrap với memory
chain_with_memory = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="question",
    history_messages_key="history",
)

# 5. Run
config = {"configurable": {"session_id": "user_1"}}

response1 = chain_with_memory.invoke({"question": "Hi, I'm Jane."}, config=config)
print(response1.content)

response2 = chain_with_memory.invoke(
    {"question": "Do you remember my name?"}, config=config
)
print(response2.content)
