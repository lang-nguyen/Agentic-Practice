from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableBranch, RunnablePassthrough
from dotenv import load_dotenv

load_dotenv()
# llm = ChatGoogleGenerativeAI(model="gemini-4-flash")
# llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
llm = ChatOllama(model="llama3:8b", temperature=0)


def greet_handler(request: str) -> str:
    print("---------Greet Handler--------")
    return f"greet handler processed: {request}"


def bye_handler(request: str) -> str:
    print("---------Bye Handler--------")
    return f"bye handler processed: {request}"


def default_handler(request: str) -> str:
    print("---------Default Handler--------")
    return f"default handler processed: {request}"


router_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
     Classify the user's request as one of the following:
     - 'greet': if user is saying hello
     - 'bye': if user is saying goodbye
     - 'default': otherwise
     ONLY output one word: 'greet', 'bye' or 'default'
     """,
        ),
        ("user", "{request}"),
    ]
)

router_chain = router_prompt | llm | StrOutputParser()


def is_greet(x):
    return x["decision"].strip() == "greet"


def is_bye(x):
    return x["decision"].strip() == "bye"


greet_branch = RunnablePassthrough.assign(output=lambda x: greet_handler(x["request"]))
bye_branch = RunnablePassthrough.assign(output=lambda x: bye_handler(x["request"]))
default_branch = RunnablePassthrough.assign(
    output=lambda x: default_handler(x["request"])
)

# Uy quyen
delegation_chain = RunnableBranch(
    (is_greet, greet_branch), (is_bye, bye_branch), default_branch
)

# Agent dieu phoi
coordinator_agent = (
    {"decision": router_chain, "request": RunnablePassthrough()}
    | delegation_chain
    | (lambda x: x["output"])
)

# Test
test_requests = [
    "Hello!",
    "Goodbye",
    "Are you stupid",
    "My name is Lang",
    "How is going?",
]

for i, req in enumerate(test_requests, 1):
    print(f"------ Test request {i}: {req}---------")
    result = coordinator_agent.invoke({"request": req})
    print(f"------ Result {i}: {result}---------")
