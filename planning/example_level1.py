from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# Ví dụ này gộp bước plan và execute chung trong 1 lần call LLM, đơn giản, không tách và không có cơ chế thích ứng để tự tinh chỉnh plan

# 1. LLM
llm = ChatOllama(model="llama3:8b", temperature=0)

# 2. Prompt (thay cho Agent + Task)
topic = "The importance of Reinforcement Learning in AI"

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an expert technical writer and planner. "
            "Always create a clear plan before writing.",
        ),
        (
            "user",
            f"""
            1. Create a bullet-point plan for a summary on the topic: "{topic}"
            2. Then write a concise summary (~200 words) based on that plan

            Output format:

            ### Plan
            - ...

            ### Summary
            - ...
            """,
        ),
    ]
)

# 3. Chain
chain = prompt | llm | StrOutputParser()

# 4. Run
print("## Running LangChain Planning Example ##")
result = chain.invoke({})
print("\n---\n## Result ##\n---")
print(result)
