import os
from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()


def main():

    # 1. LLM
    llm = ChatOllama(model="llama3:8b", temperature=0)

    # ===== 2. RESEARCH AGENT =====
    research_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a Senior Research Analyst. "
                "Your job is to identify key trends and summarize them clearly.",
            ),
            (
                "user",
                "Research the top 3 emerging trends in Artificial Intelligence in 2024-2025. "
                "Focus on practical applications and potential impact.",
            ),
        ]
    )

    research_chain = research_prompt | llm | StrOutputParser()

    # ===== 3. WRITER AGENT =====
    writer_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a Technical Content Writer. "
                "You write clear, engaging blog posts for general audiences.",
            ),
            (
                "user",
                "Based on the following research:\n\n{research}\n\n"
                "Write a 500-word blog post that is engaging and easy to understand.",
            ),
        ]
    )

    writer_chain = writer_prompt | llm | StrOutputParser()

    # ===== 4. EXECUTION (SEQUENTIAL HANDOFF) =====
    print("## Running LangChain Multi-Agent (Research → Writer) ##")

    # Step 1: Research
    research_result = research_chain.invoke({})
    print("\n--- Research Output ---\n")
    print(research_result)

    # Step 2: Writing (handoff)
    final_output = writer_chain.invoke({"research": research_result})

    print("\n------------------\n")
    print("## Final Blog Post ##\n")
    print(final_output)


if __name__ == "__main__":
    main()
