from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage

load_dotenv()
# Model dong 2 rule: Producer va Critic
llm = ChatOllama(model="llama3:8b", temperature=0)


def run_reflection_code():
    """Demonstate a multi step AI reflection loop to progressively improve a python function"""

    task_prompt = """
    Your task is to create a Python function named `calculate_factorial`. 
    This function should do the following: 
    1. Accept a single integer `n` as input. 
    2. Calculate its factorial (n!). 
    3. Include a clear docstring explaining what the function does. 
    4. Handle edge cases: The factorial of 0 is 1. 
    5. Handle invalid input: Raise a ValueError if the input is a negative number.
    """

    max_interations = 3
    current_code = ""  # output of agent

    message_history = [HumanMessage(content=task_prompt)]  # memory

    for i in range(max_interations):
        print(
            f"---------------------Reflection loop: interations {i+1}---------------------"
        )
        if i == 0:
            print("Stage 1: Generating initial code")
            response = llm.invoke(message_history)
            current_code = response.content
        else:
            print("Refining stage:")
            message_history.append(
                HumanMessage(
                    content="Please refine the code using the critiques provided"
                )
            )
            response = llm.invoke(message_history)
            current_code = response.content
        message_history.append(response)

        print("Stage 2: Reflect on the generated code")
        reflection_prompt = [
            SystemMessage(
                content="""
               You are a senior Python engineer performing a strict code review.

                Your job is to evaluate the code ONLY based on the original task requirements.

                IMPORTANT RULES:
                - Do NOT change the original requirements
                - Do NOT introduce new requirements
                - Do NOT suggest alternative algorithms unless the current one is incorrect
                - Do NOT over-engineer solutions

                Evaluate the code based on:
                1. Correctness (does it meet the requirements?)
                2. Bugs or logical errors
                3. Missing edge cases
                4. Code clarity and readability

                Return your critique in EXACTLY this format:

                MUST FIX:
                - Issues that violate requirements or cause incorrect behavior

                SHOULD IMPROVE:
                - Minor issues (style, clarity, redundancy)

                NICE TO HAVE:
                - Optional improvements (only if truly useful)

                If and ONLY if there are NO issues in ANY category, return:
                CODE_IS_PERFECT
                """
            ),
            HumanMessage(
                content=f"Original Task:\n{task_prompt}\n\nCode to review:\n{current_code} "
            ),
        ]
        critique_response = llm.invoke(reflection_prompt)
        critique = critique_response.content

        # stopping condition
        if "CODE_IS_PERFECT" in critique:
            print("No further critiques requiired or found. Code is done")
            break
        print("Critique: ", critique)
        message_history.append(
            HumanMessage(content=f"Critiques for the previous code: {critique}")
        )
    print("Final code")
    print(current_code)


run_reflection_code()

# Task → Generate → Critique → Improve → Loop → Final
# Producer (viết code)
# Critic (review code)
# Ban đầu kêu LLM làm task_prompt (viết factorial), model generate lần đầu,
# sau đó lưu code vào current_code, memory lúc này:
# User: viết factorial
# AI: code version 1
# Sau đó, model chuyển vai thành Critic (llm.invoke(reflection_prompt))
# Bắt đầu review, nếu response trả về chứa PERFECT thì code is done, không thì đưa review đó vào trong memory,
# Sau đó đến loop tiếp theo, llm thành producer, sẽ xem trong memory (gồm task, "Please refine...",code trước đó, review của Critic)
# để tạo ra code mới sau tinh chỉnh, sau đó tiếp tục quy trình như trên cho đến khi perfect code hoặc số lần lặp
