from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Ví dụ này minh họa planning thật sự: Plan → Execute từng bước → Evaluate → Re-plan → Loop
# với Planner → Executor → Reflector
# Có planning, execute riêng, loop, adapt và state

load_dotenv()

llm = ChatOllama(model="llama3:8b", temperature=0)

# ===== 1. PLANNER =====
planner_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a planning agent."),
        ("user", "Create a step-by-step plan to accomplish this goal:\n{goal}"),
    ]
)
planner_chain = planner_prompt | llm | StrOutputParser()


# ===== 2. EXECUTOR =====
executor_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are an execution agent."),
        (
            "user",
            "Goal: {goal}\n"
            "Plan:\n{plan}\n"
            "Completed steps:\n{history}\n\n"
            "Execute the NEXT step only.",
        ),
    ]
)
executor_chain = executor_prompt | llm | StrOutputParser()


# ===== 3. REFLECTOR =====
reflector_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a strict reviewer."),
        (
            "user",
            "Goal: {goal}\n"
            "Plan:\n{plan}\n"
            "Progress so far:\n{history}\n\n"
            "Is the task complete? If yes, say DONE.\n"
            "Otherwise, say CONTINUE and explain what's missing.",
        ),
    ]
)
reflector_chain = reflector_prompt | llm | StrOutputParser()


# ===== MAIN LOOP =====
def run_planning_agent(goal: str, max_steps=5):
    print(f"\n Goal: {goal}\n")

    # Step 1: Create plan
    plan = planner_chain.invoke({"goal": goal})
    print(" PLAN:\n", plan)

    history = ""

    for step in range(max_steps):
        print(f"\n Step {step+1}")

        # Step 2: Execute next step
        result = executor_chain.invoke({"goal": goal, "plan": plan, "history": history})

        print(" Execution result:\n", result)

        history += f"\nStep {step+1}: {result}\n"

        # Step 3: Reflect
        review = reflector_chain.invoke(
            {"goal": goal, "plan": plan, "history": history}
        )

        print(" Reflection:\n", review)

        if "DONE" in review:
            print("\n Task completed!")
            break

    print("\n FINAL RESULT:\n", history)


# ===== RUN =====
if __name__ == "__main__":
    run_planning_agent("Write a concise summary about Reinforcement Learning in AI")
