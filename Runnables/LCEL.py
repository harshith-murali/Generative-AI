"""
===============================================================================
                       LANGCHAIN EXPRESSION LANGUAGE (LCEL)
===============================================================================

1. CORE CONCEPT
   LCEL is a declarative language used to build and compose standard execution 
   chains in LangChain. Every component that adheres to this system implements 
   the 'Runnable' interface.

2. STANDARD RUNNABLE COMPONENTS
   - RunnableSequence : Standard pipeline executing items sequentially.
   - RunnableParallel : Executes multiple runnables concurrently on identical input.
   - RunnablePassthrough : Forwards incoming inputs unchanged.
   - RunnableLambda : Wraps arbitrary Python functions into runnable components.
   - RunnableBranch : Conditional routing mechanism based on custom predicate checks.

3. EXECUTION INTERFACES
   - invoke(input) : Direct single-input sync execution.
   - batch([input1, input2]) : Concurrent execution over multiple inputs.
   - stream(input) : Yields streamed tokens real-time.
   - ainvoke / abatch / astream : Async equivalents for non-blocking setups.
===============================================================================
"""

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import (
    RunnableBranch,
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
    RunnableSequence,
)

load_dotenv()

# --- MODEL SETUP ---
model = ChatAnthropic(
    model="claude-haiku-4-5-20251001",
    max_tokens=1000,
)
parser = StrOutputParser()


# --- SECTION 1: SEQUENTIAL EXECUTION (RunnableSequence) ---
# Inputs flow sequentially from standard prompt to model to parser.

seq_prompt = PromptTemplate(
    template="Define {concept} in one sentence.",
    input_variables=["concept"],
)

single_seq_chain = RunnableSequence(seq_prompt, model, parser)


# --- SECTION 2: PARALLEL EXECUTION & PASSTHROUGH (RunnableParallel) ---
# Generates base response, then executes multiple evaluation branches in parallel.

gen_prompt = PromptTemplate(
    template="Write a short summary about {topic}.",
    input_variables=["topic"],
)

gen_chain = RunnableSequence(gen_prompt, model, parser)

eval_parallel_chain = RunnableParallel({
    "original_text": RunnablePassthrough(),
    "character_count": RunnableLambda(lambda text: len(text)),
    "word_count": RunnableLambda(lambda text: len(text.split())),
})

full_parallel_chain = RunnableSequence(gen_chain, eval_parallel_chain)


# --- SECTION 3: BATCH & STREAMING MODES ---
# Demonstrates alternatives to default single invoke.

batch_inputs = [
    {"concept": "Dynamic Programming"},
    {"concept": "Memoization"},
]


# --- SECTION 4: CONDITIONAL ROUTING (RunnableBranch) ---
# Routes processing conditionally depending on the classifier evaluation.

classify_prompt = PromptTemplate(
    template="Classify key '{input}' as 'code' or 'text'. Output only the category.",
    input_variables=["input"],
)

code_prompt = PromptTemplate(
    template="Optimize this C++ solution: {input}",
    input_variables=["input"],
)

text_prompt = PromptTemplate(
    template="Summarize this text: {input}",
    input_variables=["input"],
)

classifier_chain = RunnableSequence(classify_prompt, model, parser)

branch_router = RunnableBranch(
    (lambda x: "code" in x["category"].lower(), RunnableSequence(code_prompt, model, parser)),
    RunnableSequence(text_prompt, model, parser)
)

conditional_chain = RunnableSequence(
    RunnableParallel({
        "category": classifier_chain,
        "input": RunnablePassthrough()
    }),
    branch_router
)


# --- EXECUTION ---
if __name__ == "__main__":
    print("--- 1. SINGLE SEQUENCE EXECUTION ---")
    print(single_seq_chain.invoke({"concept": "Recursion"}))
    print("\n")

    print("--- 2. PARALLEL PROCESSING ---")
    print(full_parallel_chain.invoke({"topic": "Garbage Collection"}))
    print("\n")

    print("--- 3. BATCH EXECUTION ---")
    batch_results = single_seq_chain.batch(batch_inputs)
    for result in batch_results:
        print(result)
    print("\n")

    print("--- 4. STREAMING EXECUTION ---")
    for chunk in single_seq_chain.stream({"concept": "Tail Call Optimization"}):
        print(chunk, end="", flush=True)
    print("\n\n")

    print("--- 5. CONDITIONAL ROUTING ---")
    print(conditional_chain.invoke({"input": "int x = 10; x += 5;"}))