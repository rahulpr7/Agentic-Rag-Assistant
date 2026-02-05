from functools import lru_cache
from py_compile import main

from langgraph.graph import START, StateGraph

from workflows.state import WorkflowState
from workflows.nodes import (
    answer_or_retrieve,
    generate_answer,
    retrieve,
    rewrite_query,
    score_documents,
    handle_memories,
    summarization_node,
)

@lru_cache(maxsize=1)
def create_graph() -> StateGraph:
    graph = (
        StateGraph(WorkflowState)
        .add_node(handle_memories)
        .add_node("summarize_messages", summarization_node)
        .add_node(answer_or_retrieve)
        .add_node(retrieve)
        .add_node(score_documents)
        .add_node(rewrite_query)
        .add_node(generate_answer)
    )

    # Edges
    graph.add_edge(START, "summarize_messages")
    graph.add_edge(START, "handle_memories")
    graph.add_edge("summarize_messages", "answer_or_retrieve")
    graph.add_edge("retrieve", "score_documents")

    return graph

# Compiled graph
graph = create_graph().compile()

def get_workflow_graph() -> StateGraph:
    """Get the compiled workflow graph."""
    return graph


# For visualization purposes
''''def main():
    graph = get_workflow_graph()

    print("Workflow graph compiled successfully.")

    mermaid_str = graph.get_graph().draw_mermaid()

    # save to file
    with open("workflow_graph.md", "w", encoding="utf-8") as f:
        f.write(mermaid_str)

    print("Mermaid graph saved as workflow_graph.md")

if __name__ == "__main__":
    main()
    '''''