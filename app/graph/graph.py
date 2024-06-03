from dotenv import load_dotenv
from langgraph.graph import END, StateGraph
from app.graph.nodes import generate_rag, retrieve, transform_query, generate_casual, response_enhancer
from app.graph.state import GraphState
from app.graph.edges import decide_to_generate, route_question, grade_answer, grade_generation_v_documents_and_question
from langgraph.graph import END, StateGraph


load_dotenv()
workflow = StateGraph(GraphState)

workflow.add_node("retrieve", retrieve) # retrieve

workflow.add_node("transform_query", transform_query)  # transform_query
workflow.add_node("generate_casual", generate_casual)  
workflow.add_node("response_enhancer", response_enhancer) # response enhancer
workflow.add_node("generate_rag", generate_rag) 


# Build graph
workflow.set_conditional_entry_point(
    route_question,
    {
        "generate_casual": "generate_casual",
        "vectorstore": "retrieve",
    },
)
workflow.add_edge("generate_casual", END)

workflow.add_edge("retrieve","response_enhancer" )


workflow.add_conditional_edges(
    "response_enhancer", # start: node
    grade_generation_v_documents_and_question, # defined function
    {
        "hallucinate": "generate_rag", #returns of the function
        "useful": END,               #returns of the function
        "not useful": "transform_query",   # GENERATION DOES NOT ADDRESS QUESTION
    },
)


workflow.add_edge("transform_query", "retrieve")
workflow.add_edge("generate_rag", "response_enhancer")



# Compile
app = workflow.compile()

app.get_graph().draw_mermaid_png(output_file_path="graph.png")