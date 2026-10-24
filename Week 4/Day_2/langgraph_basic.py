from typing import TypedDict
from langgraph.graph import StateGraph, START, END

class State(TypedDict):
    number: int

def double(state:State) -> dict:
    num=state["number"]
    newNum=2*num
    print("in double fn")
    print(newNum)
    return{
        "number": newNum
    }

def finish(state:State) -> dict:
    num=state["number"]
    print("in finish fn")
    print(num)
    return{
        "number":num
    }

def decision(state:State) -> str:
    num=state["number"]
    if num<100:
        return "double"
    else:
        return "finish"

builder=StateGraph(State)
builder.add_node("double",double)
builder.add_node("finish",finish)
builder.add_edge(START,"double")
builder.add_conditional_edges("double",decision,{"double":"double","finish":"finish"})
builder.add_edge("finish",END)

graph=builder.compile()

if __name__ == "__main__":
    result = graph.invoke({"number": 1})
    print(result)