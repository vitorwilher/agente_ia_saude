"""LangGraph ReAct agent with a single retriever tool and per-thread memory."""

from typing import Optional

from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent

from globals import configs
from rag import get_llm, get_retriever

_retriever = None


def _lazy_retriever():
    global _retriever
    if _retriever is None:
        _retriever = get_retriever()
    return _retriever


@tool
def search_health_documents(query: str) -> str:
    """Busca trechos relevantes nos relatórios epidemiológicos da ECDC (Zika,
    Sífilis e Tuberculose de 2021) para responder perguntas factuais.

    Args:
        query: Pergunta ou termo em linguagem natural. Pode estar em português
            ou inglês. Exemplos: "casos de Zika em 2021 na União Europeia",
            "taxa de sucesso tratamento tuberculose", "coinfecção sífilis HIV".

    Returns:
        Trechos concatenados dos PDFs, separados por marcadores de fonte. Se
        nada relevante for encontrado, retorna string vazia.
    """
    docs = _lazy_retriever().invoke(query)
    if not docs:
        return ""
    parts = []
    for d in docs:
        source = d.metadata.get("source", "desconhecido")
        page = d.metadata.get("page")
        header = f"[fonte: {source}" + (f", pág. {page}" if page is not None else "") + "]"
        parts.append(f"{header}\n{d.page_content}")
    return "\n\n---\n\n".join(parts)


_agent = None
_checkpointer = MemorySaver()


def _build_agent():
    return create_react_agent(
        model=get_llm(),
        tools=[search_health_documents],
        prompt=configs["agent"]["system_prompt"],
        checkpointer=_checkpointer,
    )


def get_agent():
    global _agent
    if _agent is None:
        _agent = _build_agent()
    return _agent


def invoke_agent(prompt: str, thread_id: Optional[str] = None) -> str:
    """Send a user message to the agent and return the final text response.

    Args:
        prompt: User question.
        thread_id: Stable id for the conversation. Same id → same memory.
            If None, each call is a fresh conversation.
    """
    agent = get_agent()
    config = {"configurable": {"thread_id": thread_id or "default"}}
    result = agent.invoke(
        {"messages": [{"role": "user", "content": prompt}]},
        config=config,
    )
    return result["messages"][-1].content
