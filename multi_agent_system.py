from __future__ import annotations
import uuid, io, base64, asyncio, re, json, os, time, datetime as _dt
from typing import Dict, List, Any, Optional, Literal, TypedDict
 
# LangChain / LangGraph
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

 
# Plotting / PDF
import plotly.graph_objects as go
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.lib import colors
 
# Local project modules (assumed present)
from vector_store import VectorStore
from llm_manager import LLMManager
from config import Config
 
# --------------------------------------------------------------------------- #
# 1.  Shared Agent State                                                      #
# --------------------------------------------------------------------------- #
 
class AgentState(dict):
    """
    Workflow state dictionary.  Using plain dict avoids attribute errors
    seen when mixing dataclass with MessagesState.
    """
 
    DEFAULT: Dict[str, Any] = {
        "query": "",
        "documents": [],             # list[Dict]
        "analysis_response": "",
        "chart_data": None,          # dict|None
        "chart_image": None,         # base64 str|None
        "report_path": None,         # str|None
        "supervisor_decision": "",
        "next_agent": "",
        "final_response": "",
        "metadata": {},              # dict
        "error": None,               # str|None
        "messages": []               # required by MessagesState
    }
 
    def __init__(self, **kwargs):
        super().__init__(self.DEFAULT | kwargs)
 
    # convenience wrappers
    def get_safe(self, key: str, default: Any = None) -> Any:
        return self.get(key, default)
 
# --------------------------------------------------------------------------- #
# 2.  Financial Tools                                                         #
# --------------------------------------------------------------------------- #
 
@tool
def retrieve_financial_documents(query: str, vector_store: VectorStore) -> Dict[str, Any]:
    """
    Retrieve up to 5 relevant document chunks from HNSW vector store.
    """
    try:
        docs = vector_store.get_relevant_documents(query, k=5, use_hybrid=True)
        context = "\n\n".join([d.page_content for d in docs])
        sources = [d.metadata.get("source", "Unknown") for d in docs]
        return {
            "context": context,
            "sources": sources,
            "documents": [d.dict() for d in docs]
        }
    except Exception as exc:
        return {"error": f"Document retrieval failed: {exc}"}
 
 
@tool
def analyze_financial_data(context: str, query: str) -> Dict[str, Any]:
    """
    Very naive regex-based metric extraction.
    """
    def _grab(pattern: str, text: str) -> List[str]:
        return re.findall(pattern, text, flags=re.I)
 
    revenue_pat = r"revenue[:\s₹rs\.in]*([0-9,\.]+)"
    profit_pat  = r"profit[:\s₹rs\.in]*([0-9,\.]+)"
    ebitda_pat  = r"ebitda[:\s₹rs\.in]*([0-9,\.]+)"
 
    try:
        analysis = {
            "revenue_figures": _grab(revenue_pat, context),
            "profit_figures":  _grab(profit_pat,  context),
            "ebitda_figures":  _grab(ebitda_pat,  context),
            "context_summary": context[:500] + ("…" if len(context) > 500 else "")
        }
        return analysis
    except Exception as exc:
        return {"error": f"Regex analysis failed: {exc}"}
 
 
@tool
def generate_chart_data(analysis: Dict[str, Any], chart_type: str = "auto") -> Dict[str, Any]:
    """
    Build a Plotly-ready data packet.
    """
    chart = {"type": "bar", "title": "Sample", "data": {}}
 
    try:
        if analysis.get("revenue_figures"):
            nums = [float(v.replace(",", "")) for v in analysis["revenue_figures"][:4]]
            chart |= {
                "type": "line",
                "title": "Revenue Trend",
                "data": {"x": [f"Q{i+1}" for i in range(len(nums))], "y": nums, "label": "Revenue"}
            }
        elif analysis.get("profit_figures"):
            nums = [float(v.replace(",", "")) for v in analysis["profit_figures"][:4]]
            chart |= {
                "type": "bar",
                "title": "Profit Comparison",
                "data": {"x": [f"P{i+1}" for i in range(len(nums))], "y": nums, "label": "Profit"}
            }
        return chart
    except Exception as exc:
        return {"error": f"Chart data build failed: {exc}"}
 
 
@tool
def create_financial_chart(chart_data: Dict[str, Any]) -> str:
    """
    Render Plotly chart; return base64 PNG.
    """
    try:
        fig = go.Figure()
        if chart_data["type"] == "line":
            fig.add_trace(go.Scatter(
                x=chart_data["data"]["x"],
                y=chart_data["data"]["y"],
                mode="lines+markers",
                name=chart_data["data"]["label"]
            ))
        elif chart_data["type"] == "bar":
            fig.add_trace(go.Bar(
                x=chart_data["data"]["x"],
                y=chart_data["data"]["y"],
                name=chart_data["data"]["label"]
            ))
        fig.update_layout(template="plotly_white", title=chart_data["title"])
        png_bytes = fig.to_image(format="png", width=800, height=600)
        return base64.b64encode(png_bytes).decode()
    except Exception as exc:
        return f"Error creating chart: {exc}"
 
 
@tool
def generate_financial_report(analysis: Dict[str, Any], chart_base64: str, query: str) -> str:
    """
    Build a quick PDF report; returns file path.
    """
    try:
        fname = f"financial_report_{_dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        doc   = SimpleDocTemplate(fname, pagesize=A4)
        styles = getSampleStyleSheet()
        story  = []
 
        # Title
        title_style = ParagraphStyle(
            "T", parent=styles["Heading1"], fontSize=24, alignment=TA_CENTER
        )
        story.extend([Paragraph("Financial Analysis Report", title_style), Spacer(1, 20)])
 
        # Query & summary
        story.append(Paragraph(f"<b>Query:</b> {query}", styles["Normal"]))
        story.append(Spacer(1, 12))
        story.append(Paragraph("<b>Summary:</b>", styles["Heading2"]))
        story.append(Paragraph(analysis.get("context_summary", "N/A"), styles["Normal"]))
        story.append(Spacer(1, 12))
 
        # Chart
        if chart_base64 and not chart_base64.startswith("Error"):
            img_bytes = base64.b64decode(chart_base64)
            story.append(Image(io.BytesIO(img_bytes), width=400, height=300))
            story.append(Spacer(1, 12))
 
        # Metrics table
        tbl_data = [["Metric", "Value"]]
        if analysis.get("revenue_figures"):
            tbl_data.append(["Revenue (Latest)", analysis["revenue_figures"][0]])
        if analysis.get("profit_figures"):
            tbl_data.append(["Profit (Latest)", analysis["profit_figures"][0]])
        if len(tbl_data) > 1:
            tbl = Table(tbl_data)
            tbl.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.grey),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.black)
            ]))
            story.append(tbl)
 
        doc.build(story)
        return fname
    except Exception as exc:
        return f"Error generating PDF: {exc}"
 
# --------------------------------------------------------------------------- #
# 3.  Router Schema                                                           #
# --------------------------------------------------------------------------- #
 
class RouterModel(TypedDict):
    next: Literal["DocumentQuery", "ChartGeneration", "ReportGeneration", "FINISH"]
    reasoning: str
 
# --------------------------------------------------------------------------- #
# 4.  Individual Agents                                                       #
# --------------------------------------------------------------------------- #
 
class DocumentQueryAgent:
    def __init__(self, llm: ChatGroq, vector_store: VectorStore):
        self.llm = llm
        self.vector_store = vector_store
        self.tools = [retrieve_financial_documents, analyze_financial_data]
 
    def create_agent(self):
        sys_prompt = (
            "You are a specialist in retrieving Airtel's financial documents. "
            "1. Run `retrieve_financial_documents` to fetch context. "
            "2. Run `analyze_financial_data` on that context."
        )
        return ChatPromptTemplate.from_messages([
            ("system", sys_prompt),
            ("human", "{input}")
        ]) | self.llm
 
 
class ChartGenerationAgent:
    def __init__(self, llm: ChatGroq):
        self.llm = llm
        self.tools = [generate_chart_data, create_financial_chart]
 
    def create_agent(self):
        sys_prompt = (
            "You create clear business charts from financial analysis."
        )
        return ChatPromptTemplate.from_messages([
            ("system", sys_prompt),
            ("human", "{input}")
        ]) | self.llm
 
 
class ReportGenerationAgent:
    def __init__(self, llm: ChatGroq):
        self.llm = llm
        self.tools = [generate_financial_report]
 
    def create_agent(self):
        sys_prompt = (
            "You compile comprehensive PDF reports for stakeholders."
        )
        return ChatPromptTemplate.from_messages([
            ("system", sys_prompt),
            ("human", "{input}")
        ]) | self.llm
 
 
class SupervisorAgent:
    """
    Decides the next step based on current state.
    """
    def __init__(self, llm: ChatGroq):
        self.llm = llm
 
    def create_supervisor_chain(self):
        system_prompt = (
            "You supervise a 3-agent financial RAG workflow. "
            "Return JSON with keys 'next' and 'reasoning'."
        )
 
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "Query: {query}\nState: {state}")
        ])
 
        parser = JsonOutputParser(pydantic_object=RouterModel)
        return prompt | self.llm | parser
 
# --------------------------------------------------------------------------- #
# 5.  Multi-Agent Orchestrator                                                #
# --------------------------------------------------------------------------- #
 
class MultiAgentFinancialRAG:
    def __init__(self, vector_store: VectorStore, llm_manager: LLMManager):
        # Select provider
        if llm_manager.current_provider != "groq":
            raise ValueError("Unsupported LLM provider")
 
        self.llm = ChatGroq(
            model=llm_manager.current_model,
            api_key=Config.GROQ_API_KEY,
            temperature=0.1
        )
        self.vector_store = vector_store
 
        # Agents
        self.doc_agent   = DocumentQueryAgent(self.llm, vector_store)
        self.chart_agent = ChartGenerationAgent(self.llm)
        self.report_agent= ReportGenerationAgent(self.llm)
        self.supervisor  = SupervisorAgent(self.llm)
 
        self.workflow = self._create_workflow()
 
    # --------------------------------------------------------------------- #
    #  Workflow Graph                                                       #
    # --------------------------------------------------------------------- #
    def _create_workflow(self) -> StateGraph:
        sg = StateGraph(AgentState)
 
        # ---------- Supervisor Node ----------
        def supervisor_node(state: AgentState):
            chain = self.supervisor.create_supervisor_chain()
            decision: RouterModel = chain.invoke({
                "query": state["query"],
                "state": self._format_state(state)
            })
            state["supervisor_decision"] = decision["reasoning"]
            state["next_agent"] = decision["next"]
            return state
 
        # ---------- Document Query Node ----------
        def document_query_node(state: AgentState):
            try:
                doc_out = retrieve_financial_documents.invoke({
                    "query": state["query"],
                    "vector_store": self.vector_store
                })
                context = doc_out.get("context", "")
                analysis = analyze_financial_data.invoke({
                    "context": context,
                    "query": state["query"]
                })
                state["documents"] = doc_out.get("documents", [])
                state["metadata"]["analysis"] = analysis
                return state
            except Exception as exc:
                state["error"] = f"DocQuery error: {exc}"
                return state
 
        # ---------- Chart Generation Node ----------
        def chart_generation_node(state: AgentState):
            try:
                analysis = state["metadata"].get("analysis", {})
                cdata = generate_chart_data.invoke({
                    "analysis": analysis,
                    "chart_type": "auto"
                })
                cimg  = create_financial_chart.invoke({"chart_data": cdata})
                state["chart_data"]  = cdata
                state["chart_image"] = cimg
                return state
            except Exception as exc:
                state["error"] = f"ChartGen error: {exc}"
                return state
 
        # ---------- Report Generation Node ----------
        def report_generation_node(state: AgentState):
            try:
                analysis = state["metadata"].get("analysis", {})
                pdf = generate_financial_report.invoke({
                    "analysis": analysis,
                    "chart_base64": state["chart_image"],
                    "query": state["query"]
                })
                state["report_path"] = pdf
                return state
            except Exception as exc:
                state["error"] = f"ReportGen error: {exc}"
                return state
 
        # ---------- Final Response Node ----------
        def final_response_node(state: AgentState):
            try:
                summary_prompt = ChatPromptTemplate.from_messages([
                    ("system", "Write a concise business summary."),
                    ("human", "Query: {query}\nMetrics: {metrics}\n")
                ])
                analysis = state["metadata"].get("analysis", {})
                resp = (summary_prompt | self.llm).invoke({
                    "query": state["query"],
                    "metrics": json.dumps(analysis)
                })
                state["final_response"] = resp.content if hasattr(resp, "content") else str(resp)
                return state
            except Exception as exc:
                state["error"] = f"FinalResp error: {exc}"
                return state
 
        # ---- register nodes
        sg.add_node("supervisor", supervisor_node)
        sg.add_node("document_query", document_query_node)
        sg.add_node("chart_generation", chart_generation_node)
        sg.add_node("report_generation", report_generation_node)
        sg.add_node("final_response", final_response_node)
        sg.add_node("finish", lambda s: s)
 
        # ---- edges
        sg.add_edge(START, "supervisor")
 
        def choose_next(state: AgentState):
            nxt = state.get("next_agent", "FINISH")
            return {
                "DocumentQuery": "document_query",
                "ChartGeneration": "chart_generation",
                "ReportGeneration": "report_generation",
                "FINISH": "final_response"
            }.get(nxt, "final_response")
 
        sg.add_conditional_edges("supervisor", choose_next)
        sg.add_edge("document_query", "supervisor")
        sg.add_edge("chart_generation", "supervisor")
        sg.add_edge("report_generation", "supervisor")
        sg.add_edge("final_response", END)
 
        # checkpoint memory
        return sg.compile(checkpointer=MemorySaver())
 
    # --------------------------------------------------------------------- #
    #  Helpers                                                               #
    # --------------------------------------------------------------------- #
    @staticmethod
    def _format_state(state: AgentState) -> str:
        parts = []
        if state["documents"]:
            parts.append("documents retrieved")
        if state["metadata"].get("analysis"):
            parts.append("analysis ready")
        if state.get("chart_data"):
            parts.append("chart built")
        if state.get("report_path"):
            parts.append("report built")
        return ", ".join(parts) or "none yet"
 
    # --------------------------------------------------------------------- #
    #  Public method                                                         #
    # --------------------------------------------------------------------- #
    async def process_query(self, query: str) -> Dict[str, Any]:
        initial = AgentState(query=query)
        cfg = {"configurable": {"thread_id": str(uuid.uuid4())}}
        async for output in self.workflow.astream(initial, config=cfg):
            pass  # we want the final emission
        return output  # type: ignore
 
 
# --------------------------------------------------------------------------- #
# 6.  Streamlit Helper                                                        #
# --------------------------------------------------------------------------- #
 
def integrate_with_streamlit(vector_store: VectorStore, llm_manager: LLMManager):
    import streamlit as st
    if not hasattr(st.session_state, "multi_agent_system"):
        st.session_state.multi_agent_system = MultiAgentFinancialRAG(vector_store, llm_manager)
    return st.session_state.multi_agent_system
