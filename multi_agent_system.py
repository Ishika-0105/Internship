from __future__ import annotations
import uuid, io, base64, asyncio, re, json, os, time, datetime as _dt
from typing import Dict, List, Any, Optional, Literal, TypedDict
from dataclasses import dataclass, field

# LangChain / LangGraph
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END, MessagesState
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
    """Workflow state dictionary"""
    
    def __init__(self, query: str = "", **kwargs):
        super().__init__({
            "query": query,
            "documents": [],
            "analysis_response": "",
            "chart_data": None,
            "chart_image": None,
            "report_path": None,
            "supervisor_decision": "",
            "next_agent": "",
            "final_response": "",
            "metadata": {},
            "error": None,
            "messages": []
        })
        
        # Update with provided values
        for k, v in kwargs.items():
            if k in self:
                self[k] = v
    
    def __getitem__(self, key: str) -> Any:
        return self.get(key)
    
    def get(self, key: str, default: Any = None) -> Any:
        return super().get(key, default)
 
def debug_state(state: Dict, location: str) -> None:
    """Debug helper to print state information"""
    print(f"\n=== State at {location} ===")
    print(f"Type: {type(state)}")
    print(f"Keys: {state.keys()}")
    print(f"Query: {state.get('query')}")
    print(f"Error: {state.get('error')}")
    print("======================\n")

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
        """Create the workflow graph"""
        workflow = StateGraph(AgentState)

        def _ensure_state(state_dict: Dict[str, Any]) -> Dict[str, Any]:
            """Ensure state has all required fields"""
            base_state = {
                "query": "",
                "documents": [],
                "analysis_response": "",
                "chart_data": None,
                "chart_image": None,
                "report_path": None,
                "supervisor_decision": "",
                "next_agent": "",
                "final_response": "",
                "metadata": {},
                "error": None,
                "messages": []
            }
            return {**base_state, **state_dict}

        def _format_state(self, state: Dict) -> str:
            """Format state for supervisor decision making"""
            status = []
            
            # Check query
            if state.get("query"):
                status.append(f"query: {state['query']}")
            
            # Check documents
            if state.get("documents"):
                status.append("documents retrieved")
            
            # Check analysis
            if state.get("metadata", {}).get("analysis"):
                status.append("analysis ready")
            
            # Check chart
            if state.get("chart_data"):
                status.append("chart generated")
            
            # Check report
            if state.get("report_path"):
                status.append("report created")
            
            # Check errors
            if state.get("error"):
                status.append(f"error: {state['error']}")
        
            return "; ".join(status) or "initial state"

        def supervisor_node(state: Dict) -> Dict:
            """Supervisor node with safe state handling"""
            try:
                # Initialize state if empty
                if not state:
                    state = {
                        "query": "",
                        "documents": [],
                        "analysis_response": "",
                        "chart_data": None,
                        "chart_image": None,
                        "report_path": None,
                        "supervisor_decision": "",
                        "next_agent": "",
                        "final_response": "",
                        "metadata": {},
                        "error": None,
                        "messages": []
                    }
                
                # Get supervisor decision
                chain = self.supervisor.create_supervisor_chain()
                decision = chain.invoke({
                    "query": state.get("query", ""),
                    "state": self._format_state(state)
                })
                
                # Update state
                state.update({
                    "supervisor_decision": decision.get("reasoning", ""),
                    "next_agent": decision.get("next", "FINISH")
                })
                
                return state
                
            except Exception as e:
                return {
                    "query": state.get("query", ""),
                    "documents": [],
                    "analysis_response": "",
                    "chart_data": None,
                    "chart_image": None,
                    "report_path": None,
                    "supervisor_decision": "",
                    "next_agent": "FINISH",
                    "final_response": "",
                    "metadata": {},
                    "error": f"Supervisor error: {str(e)}",
                    "messages": []
                }

        def document_query_node(state: Dict) -> Dict:
            """Document retrieval and analysis node"""
            debug_state(state, "document_query_node entry")
            try:
                state = _ensure_state(state)
                query = state["query"]

                # Get documents
                doc_result = retrieve_financial_documents.invoke({
                    "query": query,
                    "vector_store": self.vector_store
                })

                if doc_result.get("error"):
                    return {
                        **state,
                        "error": doc_result["error"],
                        "next_agent": "FINISH"
                    }

                # Analyze documents
                analysis = analyze_financial_data.invoke({
                    "context": doc_result.get("context", ""),
                    "query": query
                })

                # Update state
                result = {
                    **state,
                    "documents": doc_result.get("documents", []),
                    "metadata": {
                        **state.get("metadata", {}),
                        "analysis": analysis,
                        "sources": doc_result.get("sources", [])
                    }
                }
                debug_state(result, "document_query_node exit")
                return result

            except Exception as e:
                error_result = {
                    **state,
                    "error": f"Document query error: {str(e)}",
                    "next_agent": "FINISH"
                }
                debug_state(error_result, "document_query_node error")
                return error_result

        def chart_generation_node(state: Dict) -> Dict:
            """Chart generation node"""
            debug_state(state, "chart_generation_node entry")
            try:
                state = _ensure_state(state)
                analysis = state.get("metadata", {}).get("analysis", {})

                # Generate chart data
                chart_data = generate_chart_data.invoke({
                    "analysis": analysis,
                    "chart_type": "auto"
                })

                if chart_data.get("error"):
                    return {
                        **state,
                        "error": chart_data["error"],
                        "next_agent": "FINISH"
                    }

                # Create chart image
                chart_image = create_financial_chart.invoke({
                    "chart_data": chart_data
                })

                # Update state
                result = {
                    **state,
                    "chart_data": chart_data,
                    "chart_image": chart_image
                }
                debug_state(result, "chart_generation_node exit")
                return result

            except Exception as e:
                error_result = {
                    **state,
                    "error": f"Chart generation error: {str(e)}",
                    "next_agent": "FINISH"
                }
                debug_state(error_result, "chart_generation_node error")
                return error_result

        def report_generation_node(state: Dict) -> Dict:
            """Report generation node"""
            debug_state(state, "report_generation_node entry")
            try:
                state = _ensure_state(state)
                analysis = state.get("metadata", {}).get("analysis", {})

                # Generate report
                report_path = generate_financial_report.invoke({
                    "analysis": analysis,
                    "chart_base64": state.get("chart_image", ""),
                    "query": state["query"]
                })

                # Update state
                result = {
                    **state,
                    "report_path": report_path
                }
                debug_state(result, "report_generation_node exit")
                return result

            except Exception as e:
                error_result = {
                    **state,
                    "error": f"Report generation error: {str(e)}",
                    "next_agent": "FINISH"
                }
                debug_state(error_result, "report_generation_node error")
                return error_result

        def final_response_node(state: Dict) -> Dict:
            """Final response node with safe state handling"""
            try:
                # Initialize state if empty
                if not state:
                    state = {
                        "query": "",
                        "documents": [],
                        "analysis_response": "",
                        "chart_data": None,
                        "chart_image": None,
                        "report_path": None,
                        "supervisor_decision": "",
                        "next_agent": "",
                        "final_response": "",
                        "metadata": {},
                        "error": None,
                        "messages": []
                    }
                
                # Handle existing errors
                if state.get("error"):
                    return {
                        **state,
                        "final_response": f"Error occurred: {state['error']}"
                    }
                
                # Generate summary
                summary_prompt = ChatPromptTemplate.from_messages([
                    ("system", "Write a concise business summary."),
                    ("human", "Query: {query}\nMetrics: {metrics}")
                ])
                
                analysis = state.get("metadata", {}).get("analysis", {})
                response = summary_prompt.invoke({
                    "query": state.get("query", "No query provided"),
                    "metrics": json.dumps(analysis)
                })
                
                # Update state
                state["final_response"] = response.content if hasattr(response, "content") else str(response)
                return state
                
            except Exception as e:
                return {
                    **state,
                    "error": f"Final response error: {str(e)}",
                    "final_response": "An error occurred while generating the response."
                }

        # Register nodes
        workflow.add_node("supervisor", supervisor_node)
        workflow.add_node("document_query", document_query_node)
        workflow.add_node("chart_generation", chart_generation_node)
        workflow.add_node("report_generation", report_generation_node)
        workflow.add_node("final_response", final_response_node)
        workflow.add_node("finish", lambda s: s)

        # Add edges
        workflow.add_edge(START, "supervisor")

        def route_next(state: Dict) -> str:
            """Determine next node based on state"""
            debug_state(state, "route_next entry")
            next_agent = state.get("next_agent", "FINISH")
            route = {
                "DocumentQuery": "document_query",
                "ChartGeneration": "chart_generation",
                "ReportGeneration": "report_generation",
                "FINISH": "final_response"
            }.get(next_agent, "final_response")
            print(f"Routing to: {route}")
            return route

        # Add edges
        workflow.add_conditional_edges(
            "supervisor",
            route_next
        )
        workflow.add_edge("document_query", "supervisor")
        workflow.add_edge("chart_generation", "supervisor")
        workflow.add_edge("report_generation", "supervisor")
        workflow.add_edge("final_response", END)

        return workflow.compile(checkpointer=MemorySaver())

    async def process_query(self, query: str) -> Dict[str, Any]:
        """Process a query through the workflow"""
        try:
            # Create initial state
            initial_state = {
                "query": query,
                "documents": [],
                "analysis_response": "",
                "chart_data": None,
                "chart_image": None,
                "report_path": None,
                "supervisor_decision": "",
                "next_agent": "",
                "final_response": "",
                "metadata": {},
                "error": None,
                "messages": []
            }
            debug_state(initial_state, "process_query initial_state")

            # Run workflow
            config = {
                "configurable": {
                    "thread_id": str(uuid.uuid4()),
                    "checkpoint_ns": int(time.time() * 1e9)
                }
            }

            final_result = None
            async for chunk in self.workflow.astream(initial_state, config=config):
                if chunk:
                    debug_state(chunk, "process_query chunk")
                    final_result = chunk

            if not final_result:
                return {
                    "query": query,
                    "error": "No response from workflow",
                    "final_response": "The system did not generate a response."
                }

            debug_state(final_result, "process_query final_result")
            return final_result

        except Exception as e:
            error_state = {
                "query": query,
                "error": f"Workflow error: {str(e)}",
                "final_response": "An error occurred while processing your request."
            }
            debug_state(error_state, "process_query error")
            return error_state

# --------------------------------------------------------------------------- #
# 6.  Streamlit Helper                                                        #
# --------------------------------------------------------------------------- #
 
def integrate_with_streamlit(vector_store: VectorStore, llm_manager: LLMManager):
    import streamlit as st
    if not hasattr(st.session_state, "multi_agent_system"):
        st.session_state.multi_agent_system = MultiAgentFinancialRAG(vector_store, llm_manager)
    return st.session_state.multi_agent_system
