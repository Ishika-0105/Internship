from dataclasses import dataclass, field
import asyncio
import json
import base64
import io
from typing import Literal, TypedDict
from langchain.schema import BaseMessage
from typing import Dict, List, Any, Optional, Union, Annotated
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT

# LangGraph imports
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_groq import ChatGroq
from vector_store import VectorStore
from llm_manager import LLMManager
from config import Config
import streamlit as st

# Agent State Definition
@dataclass
class AgentState(MessagesState):
    """State shared across all agents"""
    query: str = ""
    documents: List[Dict] = field(default_factory=list)
    analysis_response: str = ""
    chart_data: Optional[Dict] = None
    chart_image: Optional[str] = None
    report_path: Optional[str] = None
    supervisor_decision: str = ""
    next_agent: str = ""
    final_response: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    
    def __post_init__(self):
        super().__init__()
        self.update({
            'query': self.query,
            'documents': self.documents,
            'analysis_response': self.analysis_response,
            'chart_data': self.chart_data,
            'chart_image': self.chart_image,
            'report_path': self.report_path,
            'supervisor_decision': self.supervisor_decision,
            'next_agent': self.next_agent,
            'final_response': self.final_response,
            'metadata': self.metadata,
            'error': self.error
        })

    def get(self, key: str, default=None):
        """Helper method to maintain dict-like access"""
        return self[key] if key in self else default

# Financial Data Analysis Tools
@tool
def retrieve_financial_documents(query: str, vector_store: VectorStore) -> str:
    """Retrieve relevant financial documents for the query using HNSW search"""
    try:
        relevant_docs = vector_store.get_relevant_documents(query, k=5, use_hybrid=True)
        if relevant_docs:
            context = "\n\n".join([doc.page_content for doc in relevant_docs])
            sources = [doc['metadata'].get('source', 'Unknown') for doc in relevant_docs]
            return f"Retrieved Documents:\n{context}\n\nSources: {', '.join(sources)}"
        return "No relevant documents found."
    except Exception as e:
        return f"Error retrieving documents: {str(e)}"

@tool
def analyze_financial_data(context: str, query: str) -> Dict[str, Any]:
    """Analyze financial data and extract key metrics"""
    try:
        # Extract numerical data from context
        import re
        
        # Common financial patterns
        revenue_pattern = r'revenue[:\s]*(?:₹|rs\.?|inr)?\s*([0-9,]+\.?[0-9]*)\s*(?:cr|crore|million|billion)?'
        profit_pattern = r'profit[:\s]*(?:₹|rs\.?|inr)?\s*([0-9,]+\.?[0-9]*)\s*(?:cr|crore|million|billion)?'
        ebitda_pattern = r'ebitda[:\s]*(?:₹|rs\.?|inr)?\s*([0-9,]+\.?[0-9]*)\s*(?:cr|crore|million|billion)?'
        
        analysis = {
            'revenue_figures': re.findall(revenue_pattern, context.lower()),
            'profit_figures': re.findall(profit_pattern, context.lower()),
            'ebitda_figures': re.findall(ebitda_pattern, context.lower()),
            'key_metrics': [],
            'trends': [],
            'context_summary': context[:500] + "..." if len(context) > 500 else context
        }
        
        return analysis
    except Exception as e:
        return {"error": f"Analysis failed: {str(e)}"}

@tool
def generate_chart_data(analysis: Dict[str, Any], chart_type: str = "auto") -> Dict[str, Any]:
    """Generate appropriate chart data based on financial analysis"""
    try:
        chart_data = {
            'type': chart_type,
            'data': {},
            'config': {},
            'title': 'Financial Analysis Chart'
        }
        
        # Extract sample data for demonstration
        if analysis.get('revenue_figures'):
            revenue_data = [float(x.replace(',', '')) for x in analysis['revenue_figures'][:4]]
            chart_data['data']['revenue'] = revenue_data
            chart_data['data']['quarters'] = [f'Q{i+1}' for i in range(len(revenue_data))]
            chart_data['type'] = 'line'
            chart_data['title'] = 'Revenue Trend Analysis'
        
        if analysis.get('profit_figures'):
            profit_data = [float(x.replace(',', '')) for x in analysis['profit_figures'][:4]]
            chart_data['data']['profit'] = profit_data
            chart_data['type'] = 'bar'
            chart_data['title'] = 'Profit Analysis'
        
        return chart_data
    except Exception as e:
        return {"error": f"Chart generation failed: {str(e)}"}

@tool
def create_financial_chart(chart_data: Dict[str, Any]) -> str:
    """Create actual chart image using Plotly"""
    try:
        fig = None
        
        if chart_data['type'] == 'line' and 'revenue' in chart_data['data']:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=chart_data['data']['quarters'],
                y=chart_data['data']['revenue'],
                mode='lines+markers',
                name='Revenue',
                line=dict(color='#1f77b4', width=3)
            ))
            fig.update_layout(
                title=chart_data['title'],
                xaxis_title="Quarter",
                yaxis_title="Revenue (Crores)",
                template="plotly_white"
            )
        
        elif chart_data['type'] == 'bar' and 'profit' in chart_data['data']:
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=chart_data['data'].get('quarters', [f'Period {i+1}' for i in range(len(chart_data['data']['profit']))]),
                y=chart_data['data']['profit'],
                name='Profit',
                marker_color='#2ca02c'
            ))
            fig.update_layout(
                title=chart_data['title'],
                xaxis_title="Period",
                yaxis_title="Profit (Crores)",
                template="plotly_white"
            )
        
        else:
            # Default sample chart
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=['Q1', 'Q2', 'Q3', 'Q4'],
                y=[100, 120, 140, 160],
                name='Sample Data'
            ))
            fig.update_layout(
                title="Sample Financial Chart",
                template="plotly_white"
            )
        
        # Convert to image
        img_bytes = fig.to_image(format="png", width=800, height=600)
        img_base64 = base64.b64encode(img_bytes).decode()
        
        return img_base64
    except Exception as e:
        return f"Error creating chart: {str(e)}"

@tool
def generate_financial_report(analysis: Dict[str, Any], chart_base64: str, query: str) -> str:
    """Generate downloadable PDF report"""
    try:
        # Create PDF report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"financial_report_{timestamp}.pdf"
        
        doc = SimpleDocTemplate(filename, pagesize=A4)
        styles = getSampleStyleSheet()
        story = []
        
        # Title
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            spaceAfter=30,
            alignment=TA_CENTER
        )
        story.append(Paragraph("Financial Analysis Report", title_style))
        story.append(Spacer(1, 20))
        
        # Query section
        story.append(Paragraph(f"<b>Query:</b> {query}", styles['Normal']))
        story.append(Spacer(1, 12))
        
        # Analysis section
        story.append(Paragraph("<b>Analysis Summary:</b>", styles['Heading2']))
        story.append(Paragraph(analysis.get('context_summary', 'No summary available'), styles['Normal']))
        story.append(Spacer(1, 12))
        
        # Add chart if available
        if chart_base64 and not chart_base64.startswith('Error'):
            try:
                chart_data = base64.b64decode(chart_base64)
                chart_image = Image(io.BytesIO(chart_data), width=400, height=300)
                story.append(chart_image)
                story.append(Spacer(1, 12))
            except:
                pass
        
        # Key metrics table
        if analysis.get('revenue_figures') or analysis.get('profit_figures'):
            story.append(Paragraph("<b>Key Financial Metrics:</b>", styles['Heading2']))
            
            table_data = [['Metric', 'Value']]
            if analysis.get('revenue_figures'):
                table_data.append(['Revenue (Latest)', f"₹{analysis['revenue_figures'][0]} Cr"])
            if analysis.get('profit_figures'):
                table_data.append(['Profit (Latest)', f"₹{analysis['profit_figures'][0]} Cr"])
            
            table = Table(table_data)
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 14),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            story.append(table)
        
        # Build PDF
        doc.build(story)
        
        return filename
    except Exception as e:
        return f"Error generating report: {str(e)}"

# Individual Agent Definitions
class DocumentQueryAgent:
    """Agent responsible for retrieving relevant documents"""
    
    def __init__(self, llm: ChatGroq, vector_store: VectorStore):
        self.llm = llm
        self.vector_store = vector_store
        self.tools = [retrieve_financial_documents, analyze_financial_data]
        
    def create_agent(self):
        system_prompt = """You are a financial document retrieval specialist for Airtel.
        Your role is to:
        1. Retrieve relevant financial documents using HNSW search
        2. Analyze the retrieved content for key financial metrics
        3. Provide structured analysis that other agents can use
        
        Always be thorough and extract specific numerical data when available.
        Focus on Airtel's financial performance, market position, and key metrics."""
        
        return create_react_agent(
            self.llm,
            self.tools,
            messages_modifier=system_prompt
        )

class ChartGenerationAgent:
    """Agent responsible for creating financial charts"""
    
    def __init__(self, llm: ChatGroq):
        self.llm = llm
        self.tools = [generate_chart_data, create_financial_chart]
        
    def create_agent(self):
        system_prompt = """You are a financial data visualization specialist.
        Your role is to:
        1. Analyze financial data and determine the best chart type
        2. Generate appropriate chart data structures
        3. Create professional financial charts
        
        Choose chart types based on data:
        - Line charts for trends over time
        - Bar charts for comparisons
        - Pie charts for composition
        - Combo charts for multi-metric analysis
        
        Always create clear, professional charts suitable for business presentations."""
        
        return create_react_agent(
            self.llm,
            self.tools,
            messages_modifier=system_prompt
        )

class ReportGenerationAgent:
    """Agent responsible for generating downloadable reports"""
    
    def __init__(self, llm: ChatGroq):
        self.llm = llm
        self.tools = [generate_financial_report]
        
    def create_agent(self):
        system_prompt = """You are a financial reporting specialist.
        Your role is to:
        1. Generate comprehensive financial reports
        2. Include charts and analysis in professional format
        3. Create downloadable PDF reports
        
        Reports should be professional, comprehensive, and suitable for business stakeholders."""
        
        return create_react_agent(
            self.llm,
            self.tools,
            messages_modifier=system_prompt
        )

class SupervisorAgent:
    """Supervisor agent that coordinates all other agents"""
    
    def __init__(self, llm: ChatGroq):
        self.llm = llm
        
    def create_supervisor_chain(self):
        system_prompt = """You are a supervisor for a financial analysis team working on Airtel's financial documents.
        Your team consists of:
        - DocumentQuery: Retrieves and analyzes financial documents
        - ChartGeneration: Creates financial charts and visualizations  
        - ReportGeneration: Generates downloadable reports
    
        Given the user's request, decide which agent should act next or if the task is complete.
    
        Workflow:
        1. Start with DocumentQuery for document retrieval and analysis
        2. If charts are needed, use ChartGeneration
        3. If reports are requested, use ReportGeneration
        4. Use FINISH when the task is complete
    
        You must respond with a JSON object containing:
        - "next": one of ["DocumentQuery", "ChartGeneration", "ReportGeneration", "FINISH"]
        - "reasoning": explanation for your choice
        """
    
        # Remove the "next" variable from the prompt - only use "query" and "state"
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "Given the conversation history and current state, who should act next?\n\nQuery: {query}\nCurrent State: {state}")
        ])
    
        parser = JsonOutputParser(pydantic_object=Router)
        return prompt | self.llm | parser

# Define the Router output structure
class Router(TypedDict):
    """Worker to route to next. If no workers needed, route to FINISH."""
    next: Literal["DocumentQuery", "ChartGeneration", "ReportGeneration", "FINISH"]
    reasoning: str

# Main Multi-Agent System
class MultiAgentFinancialRAG:
    """Main class that orchestrates the multi-agent system"""
    
    def __init__(self, vector_store: VectorStore, llm_manager: LLMManager):
        self.vector_store = vector_store
        self.llm_manager = llm_manager
        
        # Initialize LLM based on available provider
        if llm_manager.current_provider == 'groq':
            self.llm = ChatGroq(
                model=llm_manager.current_model,
                api_key=Config.GROQ_API_KEY,
                temperature=0.1
            )
        else:
            raise ValueError(f"Unsupported LLM provider: {llm_manager.current_provider}")
        
        # Initialize agents
        self.doc_agent = DocumentQueryAgent(self.llm, self.vector_store)
        self.chart_agent = ChartGenerationAgent(self.llm)
        self.report_agent = ReportGenerationAgent(self.llm)
        self.supervisor = SupervisorAgent(self.llm)
        
        # Create the workflow graph
        self.workflow = self._create_workflow()

    def _format_state_for_supervisor(self, state: AgentState) -> str:
        """Format state information for supervisor decision making"""
        state_info = []
        
        # Use dictionary access, not attribute access
        if state.get("documents"):
            state_info.append(f"Documents retrieved: {len(state['documents'])} documents")
        if state.get("analysis_response"):
            state_info.append("Document analysis: Completed")
        if state.get("chart_data"):
            state_info.append("Chart generation: Completed")
        if state.get("report_path"):
            state_info.append("Report generation: Completed")
        if state.get("error"):
            state_info.append(f"Error occurred: {state['error']}")
            
        return "; ".join(state_info) if state_info else "Initial state - no actions completed"
        
    
    def _create_workflow(self) -> StateGraph:
        """Create the LangGraph workflow with proper checkpointer configuration"""
    
        # Create the workflow graph
        workflow = StateGraph(AgentState)
    
        # Define agent functions
        def supervisor_node(state: AgentState):
            """Supervisor decision node"""
            try:
                supervisor_chain = self.supervisor.create_supervisor_chain()
                decision = supervisor_chain.invoke({
                    "query": state.get("query", ""),
                    "state": self._format_state_for_supervisor(state)
                })

                state["supervisor_decision"] = decision.get("reasoning", "")
                state["next_agent"] = decision.get("next", "FINISH")
                return state
            except Exception as e:
                print(f"Supervisor error: {str(e)}")
                state["supervisor_decision"] = f"Error in supervision: {str(e)}"
                state["next_agent"] = "FINISH"
                return state

        def document_query_node(state: AgentState):
            """Document query and analysis node"""
            try:
                # Use dict access for state
                docs_result = retrieve_financial_documents.stream({
                    "query": state["query"],
                    "vector_store": self.vector_store
                })
        
                analysis = analyze_financial_data.invoke({
                    "context": docs_result,
                    "query": state["query"]
                })
        
                # Update state using dict assignment
                state["documents"] = docs_result.get("documents", [])
                state["analysis_response"] = analysis.get("response", "")
                if analysis.get("metadata"):
                    state["metadata"].update(analysis.get("metadata", {}))
                    print("Updated state['metadata']:", state['metadata'])
                
                else:
                    print("No metadata found in analysis result")
                return state
        
            except Exception as e:
                state["error"] = f"Document query failed: {str(e)}"
                state["analysis_response"] = f"Error in document analysis: {str(e)}"
                return state
        
        def chart_generation_node(state: AgentState):
            """Chart generation node"""
            try:
                analysis = state["metadata"].get("analysis", {})
        
                chart_data = generate_chart_data.invoke({
                    "analysis": analysis,
                    "chart_type": "auto"
                })

                chart_image = create_financial_chart.invoke({
                    "chart_data": chart_data
                })

                state["chart_data"] = chart_data
                state["chart_image"] = chart_image
                return state
            except Exception as e:
                state["error"] = f"Chart generation failed: {str(e)}"
                state["chart_image"] = f"Error creating chart: {str(e)}"
                return state

        def report_generation_node(state: AgentState):
            """Report generation node"""
            try:
                analysis = state["metadata"].get("analysis", {})

                report_path = generate_financial_report.invoke({
                    "analysis": analysis,
                    "chart_base64": state["chart_image"],
                    "query": state["query"]
                })

                state["report_path"] = report_path
                return state
            except Exception as e:
                state["error"] = f"Report generation failed: {str(e)}"
                state["report_path"] = f"Error creating report: {str(e)}"
                return state

        def final_response_node(state: AgentState):
            """Final response node"""
            try:
                # Get all relevant information
                analysis = state["metadata"].get("analysis", {})
        
                # Generate final response
                final_prompt = ChatPromptTemplate.from_messages([
                ("system", "Generate a comprehensive summary of the financial analysis results."),
                ("human", """
                Query: {query}
                Analysis: {analysis}
                Chart Data Available: {has_chart}
                Report Generated: {has_report}
            
                Provide a clear, concise summary of the results.
                """)
            ])
                final_chain = final_prompt | self.llm
                response = final_chain.invoke({
                    "query": state["query"],
                    "analysis": str(analysis),
                    "has_chart": bool(state["chart_data"]),
                    "has_report": bool(state["report_path"])
                })
        
                # Update state
                state["final_response"] = response.content if hasattr(response, 'content') else str(response)
                return state
            except Exception as e:
                state["error"] = f"Final response generation failed: {str(e)}"
                state["final_response"] = f"Error generating final response: {str(e)}"
                return state
        

        # Add nodes
        workflow.add_node("supervisor", supervisor_node)
        workflow.add_node("document_query", document_query_node)
        workflow.add_node("chart_generation", chart_generation_node)
        workflow.add_node("report_generation", report_generation_node)
        workflow.add_node("final_response", final_response_node)
        workflow.add_node("finish", lambda state: state)
        
        # Add edges
        workflow.add_edge(START, "supervisor")
        
        # Conditional edges from supervisor
        def route_supervisor(state: AgentState):
            next_agent = state.get("next_agent", "FINISH")
            if next_agent == "DocumentQuery":
                return "document_query"
            elif next_agent == "ChartGeneration":
                return "chart_generation"
            elif next_agent == "ReportGeneration":
                return "report_generation"
            else:
                return "final_response"
        
        workflow.add_conditional_edges("supervisor", route_supervisor)
        
        # Edges back to supervisor
        workflow.add_edge("document_query", "supervisor")
        workflow.add_edge("chart_generation", "supervisor")
        workflow.add_edge("report_generation", "supervisor")
        workflow.add_edge("final_response", END)
        
        # Set up checkpointer
        memory = MemorySaver()
        return workflow.compile(checkpointer=memory)
    
    async def process_query(self, query: str, session_id: str = "default"):
        """Process a query through the multi-agent system"""
        try:
            # Initialize state with proper checkpointer configuration
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
            "messages": []  # Required for MessagesState
        }
            
            # Run the workflow
            config = {"configurable": {"thread_id": str(uuid.uuid4())}, "max_iterations": 3}
            async for chunk in self.workflow.astream(initial_state, config=config):  
                if chunk:
                    return chunk
        
            return {"error": "No response received from workflow"}
        
        except Exception as e:
            return {"error": f"Workflow error: {str(e)}"}
    

# Integration function for Streamlit
def integrate_with_streamlit(vector_store: VectorStore, llm_manager: LLMManager):
    """Initialize the multi-agent system for Streamlit"""
    # Initialize the multi-agent system if not already initialized
    if not hasattr(st.session_state, 'multi_agent_system') or st.session_state.multi_agent_system is None:
        try:
            print("Initializing multi-agent system...")
            st.session_state.multi_agent_system = MultiAgentFinancialRAG(vector_store, llm_manager)
            print(st.session_state.multi_agent_system)
            return st.session_state.multi_agent_system
        except Exception as e:
            print(f"Error initializing multi-agent system: {str(e)}")
            return None
    
    return st.session_state.multi_agent_system
