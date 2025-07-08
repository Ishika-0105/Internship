import streamlit as st
import os
from config import Config
from document_processor import DocumentProcessor
from vector_store import VectorStore
from chat_memory import ChatMemory
from llm_manager import LLMManager
from evaluation_metrics import TriadEvaluationMetrics, TriadMetricsDisplay
from multi_agent_system import *
import time
import asyncio
from typing import Dict, Any
import pandas as pd
import base64
import plotly.graph_objects as go
import json
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="📊 Advanced Financial RAG Assistant with Multi-Agent System",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
def initialize_session_state():
    """Initialize all necessary session state variables"""
    if 'chat_memory' not in st.session_state:
        st.session_state.chat_memory = ChatMemory()
    
    if 'processing_question' not in st.session_state:
        st.session_state.processing_question = False
    
    if 'documents_loaded' not in st.session_state:
        st.session_state.documents_loaded = False
    
    if 'vector_store' not in st.session_state:
        st.session_state.vector_store = None
    
    if 'multi_agent_system' not in st.session_state:
        st.session_state.multi_agent_system = None
    
    if 'agent_mode' not in st.session_state:
        st.session_state.agent_mode = 'single'
    
    if 'search_mode' not in st.session_state:
        st.session_state.search_mode = 'hybrid'
    
    if 'hybrid_alpha' not in st.session_state:
        st.session_state.hybrid_alpha = Config.HYBRID_SEARCH_ALPHA
    
    if 'k_docs' not in st.session_state:
        st.session_state.k_docs = 5
    
    if 'show_evaluation' not in st.session_state:
        st.session_state.show_evaluation = False
    
    if 'last_evaluation' not in st.session_state:
        st.session_state.last_evaluation = None
    
    if 'last_multi_agent_result' not in st.session_state:
        st.session_state.last_multi_agent_result = None
    
    if 'current_chart' not in st.session_state:
        st.session_state.current_chart = None
    
    if 'current_report' not in st.session_state:
        st.session_state.current_report = None

# Initialize session state
initialize_session_state()

# Enhanced CSS for professional appearance (including multi-agent styles)
st.markdown(f"""
<style>
    .stApp {{
        background: linear-gradient(135deg, {Config.COLORS['background']} 0%, #F8F9FA 100%);
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }}
    
    .main-header {{
        background: linear-gradient(135deg, {Config.COLORS['gradient_start']}, {Config.COLORS['gradient_end']});
        color: white;
        text-align: center;
        padding: 2rem 0;
        border-radius: 15px;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(178, 34, 34, 0.3);
    }}
    
    .main-header h1 {{
        margin: 0;
        font-size: 2.5rem;
        font-weight: 700;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }}
    
    .main-header p {{
        margin: 0.5rem 0 0 0;
        font-size: 1.2rem;
        opacity: 0.9;
    }}
    
    .chat-message {{
        padding: 1.5rem;
        margin: 1rem 0;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        transition: transform 0.2s ease;
    }}
    
    .chat-message:hover {{
        transform: translateY(-2px);
    }}
    
    .user-message {{
        background: linear-gradient(135deg, #FFF 0%, #F8F9FA 100%);
        border-left: 4px solid {Config.COLORS['primary']};
        margin-left: 2rem;
    }}
    
    .assistant-message {{
        background: linear-gradient(135deg, #FFF 0%, #FAFBFC 100%);
        border-left: 4px solid {Config.COLORS['accent']};
        margin-right: 2rem;
    }}
    
    .agent-message {{
        background: linear-gradient(135deg, #E8F5E8 0%, #F0F8F0 100%);
        border-left: 4px solid #4CAF50;
        margin-right: 2rem;
    }}
    
    .source-info {{
        font-size: 0.85rem;
        color: {Config.COLORS['text_secondary']};
        font-style: italic;
        margin-top: 1rem;
        padding: 0.5rem;
        background-color: {Config.COLORS['background']};
        border-radius: 8px;
        border: 1px solid {Config.COLORS['border']};
    }}
    
    .stats-container {{
        background: linear-gradient(135deg, {Config.COLORS['gradient_start']}, {Config.COLORS['gradient_end']});
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 6px 20px rgba(178, 34, 34, 0.3);
    }}
    
    .stats-container h4 {{
        margin: 0 0 1rem 0;
        font-size: 1.2rem;
        font-weight: 600;
    }}
    
    .stats-container p {{
        margin: 0.5rem 0;
        font-size: 0.95rem;
    }}
    
    .agent-status {{
        background: linear-gradient(135deg, #4CAF50, #45A049);
        color: white;
        padding: 1rem;
        border-radius: 12px;
        margin: 0.5rem 0;
        box-shadow: 0 4px 15px rgba(76, 175, 80, 0.3);
    }}
    
    .agent-status h4 {{
        margin: 0 0 0.5rem 0;
        font-size: 1.1rem;
    }}
    
    .agent-status p {{
        margin: 0.2rem 0;
        font-size: 0.9rem;
    }}
    
    .processing-steps {{
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        border-left: 4px solid #2196F3;
    }}
    
    .step-item {{
        padding: 0.8rem;
        margin: 0.5rem 0;
        border-radius: 8px;
        border-left: 3px solid #2196F3;
        background: #F8F9FA;
    }}
    
    .step-completed {{
        background: #E8F5E8;
        border-left-color: #4CAF50;
    }}
    
    .step-current {{
        background: #E3F2FD;
        border-left-color: #2196F3;
        animation: pulse 2s infinite;
    }}
    
    @keyframes pulse {{
        0% {{ opacity: 1; }}
        50% {{ opacity: 0.7; }}
        100% {{ opacity: 1; }}
    }}
    
    .chart-container {{
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }}
    
    .report-download {{
        background: linear-gradient(135deg, #FF6B6B, #FF8E8E);
        color: white;
        padding: 1rem;
        border-radius: 12px;
        margin: 1rem 0;
        text-align: center;
        box-shadow: 0 4px 15px rgba(255, 107, 107, 0.3);
    }}
    
    .stButton > button {{
        background: linear-gradient(135deg, {Config.COLORS['primary']}, {Config.COLORS['secondary']});
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.5rem 1rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(178, 34, 34, 0.3);
    }}
    
    .stButton > button:hover {{
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(178, 34, 34, 0.4);
    }}
    
    .sidebar .stSelectbox > div > div {{
        background-color: white;
        border: 2px solid {Config.COLORS['border']};
        border-radius: 8px;
    }}
    
    .welcome-card {{
        background: white;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        margin: 1rem 0;
        border-top: 4px solid {Config.COLORS['primary']};
    }}
    
    .feature-grid {{
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
        gap: 1rem;
        margin: 1rem 0;
    }}
    
    .feature-card {{
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.08);
        border-left: 4px solid {Config.COLORS['accent']};
        transition: transform 0.2s ease;
    }}
    
    .feature-card:hover {{
        transform: translateY(-3px);
    }}
    
    .processing-indicator {{
        display: flex;
        align-items: center;
        padding: 1rem;
        background: linear-gradient(135deg, #E3F2FD, #BBDEFB);
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 4px solid #2196F3;
    }}
    
    .evaluation-panel {{
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 6px 25px rgba(0,0,0,0.1);
        border-top: 4px solid {Config.COLORS['success']};
    }}
    
    .search-mode-info {{
        background: rgba(255,255,255,0.1);
        padding: 0.5rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        font-size: 0.85rem;
    }}
    
    .mode-toggle {{
        background: white;
        padding: 1rem;
        border-radius: 12px;
        margin: 1rem 0;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        border-left: 4px solid #9C27B0;
    }}
</style>
""", unsafe_allow_html=True)

# Initialize session state
def initialize_session_state():
    """Initialize all session state variables"""
    if 'vector_store' not in st.session_state:
        st.session_state.vector_store = VectorStore()
    if 'chat_memory' not in st.session_state:
        st.session_state.chat_memory = ChatMemory()
    if 'llm_manager' not in st.session_state:
        st.session_state.llm_manager = LLMManager()
    if 'document_processor' not in st.session_state:
        st.session_state.document_processor = DocumentProcessor()
    if 'search_mode' not in st.session_state:
        st.session_state.search_mode = 'hybrid' 
    if 'evaluation_metrics' not in st.session_state:
        st.session_state.evaluation_metrics = TriadEvaluationMetrics()
    if 'documents_loaded' not in st.session_state:
        st.session_state.documents_loaded = False
    if 'show_evaluation' not in st.session_state:
        st.session_state.show_evaluation = Config.EVALUATION_ENABLED
    if 'last_evaluation' not in st.session_state:
        st.session_state.last_evaluation = None
    if 'last_retrieved_docs' not in st.session_state:
        st.session_state.last_retrieved_docs = []
    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []
    if 'processing_question' not in st.session_state:
        st.session_state.processing_question = False
    if 'agent_mode' not in st.session_state:
        st.session_state.agent_mode = 'single'  # 'single' or 'multi'
    if 'multi_agent_system' not in st.session_state:
        st.session_state.multi_agent_system = None
    if 'agent_progress' not in st.session_state:
        st.session_state.agent_progress = []
    if 'current_chart' not in st.session_state:
        st.session_state.current_chart = None
    if 'current_report' not in st.session_state:
        st.session_state.current_report = None

def display_model_info():
    """Display current model information"""
    model_info = st.session_state.llm_manager.get_model_info()
    st.markdown(f"""
    <div class="stats-container">
        <h4>🎯 Current Model</h4>
        <p><strong>Provider:</strong> {model_info['provider'].title()}</p>
        <p><strong>Model:</strong> {model_info['model']}</p>
        <p><strong>Status:</strong> {'🟢 Ready' if model_info['status'] else '🔴 Not Available'}</p>
    </div>
    """, unsafe_allow_html=True)

    # Enhanced stats display with HNSW information
    stats = st.session_state.vector_store.get_stats()
    
    st.markdown(f"""
    <div class="stats-container">
        <h4>📈 Knowledge Base Status</h4>
        <p><strong>Documents:</strong> {stats['total_documents']}</p>
        <p><strong>Embeddings:</strong> {stats['index_size']}</p>
        <p><strong>Embedding Shape:</strong> {stats['embeddings_shape']}</p>
        <p><strong>Index Type:</strong> {stats.get('index_type', 'HNSW')}</p>
        <p><strong>HNSW M:</strong> {stats.get('hnsw_m', 'N/A')}</p>
        <p><strong>HNSW efSearch:</strong> {stats.get('hnsw_ef_search', 'N/A')}</p>
    </div>
    """, unsafe_allow_html=True)

    # Multi-agent system status
    if st.session_state.agent_mode == 'multi':
        st.markdown(f"""
        <div class="agent-status">
            <h4>🤖 Multi-Agent System</h4>
            <p><strong>Status:</strong> {'🟢 Active' if st.session_state.multi_agent_system else '🔴 Inactive'}</p>
            <p><strong>Agents:</strong> DocumentQuery, ChartGeneration, ReportGeneration</p>
            <p><strong>Workflow:</strong> LangGraph-based coordination</p>
        </div>
        """, unsafe_allow_html=True)

def display_agent_mode_selector():
    """Display agent mode selector"""
    st.markdown("### 🤖 AI Mode Selection")
    
    mode_options = {
        'single': '🎯 Single Agent (Fast & Direct)',
        'multi': '🤖 Multi-Agent System (Comprehensive Analysis)'
    }
    
    selected_mode = st.radio(
        "Choose AI processing mode:",
        options=list(mode_options.keys()),
        format_func=lambda x: mode_options[x],
        index=0 if st.session_state.agent_mode == 'single' else 1,
        help="Single Agent: Quick responses. Multi-Agent: Comprehensive analysis with charts and reports.",
        key="agent_mode_radio"
    )
    
    if selected_mode != st.session_state.agent_mode:
        st.session_state.agent_mode = selected_mode
        
        # Initialize multi-agent system if selected
        if selected_mode == 'multi' and st.session_state.documents_loaded:
            initialize_multi_agent_system()
        
        st.rerun()
    
def initialize_multi_agent_system():
    """Initialize the multi-agent system"""
    try:
        if st.session_state.documents_loaded:
            st.session_state.multi_agent_system = integrate_with_streamlit(
                st.session_state.vector_store, 
                st.session_state.llm_manager
            )
            st.success("✅ Multi-Agent System initialized!")
        else:
            st.warning("⚠️ Please load documents first to enable multi-agent mode.")
    except Exception as e:
        st.error(f"❌ Failed to initialize multi-agent system: {str(e)}")

def process_documents(uploaded_files, website_url):
    """Process uploaded documents and website content with enhanced feedback"""
    all_documents = []
    
    # Create processing indicator
    progress_container = st.container()
    
    with progress_container:
        st.markdown('<div class="processing-indicator">📚 Processing financial documents with HNSW indexing...</div>', unsafe_allow_html=True)
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        total_steps = len(uploaded_files or []) + (1 if website_url else 0)
        current_step = 0
        
        # Process uploaded files
        if uploaded_files:
            for i, file in enumerate(uploaded_files):
                status_text.text(f"Processing {file.name}...")
                
                file_extension = os.path.splitext(file.name)[1].lower()
                
                if file_extension == '.pdf':
                    docs = st.session_state.document_processor.process_pdf(file)
                elif file_extension == '.csv':
                    docs = st.session_state.document_processor.process_csv(file) 
                elif file_extension in ['.xlsx', '.xls']:
                    docs = st.session_state.document_processor.process_xlsx(file)
                elif file_extension == '.json':
                    docs = st.session_state.document_processor.process_json(file)
                else:
                    st.warning(f"⚠️ Unsupported file type: {file.name}")
                    continue
                
                if docs:
                    all_documents.extend(docs)
                    st.success(f"✅ Processed: {file.name} ({len(docs)} chunks)")
                
                current_step += 1
                progress_bar.progress(current_step / total_steps)
        
        # Process website URL
        if website_url:
            status_text.text("Processing website content...")
            docs = st.session_state.document_processor.process_website(website_url)
            if docs:
                all_documents.extend(docs)
                st.success(f"✅ Processed website: {len(docs)} chunks")
            
            current_step += 1
            progress_bar.progress(current_step / total_steps)
        
        # Add documents to vector store
        if all_documents:
            status_text.text("Building knowledge base...")
            success = st.session_state.vector_store.add_documents(all_documents)
            if success:
                st.session_state.documents_loaded = True
                
                # Initialize multi-agent system if in multi-agent mode
                if st.session_state.agent_mode == 'multi':
                    initialize_multi_agent_system()
                
                # Generate and display document summary
                sample_content = " ".join([doc.page_content[:300] for doc in all_documents[:5]])
                summary = st.session_state.llm_manager.generate_response(
                    query="Provide a comprehensive summary of these financial documents including company name, reporting period, key financial highlights, and main business segments.",
                    context=sample_content
                )
                
                progress_bar.progress(1.0)
                status_text.text("✅ Processing complete!")
                
                st.markdown("### 📋 Document Summary")
                st.markdown(f'<div class="welcome-card">{summary}</div>', unsafe_allow_html=True)
                
                time.sleep(1)
                st.rerun()
        else:
            st.error("❌ No documents were successfully processed.")

def display_welcome_message():
    """Display welcome message when no documents are loaded"""
    st.markdown("""
    ### 👋 Welcome to Advanced Financial RAG Assistant!
    
    I'm your AI-powered financial analysis assistant with both single-agent and multi-agent capabilities. Here's how to get started:
    
    **📤 Upload Your Documents:**
    - 📄 PDF files (Annual/Quarterly reports)
    - 📊 CSV files (Financial data)
    - 📈 Excel files (Spreadsheets)
    - 🌐 Website URLs (Financial pages)
    
    **🤖 Choose Your AI Mode:**
    - **Single Agent:** Quick Q&A and document search
    - **Multi-Agent:** Comprehensive analysis with automatic chart generation and reports
    
    **💬 What You Can Ask:**
    - "*What were the key financial highlights this quarter?*"
    - "*Show me the revenue breakdown by segment*"
    - "*What are the main risks mentioned in the report?*"
    - "*Compare this year's performance with last year*"
    - "*What is the company's debt-to-equity ratio?*"
    - "*Generate a comprehensive financial report*" (Multi-Agent mode)
    
    **🎯 Multi-Agent Features:**
    - 🧠 **DocumentQuery Agent:** Intelligent document retrieval and analysis
    - 📊 **ChartGeneration Agent:** Automatic financial chart creation
    - 📄 **ReportGeneration Agent:** Comprehensive PDF report generation
    - 🎭 **Supervisor Agent:** Coordinates workflow and task delegation
    
    **Getting Started:**
    1. Upload your financial documents using the sidebar
    2. Click "Process Documents" to build your knowledge base
    3. Choose your preferred AI mode (Single or Multi-Agent)
    4. Start asking questions about your financial data!
    
    *Ready to dive into your financial analysis? Upload some documents to begin! 🚀*
    """)

def display_agent_progress(progress_steps):
    """Display multi-agent processing progress"""
    if progress_steps:
        st.markdown("""
        <div class="processing-steps">
            <h4>🤖 Multi-Agent Processing Steps</h4>
        </div>
        """, unsafe_allow_html=True)
        
        for i, step in enumerate(progress_steps):
            status_class = "step-completed" if step.get('completed', False) else ("step-current" if step.get('current', False) else "")
            status_icon = "✅" if step.get('completed', False) else ("⏳" if step.get('current', False) else "⏸️")
            
            st.markdown(f"""
            <div class="step-item {status_class}">
                <strong>{status_icon} {step['agent']}:</strong> {step['description']}
            </div>
            """, unsafe_allow_html=True)
def process_single_agent_question(user_input: str):
    """Process user question using single agent mode"""
    st.session_state.processing_question = True
    
    try:
        # Add user message to chat memory
        st.session_state.chat_memory.add_message('user', user_input)
        
        with st.spinner("🔍 Searching documents and generating response..."):
            # Get relevant documents
            relevant_docs = st.session_state.vector_store.get_relevant_documents(
                query=user_input,
                k=st.session_state.get('k_docs', 5),
                use_hybrid=True
            )
            
            st.session_state.last_retrieved_docs = relevant_docs
            
            # Generate response using LLM
            context = "\n\n".join([doc.page_content for doc in relevant_docs])
            
            prompt = f"""
            Based on the following financial documents, please provide a comprehensive answer to the user's question.
            
            Context from documents:
            {context}
            
            User Question: {user_input}
            
            Please provide a detailed, accurate response based on the available information.
            """
            
            response = st.session_state.llm_manager.generate_response(
                query=prompt,
                context=context
            )
            
            # Add sources information
            sources = list(set([doc.metadata.get('source', 'Unknown') for doc in relevant_docs]))
            
            # Add response to chat memory with sources
            st.session_state.chat_memory.add_message('assistant', response, sources=sources)
            
            # Evaluation (if enabled)
            if st.session_state.show_evaluation and relevant_docs:
                try:
                    evaluation = st.session_state.evaluation_metrics.evaluate_response(
                        answer=response,
                        query=user_input,
                        context=context,
                        retrieved_docs=[doc.page_content for doc in relevant_docs],
                        reference=context
                    )
                    st.session_state.last_evaluation = evaluation
                except Exception as e:
                    st.error(f"⚠️ Evaluation failed: {str(e)}")
                    st.session_state.last_evaluation = None
            
    except Exception as e:
        st.error(f"❌ Error processing question: {str(e)}")
        response = "I apologize, but I encountered an error while processing your question."
        st.session_state.chat_memory.add_message('assistant', response)
        st.session_state.last_evaluation = None
    
    finally:
        st.session_state.processing_question = False
        st.rerun()

async def run_multi_agent_query(user_input: str):
    """Run the multi-agent workflow"""
    try:
        # Create initial state using AgentState
        initial_state = AgentState(query=user_input).to_dict()
        debug_state(initial_state, "run_multi_agent_query initial_state")
        
        # Set up config
        config = {
            "configurable": {
                "thread_id": str(time.time()),
                "checkpoint_ns": int(time.time() * 1e9)
            }
        }
        
        # Run workflow
        final_result = None
        async for chunk in st.session_state.multi_agent_system.workflow.astream(
            initial_state, 
            config=config
        ):
            if chunk:
                debug_state(chunk, "run_multi_agent_query chunk")
                final_result = chunk
        
        if not final_result:
            return AgentState(
                query=user_input,
                error="No response from workflow",
                final_response="The system did not generate a response."
            ).to_dict()
            
        debug_state(final_result, "run_multi_agent_query final_result")
        return final_result
        
    except Exception as e:
        return AgentState(
            query=user_input,
            error=f"Workflow error: {str(e)}",
            final_response="An error occurred while processing your request."
        ).to_dict()

def process_multi_agent_question(user_input: str):
    """Process the user's question"""
    st.session_state.processing_question = True
    
    try:
        # Add user message
        st.session_state.chat_memory.add_message('user', user_input)
        
        if not st.session_state.multi_agent_system:
            st.error("Multi-agent system not initialized")
            return

        with st.spinner("Processing..."):
            # Run query
            result = asyncio.run(run_multi_agent_query(user_input))
            
            print(f"Final result: {result}")  # Debug print
            
            # Handle errors
            if result.get('error'):
                st.error(f"Error: {result['error']}")
                st.session_state.chat_memory.add_message(
                    'assistant',
                    f"Error: {result['error']}"
                )
                return
            
            # Process successful result
            response = result.get('final_response', 'No response generated')
            
            # Update state
            st.session_state.current_chart = result.get('chart_data')
            st.session_state.current_report = result.get('report_path')
            
            # Add response
            st.session_state.chat_memory.add_message(
                'assistant',
                response,
                sources=result.get('metadata', {}).get('sources', [])
            )
            
    except Exception as e:
        st.error(f"Error: {str(e)}")
        print(f"Process error: {str(e)}")  # Debug print
        traceback.print_exc()
        
        st.session_state.chat_memory.add_message(
            'assistant',
            "An error occurred while processing your request."
        )
    
    finally:
        st.session_state.processing_question = False
        st.rerun()

# 4. Add the missing display_suggested_questions function
def display_suggested_questions():
    """Display suggested questions for users"""
    if st.session_state.documents_loaded:
        st.markdown("### 💡 Suggested Questions")
        
        # Generate questions based on document content
        if st.session_state.last_retrieved_docs:
            # Use the last retrieved documents to generate relevant questions
            context = "\n\n".join([doc.page_content for doc in st.session_state.last_retrieved_docs[:3]])
            suggestions = [
                f"What are the key financial metrics in this document?",
                f"Show me the trends and patterns in the data",
                f"What risks and challenges are mentioned?",
                f"How is the financial performance compared to previous periods?",
                f"What are the main business segments discussed?",
                f"What strategic decisions were made based on this data?"
            ]
        else:
            suggestions = [
                "What are the key financial metrics for this period?",
                "Show me the revenue trends and growth rates",
                "What are the major risks and challenges mentioned?",
                "How is the company's cash flow position?",
                "What are the main business segments and their performance?",
                "What capital expenditures were made this quarter?"
            ]
        
        cols = st.columns(3)
        for i, suggestion in enumerate(suggestions):
            col = cols[i % 3]
            if col.button(suggestion, key=f"question_{i}"):
                st.session_state.processing_question = True
                st.session_state.processing_question_text = suggestion
                st.rerun()

def display_chart_result(chart_data):
    """Display generated chart"""
    if chart_data and not chart_data.get('error'):
        st.markdown("""
        <div class="chart-container">
            <h4>📊 Generated Financial Chart</h4>
        </div>
        """, unsafe_allow_html=True)
        
        try:
            # Create chart from data
            if chart_data.get('type') == 'line' and chart_data.get('data', {}).get('revenue'):
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=chart_data['data']['quarters'],
                    y=chart_data['data']['revenue'],
                    mode='lines+markers',
                    name='Revenue',
                    line=dict(color='#1f77b4', width=3)
                ))
                fig.update_layout(
                    title=chart_data.get('title', 'Financial Chart'),
                    xaxis_title="Quarter",
                    yaxis_title="Revenue (Crores)",
                    template="plotly_white"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            elif chart_data.get('type') == 'bar' and chart_data.get('data', {}).get('profit'):
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=chart_data['data'].get('quarters', [f'Period {i+1}' for i in range(len(chart_data['data']['profit']))]),
                    y=chart_data['data']['profit'],
                    name='Profit',
                    marker_color='#2ca02c'
                ))
                fig.update_layout(
                    title=chart_data.get('title', 'Financial Chart'),
                    xaxis_title="Period",
                    yaxis_title="Profit (Crores)",
                    template="plotly_white"
                )
                st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.error(f"❌ Error displaying chart: {str(e)}")

def display_report_download(report_path):
    """Display report download link"""
    if report_path and not report_path.startswith('Error'):
        st.markdown(f"""
        <div class="report-download">
            <h4>📄 Generated Report</h4>
            <p>Your comprehensive financial report has been generated!</p>
            <p><strong>Filename:</strong> {os.path.basename(report_path)}</p>
        </div>
        """, unsafe_allow_html=True)
        
        try:
            if os.path.exists(report_path):
                with open(report_path, 'rb') as file:
                    st.download_button(
                        label="📥 Download Report",
                        data=file.read(),
                        file_name=os.path.basename(report_path),
                        mime='application/pdf'
                    )
        except Exception as e:
            st.error(f"❌ Error preparing download: {str(e)}")

def display_chat_interface():
    """Display the main chat interface with multi-agent support"""
    st.markdown("### 💬 Chat with Your Financial Documents")
    
    # Get chat history from memory
    chat_history = st.session_state.chat_memory.get_chat_history()
    
    # Display conversation history
    for i, message in enumerate(chat_history):
        if message['role'] == 'user':
            st.markdown(f"""
            <div class="chat-message user-message">
                <strong>👤 You:</strong> {message['content']}
            </div>
            """, unsafe_allow_html=True)
        else:
            # Assistant message
            message_class = "agent-message" if st.session_state.agent_mode == 'multi' else "assistant-message"
            agent_label = "🤖 Multi-Agent System" if st.session_state.agent_mode == 'multi' else "🤖 Assistant"
            
            st.markdown(f"""
            <div class="chat-message {message_class}">
                <strong>{agent_label}:</strong><br>{message['content']}
                {f'<div class="source-info">📚 Sources: {", ".join(message.get("sources", []))}</div>' if message.get('sources') else ''}
            </div>
            """, unsafe_allow_html=True)
            
            # Show chart and report only in multi-agent mode
            if st.session_state.agent_mode == 'multi':
                # Show chart if available
                if i == len(chat_history) - 1 and st.session_state.current_chart:
                    display_chart_result(st.session_state.current_chart)
                
                # Show report download if available
                if i == len(chat_history) - 1 and st.session_state.current_report:
                    display_report_download(st.session_state.current_report)
            
            # Show evaluation metrics only for the most recent assistant response
            if (st.session_state.show_evaluation and st.session_state.last_evaluation and 
                i == len(chat_history) - 1 and message['role'] == 'assistant'):
                st.markdown('<div class="evaluation-panel">', unsafe_allow_html=True)
                TriadMetricsDisplay.display_metrics(
                    st.session_state.last_evaluation,
                    "Response Quality Evaluation"
                )
                st.markdown('</div>', unsafe_allow_html=True)
    
    # Show agent progress only in multi-agent mode
    if st.session_state.agent_mode == 'multi':
        if st.session_state.agent_progress:
            display_agent_progress(st.session_state.agent_progress)
        
        # Reset multi-agent specific state
        st.session_state.current_chart = None
        st.session_state.current_report = None

    # Chat input
    if not st.session_state.processing_question:
        user_input = st.chat_input("Ask me anything about your financial documents...")
        
        if user_input:
            if st.session_state.agent_mode == 'single':
                process_single_agent_question(user_input)
            else:
                process_multi_agent_question(user_input)
    else:
        user_input = st.session_state.processing_question_text
        if user_input:
            if st.session_state.agent_mode == 'single':
                process_single_agent_question(user_input)
            else:
                process_multi_agent_question(user_input)


def process_user_question(user_input: str):
    """Process user question using appropriate mode"""
    st.session_state.processing_question = True
    
    try:
        st.session_state.chat_memory.add_message('user', user_input)
        
        # Get relevant documents
        relevant_docs = st.session_state.vector_store.get_relevant_documents(
            query=user_input,
            k=st.session_state.get('k_docs', 5),
            use_hybrid=True
        )
        
        st.session_state.last_retrieved_docs = relevant_docs
        
        # Generate response using LLM
        context = "\n\n".join([doc.page_content for doc in relevant_docs])
        
        prompt = f"""
        Based on the following financial documents, please provide a comprehensive answer to the user's question.
        
        Context from documents:
        {context}
        
        User Question: {user_input}
        
        Please provide a detailed, accurate response based on the available information.
        """
        
        response = st.session_state.llm_manager.generate_response(
            query=prompt,
            context=context
        )
        
        # Add sources information
        sources = list(set([doc.metadata.get('source', 'Unknown') for doc in relevant_docs]))
        
        # Add response to chat memory with sources
        st.session_state.chat_memory.add_message('assistant', response, sources=sources)
        
        # Evaluation (if enabled)
        if st.session_state.show_evaluation and relevant_docs:
            try:
                evaluation = st.session_state.evaluation_metrics.evaluate_response(
                    answer=response,
                    query=user_input,
                    context=context,
                    retrieved_docs=[doc.page_content for doc in relevant_docs],
                    reference=context
                )
                st.session_state.last_evaluation = evaluation
            except Exception as e:
                st.error(f"⚠️ Evaluation failed: {str(e)}")
                st.session_state.last_evaluation = None
        
    except Exception as e:
        st.error(f"❌ Error processing question: {str(e)}")
        response = {str(e)}#"I apologize, but I encountered an error while processing your question."
        st.session_state.chat_memory.add_message('assistant', response)
        st.session_state.last_evaluation = None
    
    finally:
        st.session_state.processing_question = False
        st.rerun()


def display_control_buttons():
    """Display control buttons in sidebar"""
    st.markdown("### 🎛️ Controls")
    
    # Clear chat button
    if st.button("🗑️ Clear Chat", help="Clear conversation history"):
        st.session_state.chat_memory.clear_history()
        st.session_state.last_evaluation = None
        st.session_state.last_multi_agent_result = None
        st.success("✅ Chat history cleared!")
        st.rerun()
    
    # Reset system button
    if st.button("🔄 Reset System", help="Reset vector store and models"):
        for key in ['documents_loaded', 'vector_store', 'multi_agent_system']:
            if key in st.session_state:
                del st.session_state[key]
        st.success("✅ System reset!")
        st.rerun()
    
    # Download options
    if st.session_state.documents_loaded:
        st.markdown("### 💾 Downloads")
        
        # Custom JSON encoder to handle sets
        class SetEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, set):
                    return list(obj)
                return super().default(obj)

        # Download chat history
        if st.button("📥 Download Chat History"):
            chat_history = st.session_state.chat_memory.get_chat_history()
            if chat_history:
                chat_json = json.dumps(chat_history, indent=2, cls=SetEncoder)
                st.download_button(
                    label="📄 Download JSON",
                    data=chat_json,
                    file_name=f"chat_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
        
        # Download report if available
        if hasattr(st.session_state, 'last_multi_agent_result') and st.session_state.last_multi_agent_result:
            result = st.session_state.last_multi_agent_result
            if result.get('report_path') and not result.get('report_path', '').startswith('Error'):
                try:
                    with open(result['report_path'], 'rb') as f:
                        report_data = f.read()
                    
                    st.download_button(
                        label="📊 Download Report",
                        data=report_data,
                        file_name=result['report_path'],
                        mime="application/pdf"
                    )
                except:
                    pass

def display_evaluation_metrics():
    """Display evaluation metrics"""
    if st.session_state.last_evaluation:
        st.markdown("### 📊 Response Quality Metrics")
        
        eval_data = st.session_state.last_evaluation
    
        # Additional metrics in expandable section
        with st.expander("📈 Detailed Metrics"):
            for metric, value in eval_data.items():
                if isinstance(value, (int, float)):
                    st.write(f"**{metric.replace('_', ' ').title()}:** {value:.3f}")

def display_multi_agent_results():
    """Display multi-agent system results"""
    result = st.session_state.last_multi_agent_result
    
    if result and 'error' not in result:
        st.markdown("### 🤖 Multi-Agent Analysis")
        
        # Show chart if available
        if result.get('chart_image') and not result.get('chart_image', '').startswith('Error'):
            st.markdown("#### 📊 Generated Chart")
            try:
                chart_data = base64.b64decode(result['chart_image'])
                st.image(chart_data, caption="Financial Analysis Chart")
            except:
                st.info("Chart generated but couldn't display")
        
        # Show analysis details
        if result.get('analysis_response'):
            with st.expander("📋 Detailed Analysis"):
                st.write(result['analysis_response'])
        
        # Show supervisor decisions
        if result.get('supervisor_decision'):
            with st.expander("🎯 Agent Workflow"):
                st.write(f"**Supervisor Decision:** {result['supervisor_decision']}")
                st.write(f"**Next Agent:** {result.get('next_agent', 'Unknown')}")


# Add CSS styling
def add_custom_css():
    """Add custom CSS for better styling"""
    st.markdown("""
    <style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .search-mode-info {
        background: #f0f2f6;
        padding: 1rem;
        border-radius: 5px;
        margin: 0.5rem 0;
        border-left: 4px solid #667eea;
    }
    
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
    }
    
    .chat-message {
        margin: 1rem 0;
        padding: 1rem;
        border-radius: 8px;
    }
    
    .user-message {
        background: #e3f2fd;
        border-left: 4px solid #2196f3;
    }
    
    .assistant-message {
        background: #f1f8e9;
        border-left: 4px solid #4caf50;
    }
    </style>
    """, unsafe_allow_html=True)

# Updated main function that includes CSS
def main():
    """Main application function"""
    # Add custom CSS
    add_custom_css()
    
    # Initialize session state
    initialize_session_state()
    
    # Initialize vector store if not already initialized
    if not st.session_state.vector_store:
        st.session_state.vector_store = VectorStore()
    
    # Initialize multi-agent system if in multi-agent mode and documents are loaded
    if st.session_state.agent_mode == 'multi' and st.session_state.documents_loaded and not st.session_state.multi_agent_system:
        st.session_state.multi_agent_system = MultiAgentFinancialRAG(
            vector_store=st.session_state.vector_store,
            llm_manager=st.session_state.llm_manager
        )
    
    # Header
    st.markdown(f'''
    <div class="main-header">
        <h1>📊 Advanced Financial RAG Assistant with Multi-Agent Intelligence</h1>
        <p>AI-powered financial document analysis with specialized agents and advanced evaluation metrics</p>
    </div>
    ''', unsafe_allow_html=True)
    
    # Sidebar for document upload and management
    with st.sidebar:
        st.markdown("### 📁 Document Management")
        
        # Enhanced file upload section
        uploaded_files = st.file_uploader(
            "Upload Financial Documents",
            type=['pdf', 'csv', 'xlsx', 'xls', 'json'],
            accept_multiple_files=True,
            help="Upload PDF annual reports, CSV/Excel financial data, or JSON structured data"
        )
        
        # Website URL input
        website_url = st.text_input(
            "🌐 Financial Website URL:",
            placeholder="https://investor.company.com/annual-report",
            help="Enter URL of financial reports or investor relations pages"
        )
        
        # Search Configuration Section
        st.markdown("### 🔍 Search Configuration")
        
        # Search mode selection
        search_mode = st.selectbox(
            "Search Mode:",
            options=['hybrid', 'semantic'],
            index=['hybrid', 'semantic'].index(st.session_state.search_mode) if st.session_state.search_mode in ['hybrid', 'semantic'] else 0,
            help="Choose how to search through documents"
        )
        
        if search_mode != st.session_state.search_mode:
            st.session_state.search_mode = search_mode

        # Search mode info
        search_info = {
            'hybrid': "🔄 Combines semantic similarity with keyword matching using HNSW for best results",
            'semantic': "🧠 Pure semantic search using HNSW embeddings for fast and accurate retrieval"
        }
        
        st.markdown(f"""
        <div class="search-mode-info">
            {search_info[search_mode]}
        </div>
        """, unsafe_allow_html=True)
        
        # Search parameters
        with st.expander("⚙️ Search Parameters"):
            k_docs = st.slider("Documents to retrieve:", 1, 10, 5, help="Number of relevant documents to retrieve")
            if search_mode == 'hybrid':
                alpha = st.slider("Semantic vs Keyword weight:", 0.0, 1.0, Config.HYBRID_SEARCH_ALPHA, 0.1, 
                              help="0 = pure keyword, 1 = pure semantic")
                st.session_state.hybrid_alpha = alpha
            st.session_state.k_docs = k_docs
        
        # HNSW Parameters Section
        with st.expander("🔧 HNSW Parameters"):
            ef_search = st.slider("HNSW efSearch:", 10, 500, st.session_state.vector_store.hnsw_ef_search if hasattr(st.session_state.vector_store, 'hnsw_ef_search') else 100, 
                             help="Higher values = better accuracy, slower search")
            if st.button("Update HNSW Parameters"):
                st.session_state.vector_store.update_hnsw_parameters(ef_search=ef_search)
                st.success("✅ HNSW parameters updated!")
        
        # Evaluation settings
        st.markdown("### 📊 Evaluation Settings")
        enable_evaluation = st.checkbox(
            "Enable Response Evaluation",
            value=st.session_state.show_evaluation,
            help="Show detailed evaluation metrics for each response"
        )
        st.session_state.show_evaluation = enable_evaluation
        
        # Process documents button
        if st.button("🔄 Process Documents", type="primary"):
            if uploaded_files or website_url:
                process_documents(uploaded_files, website_url)
            else:
                st.warning("⚠️ Please upload files or enter a website URL first.")
        
        # Agent Mode Section
        display_agent_mode_selector()

        # Initialize vector store if not already initialized
        # Model selection section
        st.markdown("### 🤖 AI Model Configuration")
        
        # Get available providers
        available_providers = st.session_state.llm_manager.get_available_providers()
        
        # Provider selection
        enabled_providers = [provider for provider, available in available_providers.items() if available]
        
        if enabled_providers:
            selected_provider = st.selectbox(
                "LLM Provider:",
                options=enabled_providers,
                index=enabled_providers.index(st.session_state.llm_manager.current_provider) if st.session_state.llm_manager.current_provider in enabled_providers else 0,
                help="Select your preferred LLM provider"
            )
            
            # Model selection based on provider
            if selected_provider in Config.LLM_MODELS:
                available_models = list(Config.LLM_MODELS[selected_provider].keys())
                selected_model = st.selectbox(
                    "Model:",
                    options=available_models,
                    index=available_models.index(st.session_state.llm_manager.current_model) if st.session_state.llm_manager.current_model in available_models else 0,
                    help=f"Select a model from {selected_provider}"
                )
                
                # Update model if changed
                if (selected_provider != st.session_state.llm_manager.current_provider or 
                    selected_model != st.session_state.llm_manager.current_model):
                    st.session_state.llm_manager.set_model(selected_provider, selected_model)
                    st.success(f"✅ Switched to {selected_provider}: {selected_model}")
        
        # Display current model info
        display_model_info()
        
        # Document processing controls
        display_control_buttons()
        
        # Evaluation metrics toggle
        display_evaluation_metrics()

    # Main content area
    if not st.session_state.documents_loaded:
        display_welcome_message()
    else:
        display_chat_interface()
    
    # Add suggested questions at the bottom
    display_suggested_questions()

if __name__ == "__main__":
    main()
