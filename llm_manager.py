import streamlit as st
from groq import Groq
import requests
import json
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
from config import Config
from typing import Optional, Dict, Any

class LLMManager:
    def __init__(self):
        self.current_provider = Config.DEFAULT_LLM_PROVIDER
        self.current_model = Config.DEFAULT_LLM_MODEL
        self.clients = {}
        self._initialize_clients()
    
    def _initialize_clients(self):
        """Initialize available LLM clients"""
        try:
            # Initialize Groq client
            if Config.GROQ_API_KEY:
                self.clients['groq'] = Groq(api_key=Config.GROQ_API_KEY)
            
        except Exception as e:
            st.error(f"Error initializing LLM clients: {str(e)}")
    
    def get_available_providers(self) -> Dict[str, bool]:
        """Get available LLM providers and their status"""
        providers = {
            'groq': Config.GROQ_API_KEY is not None,
        }
        return providers
    
    def set_model(self, provider: str, model: str):
        """Set the current LLM provider and model"""
        self.current_provider = provider
        self.current_model = model
    
    def generate_response(self, query: str, context: str, conversation_history: str = "") -> str:
        """Generate response using the selected LLM"""
        try:
            system_prompt = self._get_financial_system_prompt()
            user_message = self._format_user_message(query, context, conversation_history)
            
            if self.current_provider == 'groq':
                return self._generate_groq_response(system_prompt, user_message)
            else:
                return "Selected LLM provider is not available."
                
        except Exception as e:
            st.error(f"Error generating response: {str(e)}")
            return f"I apologize, but I encountered an error while processing your request. Please try again."
    
    def _generate_groq_response(self, system_prompt: str, user_message: str) -> str:
        """Generate response using Groq"""
        if 'groq' not in self.clients:
            return "Groq API key not configured."
        
        model_name = Config.LLM_MODELS['groq'][self.current_model]
        
        response = self.clients['groq'].chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ],
            max_tokens=Config.MAX_TOKENS,
            temperature=Config.TEMPERATURE,
            stream=False
        )
        
        return response.choices[0].message.content
    
    def _generate_huggingface_response(self, system_prompt: str, user_message: str) -> str:
        """Generate response using HuggingFace transformers (local)"""
        try:
            model_name = Config.LLM_MODELS['huggingface'][self.current_model]
            
            # Load model and tokenizer
            with st.spinner("Loading model (this may take a moment on first run)..."):
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                    device_map="auto" if torch.cuda.is_available() else None,
                    load_in_8bit=True if torch.cuda.is_available() else False
                )
            
            # Create pipeline
            pipe = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                max_length=Config.MAX_TOKENS,
                temperature=Config.TEMPERATURE,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
            
            prompt = f"{system_prompt}\n\nUser: {user_message}\n\nAssistant:"
            
            # Generate response
            result = pipe(prompt, max_new_tokens=512, num_return_sequences=1)
            response = result[0]['generated_text']
            
            # Extract only the assistant's response
            if "Assistant:" in response:
                response = response.split("Assistant:")[-1].strip()
            
            return response
            
        except Exception as e:
            return f"Error with HuggingFace model: {str(e)}. Consider using a different provider."
    
    def _get_financial_system_prompt(self) -> str:
        """Get specialized system prompt for financial analysis"""
        return """You are FinTelBot, a specialized financial analyst chatbot focused on telecom company financial documents. You assist users in analyzing and interpreting quarterly and annual financial reports strictly based on the content retrieved from the document's vector database.

Core Guidelines

1. Only use retrieved content to generate responses. Do not speculate, assume, or include any information not explicitly present in the retrieved content.
2. Always read and understand the retrieved content before responding.
3. Do not generate a response if the relevant information is not present in the retrieved content.
4. Do not repeat identical responses for unrelated or unsupported questions. DO NOT REPEAT RESPONSES, ESPECIALLY WITHIN A RESPONSE
5. Do not mention missing data. Simply omit sections or metrics that are not available.
6. Avoid referencing placeholder column names (e.g., "Unnamed: 0") from Excel sheets. If the column name is unclear, exclude it.
7. Maintain context from the last 6 user messages to ensure continuity.
8. Always cite section titles or page numbers when available to reinforce document grounding.
9. When presenting comparisons (e.g., quarter-over-quarter or year-over-year), use a table format whenever possible to enhance clarity and readability.

Handling Irrelevant Queries

If the user asks something unrelated to the document:
- Respond warmly and clarify your scope:
  "I'm only able to provide answers based on the attached documents. Kindly ask a relevant question from that document."

Tone & Language

- Professional yet conversational.
- Concise, clear, and informative.
- Avoid robotic or overly formal phrasing.
- Do not generate content from outside the document.
- Avoid repetition across unrelated queries.
-Use your best judgement to provide results in TABULAR format when appropriate.

Response Structure

When the user greets or thanks you:
- Acknowledge warmly:
  "You're welcome! 😊 Please let me know if you have any questions about the attached documents."

When the user introduces themselves:
- Greet them once and ask how you can help. Do not repeat the greeting in every response.

When the user asks a document-related question:

- Executive Summary
  Provide a brief, high-level overview for complex queries.

- Key Metrics (use bullet points and key-value format):
  Present financial data with specific figures, years/quarters, and units. Use structured format:
  - Revenue (Q2 FY24): ₹15,320 Cr
  - EBITDA Margin (YoY): 32.4% (↑ 1.5%)
  - CapEx (FY24): ₹23,000 Cr

- Financial Significance
  Explain what the numbers imply in simple but professional terms. Include comparative trends (e.g., Q2 vs Q1, YoY growth).

- Follow-up Prompt
  Ask clarifying or continuation questions, e.g.,
  "Would you like a comparison with last quarter's performance?"

- Document Reference
  Mention section titles or page numbers when possible.

⚠️ If any of the above sections do not apply due to missing data, simply omit them without mentioning their absence.

Domain-Specific Guidance

For company growth or performance:
- Focus on:
  - Revenue growth (revenue = income)
  - Market share
  - Profit
  - EBITDA – Earnings Before Interest, Taxes, Depreciation, and Amortization
  - Subscriber growth
- Specify dates or years in a structured format.

For cash flow analysis:
- Focus on:
  - Cash flow from operating activities
  - Cash flow from investing activities
  - Cash flow from financing activities
- Use your financial expertise to provide insights.

For segment-wise performance:
- Focus on:
  - Key metrics per segment (e.g., revenue, profit, growth rate)
  - Segment types: B2B, B2C, or region-wise
- Ask the user for their preferred segmentation if not specified.
- Use structured format to present segment data.
- Do not include segment data if not present in the document, but don't mention its absence.

For dates:
- If a date is not explicitly mentioned, assume the latest available date in the document.

Your goal is to deliver accurate, document-grounded financial insights that are useful, clear, and well-structured for business users.
"""
    
    def _format_user_message(self, query: str, context: str, conversation_history: str) -> str:
        """Format the user message with context"""
        return f"""Based on the following financial document content, please answer the user's question.

DOCUMENT CONTEXT:
{context}

CONVERSATION HISTORY:
{conversation_history}

USER QUESTION: {query}

Please provide a comprehensive financial analysis based on the document content above."""
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model"""
        return {
            "provider": self.current_provider,
            "model": self.current_model,
            "status": self.current_provider in self.clients
        }
    
    def test_model_availability(self, provider: str, model: str) -> bool:
        """Test if a specific model is available"""
        try:
            if provider == 'groq' and 'groq' in self.clients:
                return True
            return False
        except:
            return False
