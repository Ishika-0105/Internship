from groq import Groq
from config import Config
import streamlit as st
from typing import List, Dict, Any

class GroqClient:
    def __init__(self):
        if not Config.GROQ_API_KEY:
            st.error("❌ GROQ_API_KEY not found. Please add it to your .env file.")
            st.stop()
        
        self.client = Groq(api_key=Config.GROQ_API_KEY)
        self.model = "llama3-70b-8192"  # Using llama for better performance
    
    def generate_response(self, query: str, context: str, conversation_history: str = "") -> str:
        """Generate response using Groq with RAG context"""
        try:
            # Create system prompt for financial document analysis
            system_prompt = """You are FinTelBot, a specialized financial analyst chatbot focused on telecom company financial documents. You assist users in analyzing and interpreting quarterly and annual financial reports strictly based on the content retrieved from the document's vector database.

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
- Do not include segment data if not present in the document.

For dates:
- If a date is not explicitly mentioned, assume the latest available date in the document.

Your goal is to deliver accurate, document-grounded financial insights that are useful, clear, and well-structured for business users."""

            # Prepare the user message with context
            user_message = f"""Based on the attached financial document content, please answer the user's question.

DOCUMENT CONTEXT:
{context}

CONVERSATION HISTORY:
{conversation_history}

USER QUESTION: {query}

Please provide a comprehensive answer based on the document content above."""

            # Make the API call
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message}
                ],
                max_tokens=Config.MAX_TOKENS,
                temperature=Config.TEMPERATURE,
                stream=False
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            st.error(f"❌ Error generating response: {str(e)}")
            print(f"❌ Error generating response: {str(e)}")
            return f"I apologize, but I encountered an error while processing your request. Please try again."
    
    def generate_summary(self, documents_content: str) -> str:
        """Generate a summary of uploaded documents"""
        try:
            prompt = f"""Please provide a brief summary of the following financial documents. Focus on:
1. Type of document (annual report, quarterly report, etc.)
2. Company name and reporting period
3. Key financial highlights
4. Main sections or topics covered

Document content:
{documents_content[:2000]}...

Provide a concise summary in 3-4 sentences."""

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=300,
                temperature=0.7,
                stream=False
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            return f"Error generating summary: {str(e)}"
