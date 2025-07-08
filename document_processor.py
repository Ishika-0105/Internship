import pandas as pd
import requests
from bs4 import BeautifulSoup
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from config import Config
import streamlit as st
import io
import fitz  # PyMuPDF 
from typing import List, Dict, Any
import logging
import pytesseract
from PIL import Image
import numpy as np

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DocumentProcessor:
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
    
    def process_pdf(self, file):
        """Process PDF files with enhanced text extraction"""
        try:
            documents = []
            pdf_document = fitz.open(stream=file.read(), filetype="pdf")
            
            for page_num in range(len(pdf_document)):
                page = pdf_document.load_page(page_num)
                
                # Extract text with better formatting
                text = page.get_text("text")
                
                # If text extraction fails, try with different method
                if not text or len(text.strip()) < 50:
                    text = page.get_text("dict")
                    text_content = ""
                    if 'blocks' in text:
                        for block in text['blocks']:
                            if 'lines' in block:
                                for line in block['lines']:
                                    if 'spans' in line:
                                        for span in line['spans']:
                                            if 'text' in span:
                                                text_content += span['text'] + " "
                    text = text_content
                
                # Clean and process text
                if text and len(text.strip()) > 20:
                    cleaned_text = self._clean_text(text)
                    if cleaned_text:
                        documents.append(Document(
                            page_content=cleaned_text,
                            metadata={
                                "source": file.name,
                                "page": page_num + 1,
                                "type": "pdf"
                            }
                        ))
            
            pdf_document.close()
            
            if not documents:
                st.error("❌ Could not extract readable text from PDF. Please ensure the PDF contains text content.")
                return []
            
            # Split documents into chunks
            all_chunks = []
            for doc in documents:
                chunks = self.text_splitter.split_documents([doc])
                all_chunks.extend(chunks)
            
            return all_chunks
            
        except Exception as e:
            st.error(f"❌ Error processing PDF: {str(e)}")
            return []
    
    def process_csv(self, file) -> List[Document]:
        """Process CSV files with enhanced error handling"""
        try:
            # Try different encodings
            encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
            df = None
            
            for encoding in encodings:
                try:
                    file.seek(0)  # Reset file pointer
                    df = pd.read_csv(file, encoding=encoding)
                    break
                except UnicodeDecodeError:
                    continue
            
            if df is None:
                st.error("❌ Could not read CSV file with any supported encoding.")
                return []
            
            # Create comprehensive text representation
            text_content = self._create_csv_text_representation(df, file.name)
            
            document = Document(
                page_content=text_content,
                metadata={
                    "source": file.name,
                    "type": "csv",
                    "rows": len(df),
                    "columns": len(df.columns),
                    "column_names": list(df.columns),
                    "numeric_columns": list(df.select_dtypes(include=[np.number]).columns),
                    "text_columns": list(df.select_dtypes(include=['object']).columns)
                }
            )
            
            chunks = self.text_splitter.split_documents([document])
            
            # Add chunk-specific metadata
            for i, chunk in enumerate(chunks):
                chunk.metadata.update({
                    "chunk_id": i,
                    "total_chunks": len(chunks)
                })
            
            logger.info(f"Successfully processed CSV: {file.name} - {len(chunks)} chunks created")
            st.success(f"✅ Processed CSV: {file.name} ({len(df)} rows, {len(df.columns)} columns)")
            
            return chunks
            
        except Exception as e:
            logger.error(f"Error processing CSV {file.name}: {str(e)}")
            st.error(f"❌ Error processing CSV: {str(e)}")
            return []
    
    def process_xlsx(self, file) -> List[Document]:
        """Process Excel files with comprehensive multi-sheet support"""
        try:
            documents = []
            
            # Read the Excel file
            file_bytes = file.read()
            
            with st.spinner("📊 Processing all Excel sheets..."):
                try:
                    # Get all sheet names
                    excel_file = pd.ExcelFile(io.BytesIO(file_bytes))
                    sheet_names = excel_file.sheet_names
                    
                    st.info(f"📋 Found {len(sheet_names)} sheets: {', '.join(sheet_names)}")
                    
                    for sheet_name in sheet_names:
                        try:
                            # Read each sheet
                            df = pd.read_excel(io.BytesIO(file_bytes), sheet_name=sheet_name)
                            
                            if df.empty:
                                logger.warning(f"Sheet '{sheet_name}' is empty, skipping")
                                continue
                            
                            # Create comprehensive text representation for this sheet
                            text_content = self._create_excel_sheet_text_representation(df, file.name, sheet_name)
                            
                            if text_content:
                                document = Document(
                                    page_content=text_content,
                                    metadata={
                                        "source": file.name,
                                        "sheet_name": sheet_name,
                                        "type": "xlsx",
                                        "rows": len(df),
                                        "columns": len(df.columns),
                                        "column_names": list(df.columns),
                                        "numeric_columns": list(df.select_dtypes(include=[np.number]).columns),
                                        "text_columns": list(df.select_dtypes(include=['object']).columns),
                                        "has_formulas": any(str(cell).startswith('=') for col in df.columns for cell in df[col].astype(str) if pd.notna(cell))
                                    }
                                )
                                
                                # Split sheet content into chunks
                                sheet_chunks = self.text_splitter.split_documents([document])
                                
                                # Update metadata for each chunk
                                for i, chunk in enumerate(sheet_chunks):
                                    chunk.metadata.update({
                                        "chunk_id": i,
                                        "total_chunks_in_sheet": len(sheet_chunks),
                                        "sheet_chunk_id": f"{sheet_name}_chunk_{i}"
                                    })
                                
                                documents.extend(sheet_chunks)
                                logger.info(f"Processed sheet '{sheet_name}': {len(sheet_chunks)} chunks")
                                
                        except Exception as e:
                            logger.error(f"Error processing sheet '{sheet_name}': {str(e)}")
                            st.warning(f"⚠️ Could not process sheet '{sheet_name}': {str(e)}")
                            continue
                    
                except Exception as e:
                    logger.error(f"Error reading Excel file structure: {str(e)}")
                    st.error(f"❌ Error reading Excel file structure: {str(e)}")
                    return []
            
            if documents:
                total_sheets = len(set(doc.metadata.get('sheet_name', 'Unknown') for doc in documents))
                st.success(f"✅ Processed Excel file: {file.name} ({total_sheets} sheets, {len(documents)} total chunks)")
                logger.info(f"Successfully processed Excel: {file.name} - {len(documents)} chunks from {total_sheets} sheets")
            else:
                st.error("❌ No data could be extracted from any Excel sheets.")
            
            return documents
            
        except Exception as e:
            logger.error(f"Error processing Excel file {file.name}: {str(e)}")
            st.error(f"❌ Error processing Excel file: {str(e)}")
            return []
    
    def process_website(self, url: str) -> List[Document]:
        """Process website content with enhanced extraction"""
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            
            with st.spinner(f"🌐 Fetching content from {url}..."):
                response = requests.get(url, headers=headers, timeout=15)
                response.raise_for_status()
                
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Remove unwanted elements
                for element in soup(["script", "style", "nav", "header", "footer", "aside"]):
                    element.decompose()
                
                # Extract text content
                text = soup.get_text()
                cleaned_text = self._clean_text(text)
                
                if not cleaned_text or len(cleaned_text.strip()) < 100:
                    st.error("❌ Could not extract meaningful content from the website.")
                    return []
                
                document = Document(
                    page_content=cleaned_text,
                    metadata={
                        "source": url,
                        "type": "website",
                        "title": soup.title.string if soup.title else "Unknown",
                        "content_length": len(cleaned_text)
                    }
                )
                
                chunks = self.text_splitter.split_documents([document])
                
                # Add chunk metadata
                for i, chunk in enumerate(chunks):
                    chunk.metadata.update({
                        "chunk_id": i,
                        "total_chunks": len(chunks)
                    })
                
                st.success(f"✅ Processed website: {len(chunks)} chunks extracted")
                return chunks
                
        except Exception as e:
            logger.error(f"Error processing website {url}: {str(e)}")
            st.error(f"❌ Error processing website: {str(e)}")
            return []
    
    def _create_csv_text_representation(self, df: pd.DataFrame, filename: str) -> str:
        """Create a comprehensive text representation of CSV data"""
        text_parts = []
        
        # Add header information
        text_parts.append(f"=== CSV FILE: {filename} ===")
        text_parts.append(f"Rows: {len(df)}, Columns: {len(df.columns)}")
        text_parts.append(f"Column Names: {', '.join(df.columns)}")
        
        # Add data types information
        text_parts.append("\n=== COLUMN INFORMATION ===")
        for col in df.columns:
            dtype = str(df[col].dtype)
            non_null = df[col].count()
            text_parts.append(f"{col}: {dtype} ({non_null} non-null values)")
        
           # Add sample data
        text_parts.append(f"\n=== SAMPLE DATA (First {min(10, len(df))} rows) ===")
        text_parts.append(df.head(10).to_string(index=False))
        
        # Add full data if it's not too large
        if len(df) <= 100:
            text_parts.append(f"\n=== COMPLETE DATA ===")
            text_parts.append(df.to_string(index=False))
        
        return "\n".join(text_parts)
    
    def _create_excel_sheet_text_representation(self, df: pd.DataFrame, filename: str, sheet_name: str) -> str:
        """Create a comprehensive text representation of Excel sheet data"""
        text_parts = []
        
        # Add header information
        text_parts.append(f"=== EXCEL FILE: {filename} | SHEET: {sheet_name} ===")
        text_parts.append(f"Rows: {len(df)}, Columns: {len(df.columns)}")
        text_parts.append(f"Column Names: {', '.join(df.columns)}")
        
        # Add data types information
        text_parts.append(f"\n=== COLUMN INFORMATION (Sheet: {sheet_name}) ===")
        for col in df.columns:
            dtype = str(df[col].dtype)
            non_null = df[col].count()
            unique_vals = df[col].nunique()
            text_parts.append(f"{col}: {dtype} ({non_null} non-null, {unique_vals} unique values)")
        
        # Add categorical summaries
        text_cols = df.select_dtypes(include=['object']).columns
        if len(text_cols) > 0:
            text_parts.append(f"\n=== CATEGORICAL SUMMARY (Sheet: {sheet_name}) ===")
            for col in text_cols:
                if df[col].count() > 0:
                    value_counts = df[col].value_counts().head(5)
                    text_parts.append(f"{col} top values: {dict(value_counts)}")
        
        # Add sample data
        text_parts.append(f"\n=== SAMPLE DATA (Sheet: {sheet_name}, First {min(10, len(df))} rows) ===")
        text_parts.append(df.head(10).to_string(index=False))
        
        # Add full data if it's reasonably sized
        if len(df) <= 50:
            text_parts.append(f"\n=== COMPLETE DATA (Sheet: {sheet_name}) ===")
            text_parts.append(df.to_string(index=False))
        
        return "\n".join(text_parts)
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text with enhanced processing"""
        if not text:
            return ""
        
        # Split into lines and clean each line
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            line = line.strip()
            # Skip very short lines, but keep meaningful short lines like numbers
            if line and (len(line) > 2 or line.isdigit() or any(char.isalnum() for char in line)):
                # Remove excessive whitespace
                line = ' '.join(line.split())
                cleaned_lines.append(line)
        
        # Join lines and handle special cases
        cleaned_text = '\n'.join(cleaned_lines)
        
        # Remove excessive blank lines
        while '\n\n\n' in cleaned_text:
            cleaned_text = cleaned_text.replace('\n\n\n', '\n\n')
        
        return cleaned_text.strip()
