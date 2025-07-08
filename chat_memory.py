from typing import List, Dict, Any
import streamlit as st
from datetime import datetime

class ChatMemory:
    def __init__(self, max_history: int = 10):
        self.max_history = max_history
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
        if 'conversation_context' not in st.session_state:
            st.session_state.conversation_context = []
    
    def add_message(self, role: str, content: str, sources: List[str] = None):
        """Add a message to chat history"""
        message = {
            'role': role,
            'content': content,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'sources': sources or []
        }
        
        st.session_state.chat_history.append(message)
        
        # Maintain conversation context for better responses
        if role == 'user':
            st.session_state.conversation_context.append(f"Human: {content}")
        elif role == 'assistant':
            st.session_state.conversation_context.append(f"Assistant: {content}")
        
        # Keep only recent context to avoid token limits
        if len(st.session_state.conversation_context) > self.max_history * 2:
            st.session_state.conversation_context = st.session_state.conversation_context[-self.max_history * 2:]
    
    def get_chat_history(self) -> List[Dict[str, Any]]:
        """Get full chat history"""
        return st.session_state.chat_history
    
    def get_conversation_context(self, include_last_n: int = 5) -> str:
        """Get recent conversation context for the model"""
        if not st.session_state.conversation_context:
            return ""
        
        recent_context = st.session_state.conversation_context[-include_last_n * 2:]
        return "\n".join(recent_context)
    
    def clear_history(self):
        """Clear chat history"""
        st.session_state.chat_history = []
        st.session_state.conversation_context = []
    
    def get_last_user_messages(self, n: int = 3) -> List[str]:
        """Get last n user messages for context"""
        user_messages = []
        for message in reversed(st.session_state.chat_history):
            if message['role'] == 'user':
                user_messages.append(message['content'])
                if len(user_messages) >= n:
                    break
        return list(reversed(user_messages))
