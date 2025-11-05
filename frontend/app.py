#!/usr/bin/env python3
"""
Streamlit Frontend for Chatbot Aktuaria

This module provides a web interface for the actuarial chatbot using Streamlit.
It connects to the Flask backend API to process user queries and display responses.
"""

import streamlit as st
import requests
import json
import logging
from typing import Dict, Any, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
FLASK_API_URL = "http://localhost:5001"  # Updated to match backend port
API_ENDPOINTS = {
    "chat": f"{FLASK_API_URL}/api/chat",
    "health": f"{FLASK_API_URL}/health"
}

def check_backend_health() -> bool:
    """
    Check if Flask backend is running and healthy.
    
    Returns:
        bool: True if backend is healthy, False otherwise
    """
    try:
        response = requests.get(API_ENDPOINTS["health"], timeout=5)
        return response.status_code == 200
    except requests.exceptions.RequestException as e:
        logger.error(f"Backend health check failed: {e}")
        return False

def send_chat_message(message: str) -> Optional[Dict[str, Any]]:
    """
    Send chat message to Flask backend.
    
    Args:
        message (str): User message to send
        
    Returns:
        Optional[Dict[str, Any]]: Response from backend or None if failed
    """
    try:
        payload = {"message": message}
        response = requests.post(
            API_ENDPOINTS["chat"],
            json=payload,
            timeout=30,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            logger.error(f"API request failed with status {response.status_code}")
            return None
            
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to send chat message: {e}")
        return None

def main():
    """
    Main Streamlit application.
    """
    # Page configuration
    st.set_page_config(
        page_title="Chatbot Aktuaria",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Header
    st.title("🤖 Chatbot Aktuaria")
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.header("ℹ️ Informasi")
        
        # Backend health status
        if check_backend_health():
            st.success("✅ Backend Connected")
        else:
            st.error("❌ Backend Disconnected")
            st.warning("Pastikan Flask backend berjalan di port 5000")
        
        st.markdown("---")
        st.markdown("""
        **Fitur:**
        - Query expansion dengan sinonim
        - Retrieval multi-query
        - Reranking hasil
        - Confidence scoring
        """)
        
        # Clear chat button
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
        # Add welcome message
        st.session_state.messages.append({
            "role": "assistant",
            "content": "Halo! Saya adalah chatbot aktuaria. Silakan tanyakan pertanyaan terkait aktuaria, PSAK 219, atau topik terkait lainnya."
        })
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ketik pertanyaan Anda di sini..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get response from backend
        with st.chat_message("assistant"):
            with st.spinner("Memproses pertanyaan..."):
                response = send_chat_message(prompt)
                
                if response and "response" in response:
                    assistant_response = response["response"]
                    st.markdown(assistant_response)
                    
                    # Add assistant response to chat history
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": assistant_response
                    })
                    
                    # Show additional info if available
                    if "metadata" in response:
                        with st.expander("📊 Detail Respons"):
                            metadata = response["metadata"]
                            
                            if "confidence_score" in metadata:
                                st.metric("Confidence Score", f"{metadata['confidence_score']:.2f}")
                            
                            if "sources" in metadata:
                                st.subheader("📚 Sumber Dokumen")
                                for i, source in enumerate(metadata["sources"][:3], 1):
                                    # Check if source has new format with full content
                                    if isinstance(source, dict):
                                        if "ai_ready_format" in source and source["ai_ready_format"]:
                                            # Display AI-ready format with full content
                                            st.text_area(
                                                f"Sumber {i}:",
                                                value=source["ai_ready_format"],
                                                height=150,
                                                disabled=True,
                                                key=f"source_{i}"
                                            )
                                        elif "display_content" in source and source["display_content"]:
                                            # Display content with filename
                                            filename = source.get("filename", "Unknown")
                                            content = source["display_content"]
                                            st.text_area(
                                                f"Sumber {i} - {filename}:",
                                                value=content,
                                                height=150,
                                                disabled=True,
                                                key=f"source_{i}"
                                            )
                                        else:
                                            # Fallback to old format
                                            st.write(f"{i}. {source}")
                                    else:
                                        # Old string format
                                        st.write(f"{i}. {source}")
                            
                            if "query_expansion" in metadata:
                                st.subheader("🔍 Query Expansion")
                                for i, query in enumerate(metadata["query_expansion"], 1):
                                    st.write(f"{i}. {query}")
                else:
                    error_message = "Maaf, terjadi kesalahan saat memproses pertanyaan Anda. Silakan coba lagi."
                    st.error(error_message)
                    
                    # Add error message to chat history
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_message
                    })

if __name__ == "__main__":
    main()