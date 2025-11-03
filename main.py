import streamlit as st
from datetime import datetime
import time
from core import runn_llm

def main():
    # Page configuration
    st.set_page_config(
        page_title="Documents RAG System",
        page_icon="🤖",
        layout="centered"
    )
    
    # Custom CSS
    st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        border-left: 4px solid #1f77b4;
    }
    .user-message {
        background-color: #e3f2fd;
        border-left-color: #2196f3;
    }
    .assistant-message {
        background-color: #f3e5f5;
        border-left-color: #9c27b0;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown('<h1 class="main-header">Documents CHATBOT</h1>', unsafe_allow_html=True)
    
    # Initialize session state for conversation history
    if 'messages' not in st.session_state:
        st.session_state.messages = []
        #list of dicts with role and content
        st.session_state.chat_history = []
    
    # Display conversation history
    if st.session_state.messages:
        st.subheader("💬 Chat History")
    
    for message in st.session_state.messages:
        if message["role"] == "user":
            st.markdown(f"""
            <div class="chat-message user-message">
                <strong>👤 You:</strong><br>
                {message["content"]}
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="chat-message assistant-message">
                <strong>🤖 Assistant:</strong><br>
                {message["content"]}
            </div>
            """, unsafe_allow_html=True)
    
    # Input form
    with st.form("chat_form", clear_on_submit=True):
        user_input = st.text_area(
            "Write your question:",
            placeholder="¿Qué información buscas en el documento?",
            height=100
        )
        
        col1, col2 = st.columns([1, 4])
        with col1:
            submit_button = st.form_submit_button("Send", use_container_width=True)
        with col2:
            clear_button = st.form_submit_button("Clear History", use_container_width=True)
        
        if clear_button:
            st.session_state.messages = []
            st.rerun()
        
        if submit_button and user_input:
            # Add user message to history
            st.session_state.messages.append({
                "role": "user", 
                "content": user_input,
                "timestamp": datetime.now()
            })
            st.session_state.chat_history.append({"role": "user", "content": user_input})
            # Process the query using core.py function
            with st.spinner("Processing your question..."):
                try:
                    # Start timing
                    start_time = time.time()
                    response = runn_llm(user_input,st.session_state.chat_history)
                    # End timing
                    elapsed_time = time.time() - start_time
                    
                    # Add assistant response to history with elapsed time
                    response_with_time = f"{response}\n\n⏱️ Response time: {elapsed_time:.2f} seconds"
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response_with_time,
                        "timestamp": datetime.now()
                    })
                    st.session_state.chat_history.append({"role": "assistant", "content": response})
                except Exception as e:
                    error_message = f"Error processing the question: {str(e)}"
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_message,
                        "timestamp": datetime.now()
                    })
                    st.session_state.chat_history.append({"role": "assistant", "content": error_message})
            st.rerun()

if __name__ == "__main__":
    main()
