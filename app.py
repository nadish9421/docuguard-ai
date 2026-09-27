from utils.vector_store import LocalVectorStore
from utils.document_parser import extract_text
import streamlit as st
import time
# from npu_engine import process_query  # We will uncomment this when we build the NPU logic!

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="DocuGuard AI - Local NPU App", page_icon="🛡️", layout="centered")

# --- HEADER ---
st.title("🛡️ DocuGuard AI")
st.caption("On-Device Document Analysis powered by Snapdragon Hexagon NPU")
st.markdown("**Developer:** Nadish CP (Solo Participant)")

# --- SESSION STATE ---
# Streamlit re-runs the script on every interaction. We use session_state to remember things.
if "messages" not in st.session_state:
    st.session_state.messages = []
if "doc_loaded" not in st.session_state:
    st.session_state.doc_loaded = False
if "vector_store" not in st.session_state:
    st.session_state.vector_store = LocalVectorStore()

# --- SIDEBAR: DOCUMENT UPLOAD ---
with st.sidebar:
    st.header("1. Secure Upload")
    st.markdown("Your document never leaves this device.")
    
    uploaded_file = st.file_uploader("Upload a PDF or TXT file", type=["pdf", "txt"])
    
    if uploaded_file is not None:
        if st.button("Process on NPU 🚀"):
            with st.spinner("Extracting text and building local index..."):
                # Here we will eventually call your utils/document_parser.py
                time.sleep(1.5) # Simulating the processing time for now
                st.session_state.doc_loaded = True
                st.success("Document processed and secured locally!")
    
    st.divider()
    st.subheader("Hardware Status")
    st.markdown("🟢 **Execution:** Hexagon NPU")
    st.markdown("🟢 **Privacy Mode:** 100% Offline")
    
with st.spinner("Extracting text and building local index..."):
    document_text = extract_text(uploaded_file)
    st.session_state.vector_store.build_index(document_text) # This builds the NPU vectors
    st.session_state.doc_loaded = True
    st.success("Document processed and secured locally!")

# --- MAIN CHAT INTERFACE ---
st.header("2. Chat with your Data")

# Display previous chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input block
if not st.session_state.doc_loaded:
    st.info("👈 Please upload and process a document in the sidebar to start asking questions.")
else:
    prompt = st.chat_input("Ask a question about your secure document...")
    
    if prompt:
        # 1. Display the user's message
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # 2. Generate the AI response
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            
            with st.spinner("NPU is generating response..."):
                # --- NPU INTEGRATION POINT ---
                # In the next step, we will route this prompt to npu_engine.py
                time.sleep(1) # Simulating NPU inference time
                
                # Placeholder response for UI testing
                full_response = f"This is a simulated on-device response from the NPU regarding: '{prompt}'. Once we connect the ONNX model, actual answers will appear here."
                
                # Creating a cool typing/streaming effect for the UI
                streamed_text = ""
                for chunk in full_response.split():
                    streamed_text += chunk + " "
                    time.sleep(0.05)
                    message_placeholder.markdown(streamed_text + "▌")
                
                message_placeholder.markdown(streamed_text)
        
        # 3. Save AI response to history
        st.session_state.messages.append({"role": "assistant", "content": full_response})