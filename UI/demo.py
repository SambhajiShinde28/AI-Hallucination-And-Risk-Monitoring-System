import streamlit as st

st.set_page_config(page_title="Chat Threads UI", layout="wide")

# ---------------- Session State ----------------
if "threads" not in st.session_state:
    st.session_state.threads = {}

if "active_thread" not in st.session_state:
    st.session_state.active_thread = None

# New Chat Button
if st.sidebar.button("➕ New Chat"):
    thread_name = f"Chat {len(st.session_state.threads) + 1}"
    st.session_state.threads[thread_name] = []
    st.session_state.active_thread = thread_name

st.sidebar.divider()

# Show Threads List
for thread in st.session_state.threads.keys():
    if st.sidebar.button(thread):
        st.session_state.active_thread = thread

# ---------------- Main Area ----------------
if st.session_state.active_thread:
    st.title(st.session_state.active_thread)
else:
    st.title("Click 'New Chat' to start")