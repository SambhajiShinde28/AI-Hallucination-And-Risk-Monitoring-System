import streamlit as st
import requests


File_Upload_URL="http://127.0.0.1:8000/file_upload/"
Question_Ask_URL="http://127.0.0.1:8000/ask/"


st.set_page_config(
    page_title="AI Hallucination And Risk Monitoring System",
    page_icon="🤖",
    layout="wide"
)

# -------------------- Session State -----------------------
if "threads" not in st.session_state:
    st.session_state.threads = {}
if "active_thread" not in st.session_state:
    st.session_state.active_thread = None
if "user_file_path" not in st.session_state:
    st.session_state.user_file_path=""


st.markdown("""
    <style>
        body {
            background: linear-gradient(120deg, #f6f9ff, #eef3ff);
            font-family: "Segoe UI", sans-serif;
        }

        /* Sidebar Styling */
        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #1e3c72, #2a5298);
            color: white;
        }

        .sidebar-title {
            font-size: 24px;
            font-weight: bold;
            color: #ffffff;
            margin-bottom: 5px;
        }

        .sidebar-desc {
            font-size: 14px;
            color: #dbe7ff;
            margin-bottom: 15px;
        }

        /* Chat Container */
        .chat-box {
            background: white;
            border-radius: 18px;
            padding: 15px;
            box-shadow: 0px 4px 15px rgba(0,0,0,0.08);
            margin-bottom: 15px;
        }

        /* Quick Action Buttons */
        div.stButton > button {
            border-radius: 12px;
            padding: 8px 18px;
            font-weight: 600;
            background: linear-gradient(90deg, #ff6a00, #ee0979);
            color: white;
            border: none;
            transition: 0.3s;
        }
        div.stButton > button:hover {
            transform: scale(1.05);
            background: linear-gradient(90deg, #ee0979, #ff6a00);
        }

        /* Chat Messages */
        .stChatMessage {
            font-size: 15px;
            line-height: 1.5;
        }

    </style>
""", unsafe_allow_html=True)

with st.sidebar:

    st.markdown("<div class='sidebar-title'>🤖 AI Hallucination And Risk Monitoring System</div>", unsafe_allow_html=True)
    st.markdown("<div class='sidebar-desc'>Your smart assistant for documents, Q&A, and insights.</div>", unsafe_allow_html=True)

    st.divider()

    file_upload,new_chat=st.tabs(["File Upload","New Chat"])

    with file_upload:
        
        st.markdown("### 📂 File Upload Section")
        uploaded_file = st.file_uploader("Upload a document (PDF Only)", type=["pdf"])

        if uploaded_file:
            files = {"file": (uploaded_file.name, uploaded_file, uploaded_file.type)}
            response = requests.post(File_Upload_URL,files=files)
            if response.status_code == 200:
                st.success("File uploaded successfully!")
                fileData=response.json()
                st.session_state.user_file_path=fileData['file_path']
            else:
                st.error("❌ Upload failed!")

        st.divider()

        st.markdown("### ✨ Helpful Instructions")
        st.info("""
        - Upload your document first  
        - Ask questions in the chat  
        - Use quick actions for summaries  
        """)

        st.divider()
        st.markdown("💡 *Powered by LangChain + LangGraph*")
    
    with new_chat:

        new_chat_BTN=st.button("➕ New Chat")
        
        if new_chat_BTN:
            thread_name = f"Chat {len(st.session_state.threads) + 1}"
            st.session_state.threads[thread_name] = []
            st.session_state.active_thread = thread_name

        with st.container(height=320):
            for thread in st.session_state.threads.keys():
                if st.button(thread):
                    st.session_state.active_thread = thread

st.markdown("## 💬 Chat with Your AI Assistant")
st.write("Ask questions, summarize documents, and explore insights.")

st.divider()

if "messages" not in st.session_state:
    st.session_state["messages"] = []

user_input = st.chat_input("Type your message here...",accept_file=True, accept_audio=True,file_type=["jpg", "jpeg", "png"])

if user_input:

    st.session_state["messages"].append({"role": "user", "content": user_input.text})

    query_data = {'ques': user_input.text, 'file_path': st.session_state.user_file_path}

    with st.spinner("🧠 Thinking..."):
        response = requests.post(Question_Ask_URL,json=query_data)
        if response.status_code == 200:
            server_data=response.json()
            st.session_state["messages"].append({"role": "assistant", "content": server_data})
        else:
            st.error("❌ Server Error!")

if st.session_state["messages"] != "":
    for i in st.session_state["messages"]:
        if i['role']=='user':
            with st.chat_message('user'):
                st.markdown(i['content'])

        elif i['role']=='assistant':
            with st.chat_message("assistant"):
                st.markdown(i['content']['answer'])
                st.markdown(
                    f"""
                    **📌 Confidence:** `{i['content']['confidence']}%`  
                    **⚠️ Risk Level:** `{i['content']['risk_level']}`  
                    **✅ Decision:** `{i['content']['decision']}`  
                    **💬 Message:** `{i['content']['message']}`
                    """
                )
            print(i['content']['answer'])
st.divider()

col1, col2, col3 = st.columns(3)

with col1:
    if st.button("📌 Summarize Document"):
        st.session_state["messages"] = [{"role": "assistant", "content": "Sure! Document summary feature will be added soon."}]

with col2:
    if st.button("❓ Ask Questions"):
        st.session_state["messages"] = [{"role": "assistant", "content": "You can ask any question about your uploaded document."}]

with col3:
    if st.button("🧹 Clear Chat"):
        st.session_state["messages"] = []

st.divider()


if __name__=="__main__":
    pass