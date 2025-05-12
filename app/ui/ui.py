import streamlit as st
from app.logic.extractor import extract_tables_from_pdf
from app.logic.cleaner import clean_bob_data, clean_ricb_data
from app.ui.chatbot import process_query
import html

def load_css(file_path="app/ui/style.css"):
    with open(file_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

def render_ui():
    st.set_page_config(layout="wide")
    load_css()
    
    st.title("📊 AI Reconciliation System")

    # Initialize session state for chat and matching
    if "match_started" not in st.session_state:
        st.session_state.match_started = False
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Sidebar for file uploads and chat
    with st.sidebar:
        st.header("📂 Upload Files")
        bob_file = st.file_uploader("Upload BOB PDF", type=["pdf"], key="bob_pdf")
        ricb_file = st.file_uploader("Upload RICBL PDF", type=["pdf"], key="ricb_pdf")

        # Chat section in the sidebar
        with st.expander("💬 Chat with Assistant", expanded=False):
            # Welcome message (shown only on first open)
            if len(st.session_state.chat_history) == 0:
                welcome_msg = "Hi! I can help with reconciliation questions. Ask me about matched records, unmatched entries, or general topics like fuzzy matching."
                st.session_state.chat_history.append({"role": "assistant", "content": welcome_msg})

            # Scrollable chat window
            chat_container = st.container(height=300)  # Set a fixed height for scrollable area
            with chat_container:
                for message in st.session_state.chat_history:
                    with st.chat_message(message["role"]):
                        # Escape the content to prevent LaTeX rendering, but allow HTML styling
                        escaped_content = html.escape(message["content"])
                        st.markdown(f'<div class="chat-message {message["role"]}">{escaped_content}</div>', unsafe_allow_html=True)

            # Chat input
            user_input = st.chat_input("Ask me anything...", key="chat_input")
            if user_input:
                # Add user message to history
                st.session_state.chat_history.append({"role": "user", "content": user_input})
                
                # Process the query using the chatbot logic
                bot_response = process_query(user_input)
                st.session_state.chat_history.append({"role": "assistant", "content": bot_response})

                # Rerun to update the chat window
                st.rerun()

    # Main content
    if not bob_file or not ricb_file:
        st.info("⬅️ Please upload both BOB and RICBL PDF files to continue.")
    else:
        bob_raw_df = extract_tables_from_pdf(bob_file)
        ricb_raw_df = extract_tables_from_pdf(ricb_file)  # Fixed typo: rocob_raw_df -> ricb_raw_df
        bob_df = clean_bob_data(bob_raw_df)
        ricb_df = clean_ricb_data(ricb_raw_df)

        view = st.radio("🔀 Select View Mode", ["📄 Raw Data", "🧹 Cleaned Tables", "🔁 Exact Matching"], horizontal=True)

        if view == "📄 Raw Data":
            st.subheader("📄 Raw Extracted Tables")
            st.markdown("### 🧾 BOB Raw Table")
            st.dataframe(bob_raw_df, use_container_width=True)
            st.markdown("### 🧾 RICBL Raw Table")
            st.dataframe(ricb_raw_df, use_container_width=True)

        elif view == "🧹 Cleaned Tables":
            st.subheader("🧹 Cleaned Tables")
            st.markdown("### ✅ BOB Cleaned Table")
            st.dataframe(bob_df, use_container_width=True)
            st.markdown("### ✅ RICBL Cleaned Table")
            st.dataframe(ricb_df, use_container_width=True)

        elif view == "🔁 Exact Matching":
            st.subheader("🔁 Exact Matching Options")
            col1, col2 = st.columns(2)
            with col1:
                bob_match_col = st.selectbox("🔷 Select BOB Column to Match", bob_df.columns.tolist())
            with col2:
                ricb_match_col = st.selectbox("🔶 Select RICBL Column to Match", ricb_df.columns.tolist())

            if st.button("▶️ Start Matching"):
                st.session_state.match_started = True
                st.session_state.bob_match_col = bob_match_col
                st.session_state.ricb_match_col = ricb_match_col

            if st.session_state.match_started:
                matched = bob_df[bob_df[st.session_state.bob_match_col].isin(ricb_df[st.session_state.ricb_match_col])]
                unmatched_bob = bob_df[~bob_df[st.session_state.bob_match_col].isin(ricb_df[st.session_state.ricb_match_col])]
                unmatched_ricb = ricb_df[~ricb_df[st.session_state.ricb_match_col].isin(bob_df[st.session_state.bob_match_col])]

                # Store DataFrames in session state for chatbot access
                st.session_state.matched_df = matched
                st.session_state.unmatched_bob_df = unmatched_bob
                st.session_state.unmatched_ricb_df = unmatched_ricb

                st.subheader("📊 Reconciliation Summary")
                c1, c2, c3 = st.columns(3)
                with c1:
                    st.markdown(f"<div class='stat-card blue'><h3>🔷 BOB Total</h3><h2>{len(bob_df)}</h2></div>", unsafe_allow_html=True)
                with c2:
                    st.markdown(f"<div class='stat-card orange'><h3>🔶 RICBL Total</h3><h2>{len(ricb_df)}</h2></div>", unsafe_allow_html=True)
                with c3:
                    st.markdown(f"<div class='stat-card green'><h3>✅ Matched</h3><h2>{len(matched)}</h2></div>", unsafe_allow_html=True)

                c4, c5 = st.columns(2)
                with c4:
                    st.markdown(f"<div class='stat-card red'><h3>❌ Unmatched in BOB</h3><h2>{len(unmatched_bob)}</h2></div>", unsafe_allow_html=True)
                with c5:
                    st.markdown(f"<div class='stat-card red'><h3>❌ Unmatched in RICBL</h3><h2>{len(unmatched_ricb)}</h2></div>", unsafe_allow_html=True)

                with st.expander("✅ Matched Records"):
                    st.dataframe(matched, use_container_width=True)

                with st.expander("❌ Unmatched BOB Records"):
                    st.dataframe(unmatched_bob, use_container_width=True)

                with st.expander("❌ Unmatched RICBL Records"):
                    st.dataframe(unmatched_ricb, use_container_width=True)