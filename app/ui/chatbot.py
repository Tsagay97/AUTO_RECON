import os
import re

import streamlit as st
import pandas as pd
from dotenv import load_dotenv
from langchain_experimental.agents import create_pandas_dataframe_agent
from openai import OpenAI

# ──────────────────────────────────────────────────────────────────────────────
# 1) Load local .env without overwriting real environment variables
load_dotenv(override=False)

# 2) Fetch API key: prefer real env-var, then Streamlit secret
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError(
        "OPENAI_API_KEY not found. "
        "Locally, put it in a `.env` file; "
        "in Streamlit Cloud, set it under Settings → Secrets."
    )

# 3) Initialize OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)

# ──────────────────────────────────────────────────────────────────────────────
def clean_latex(text: str) -> str:
    """Remove simple LaTeX commands from responses."""
    text = re.sub(r'\\boxed\{(.*?)\}', r'\1', text, flags=re.DOTALL)
    text = re.sub(r'\\text\{(.*?)\}', r'\1', text, flags=re.DOTALL)
    text = re.sub(r'\\(?:[a-zA-Z]+)\{.*?\}', '', text, flags=re.DOTALL)
    text = re.sub(r'\\[\[\]\{\}]', '', text)
    return text.strip()

class OpenAILLM:
    """Wrapper to match LangChain’s LLM interface."""
    def __init__(self):
        self.client = client

    def __call__(self, prompt, **kwargs):
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1500,
                temperature=0.7
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"Error calling OpenAI API: {e}"

openai_llm = OpenAILLM()

def create_data_agent(df: pd.DataFrame):
    return create_pandas_dataframe_agent(
        llm=openai_llm,
        df=df,
        verbose=True,
        allow_dangerous_code=True
    )

def route_query(query: str) -> str:
    q = query.lower()
    faq_keys = [
        "how do i use", "what file types", "what happens after", "exact matching",
        "fuzzy matching", "result categories", "what can i ask", "how is this different",
        "my pdf isn’t uploading", "can i export"
    ]
    data_keys = [
        "matched", "unmatched", "ricbl", "bob", "policy", "amount",
        "entries", "records", "verified matches"
    ]
    if any(k in q for k in data_keys):
        return "data"
    if any(k in q for k in faq_keys):
        return "faq"
    return "general"

FAQ_CONTENT = {
    "how do i use this system": (
        "To use the AI Reconciliation System:\n"
        "1. Start by uploading your BOB and RICBL PDF statements using the sidebar.\n"
        "2. The system will display raw data from both files.\n"
        "3. Click on 'Cleaned Data' to see the cleaned, structured version of your data.\n"
        "4. Then, go to 'Exact Matching' and select the columns you want to match.\n"
        "5. Click 'Start Matching' to perform the reconciliation.\n"
        "6. You’ll see a summary of matched, unmatched, and flagged transactions."
    ),
    "what file types can i upload": (
        "You can upload PDF files only — specifically bank statements from BOB and RICBL. "
        "The system will extract tabular data from these files."
    ),
    "what happens after i upload my files": (
        "Once both PDF files are uploaded:\n"
        "1. The system automatically extracts the tables.\n"
        "2. You’ll see the raw data under the 'Raw Data' tab.\n"
        "3. You can then clean the data and begin matching."
    ),
    "what is exact matching": (
        "Exact Matching compares the transaction amounts from RICBL and BOB. "
        "If both values match exactly, the record is marked as a verified match."
    ),
    "what is fuzzy matching": (
        "Fuzzy Matching checks if the policy/account number from RICBL appears (even partially) in the narration from BOB. "
        "It’s useful when the formats don’t match perfectly."
    ),
    "what do the result categories mean": (
        "✅ Verified Matches: Amount and policy both matched\n"
        "⚠️ Flagged Matches: Amount matched, but policy needs review\n"
        "❌ Unmatched: No match found in the other statement"
    ),
    "what can i ask the chatbot": (
        "You can ask questions like:\n"
        "• 'How many matched transactions are over 1000?'\n"
        "• 'Show unmatched RICBL records'\n"
        "• 'What is reconciliation?'\n"
        "The chatbot can answer both data-related and general questions about the system."
    ),
    "how is this chatbot different": (
        "This chatbot combines two AI tools:\n"
        "1. One for answering questions based on your uploaded data\n"
        "2. Another for explaining concepts or helping with how-to questions"
    ),
    "my pdf isn’t uploading — what should i do": (
        "Make sure your PDF is under 50MB and contains tabular data. "
        "If the file is scanned as an image, the system may not extract it correctly."
    ),
    "can i export the results": (
        "Export functionality is coming soon! You’ll be able to download matched and unmatched records as CSV files."
    )
}

def handle_faq_query(query: str) -> str:
    q = query.lower()
    for question, answer in FAQ_CONTENT.items():
        if question in q:
            return answer
    return (
        "I’m not sure how to answer that. Here’s a general guide to get you started:\n"
        + FAQ_CONTENT.get("how do i use this system", "")
    )

def select_dataframe(query: str) -> pd.DataFrame:
    q = query.lower()
    if "matched" in q or "verified matches" in q:
        return st.session_state.matched_df, "matched_df"
    if "unmatched" in q and "bob" in q:
        return st.session_state.unmatched_bob_df, "unmatched_bob_df"
    if "unmatched" in q and "ricbl" in q:
        return st.session_state.unmatched_ricb_df, "unmatched_ricb_df"
    if "bob" in q:
        return st.session_state.unmatched_bob_df, "unmatched_bob_df"
    if "ricbl" in q:
        return st.session_state.unmatched_ricb_df, "unmatched_ricb_df"
    return st.session_state.matched_df, "matched_df"

def handle_data_query(query: str) -> str:
    if not all(k in st.session_state for k in ["matched_df", "unmatched_bob_df", "unmatched_ricb_df"]):
        return "No data available. Please upload files and start matching first."
    q = query.lower()
    if "first 5" in q and "matched" in q:
        df = st.session_state.matched_df
        return df.head(5).to_string() if not df.empty else "No matched records."
    if "how many" in q and "verified matches" in q:
        return str(len(st.session_state.matched_df))
    if "show me all" in q or "show all" in q:
        df, _ = select_dataframe(query)
        return df.to_string()
    df, _ = select_dataframe(query)
    agent = create_data_agent(df)
    try:
        response = agent.run(query)
        return response.to_string() if isinstance(response, pd.DataFrame) else str(response)
    except Exception as e:
        return f"Error processing data query: {e}"

def handle_general_query(query: str) -> str:
    prompt = (
        "Answer the following question in a concise, plain‐text way (no LaTeX):\n"
        + query
    )
    resp = openai_llm(prompt)
    return clean_latex(resp)

def process_query(query: str) -> str:
    t = route_query(query)
    if t == "faq":
        return handle_faq_query(query)
    if t == "data":
        return handle_data_query(query)
    return handle_general_query(query)
