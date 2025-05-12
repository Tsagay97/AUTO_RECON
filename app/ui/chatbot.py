import streamlit as st
import pandas as pd
from langchain_experimental.agents import create_pandas_dataframe_agent
from openai import OpenAI
import re
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Initialize OpenAI client with the API key from .env
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found in .env file")
client = OpenAI(api_key=OPENAI_API_KEY)

# Function to clean LaTeX-like syntax from the response
def clean_latex(text: str) -> str:
    # Remove common LaTeX tags like \boxed{}, \text{}, etc.
    text = re.sub(r'\\boxed\{(.*?)\}', r'\1', text, flags=re.DOTALL)
    text = re.sub(r'\\text\{(.*?)\}', r'\1', text, flags=re.DOTALL)
    text = re.sub(r'\\(?:[a-zA-Z]+)\{.*?\}', '', text, flags=re.DOTALL)
    text = re.sub(r'\\[\[\]\{\}]', '', text)
    return text.strip()

# Wrapper class for OpenAI to match LangChain expectations
class OpenAILLM:
    def __init__(self):
        self.client = client

    def __call__(self, prompt, **kwargs):
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1500,
                temperature=0.7
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"Error calling OpenAI API: {str(e)}"

# Initialize OpenAI LLM
openai_llm = OpenAILLM()

# Initialize LangChain Pandas Agent with a single DataFrame
def create_data_agent(df: pd.DataFrame):
    return create_pandas_dataframe_agent(
        llm=openai_llm,
        df=df,
        verbose=True,
        allow_dangerous_code=True
    )

# Query routing logic
def route_query(query: str) -> str:
    query_lower = query.lower()
    
    # Keywords for FAQ-related queries
    faq_keywords = [
        "how do i use", "what file types", "what happens after", "exact matching", "fuzzy matching",
        "result categories", "what can i ask", "how is this different", "my pdf isn’t uploading",
        "can i export"
    ]
    is_faq_query = any(keyword in query_lower for keyword in faq_keywords)
    
    # Keywords for data-related queries
    data_keywords = ["matched", "unmatched", "ricbl", "bob", "policy", "amount", "entries", "records", "verified matches"]
    is_data_query = any(keyword in query_lower for keyword in data_keywords)
    
    if is_faq_query and not is_data_query:
        return "faq"
    elif is_data_query:
        return "data"
    else:
        return "general"

# FAQ dictionary
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

# Determine which FAQ to return based on query
def handle_faq_query(query: str) -> str:
    query_lower = query.lower()
    # Find the best matching FAQ
    for faq_question, faq_answer in FAQ_CONTENT.items():
        if faq_question in query_lower:
            return faq_answer
    # Fallback if no exact match
    return (
        "I’m not sure how to answer that. Here’s a general guide to get you started:\n"
        "To use the AI Reconciliation System:\n"
        "1. Start by uploading your BOB and RICBL PDF statements using the sidebar.\n"
        "2. The system will display raw data from both files.\n"
        "3. Click on 'Cleaned Data' to see the cleaned, structured version of your data.\n"
        "4. Then, go to 'Exact Matching' and select the columns you want to match.\n"
        "5. Click 'Start Matching' to perform the reconciliation.\n"
        "6. You’ll see a summary of matched, unmatched, and flagged transactions.\n"
        "You can also ask specific questions like 'What is fuzzy matching?' or 'How many unmatched BOB entries?'"
    )

# Determine which DataFrame to use based on the query
def select_dataframe(query: str) -> pd.DataFrame:
    query_lower = query.lower()
    if "matched" in query_lower or "verified matches" in query_lower:
        return st.session_state.matched_df, "matched_df"
    elif "unmatched" in query_lower and "bob" in query_lower:
        return st.session_state.unmatched_bob_df, "unmatched_bob_df"
    elif "unmatched" in query_lower and "ricbl" in query_lower:
        return st.session_state.unmatched_ricb_df, "unmatched_ricb_df"
    elif "bob" in query_lower:
        return st.session_state.unmatched_bob_df, "unmatched_bob_df"
    elif "ricbl" in query_lower:
        return st.session_state.unmatched_ricb_df, "unmatched_ricb_df"
    else:
        # Default to matched_df if no specific DataFrame is identified
        return st.session_state.matched_df, "matched_df"

# Handle data-related queries using LangChain Pandas Agent
def handle_data_query(query: str) -> str:
    # Check if DataFrames exist in session state
    if ("matched_df" not in st.session_state or
        "unmatched_bob_df" not in st.session_state or
        "unmatched_ricb_df" not in st.session_state):
        return "No data available. Please upload files and start matching to generate data."
    
    try:
        query_lower = query.lower()
        # Direct handling for "Give me the first 5 matched records" query
        if "first 5" in query_lower and "matched" in query_lower:
            matched_df = st.session_state.matched_df
            if len(matched_df) == 0:
                return "No matched records found."
            return matched_df.head(5).to_string()
        
        # Direct handling for "How many total verified matches" query
        if "how many" in query_lower and "verified matches" in query_lower:
            return str(len(st.session_state.matched_df))
        
        # Fallback for simple "show me all" queries
        if "show me all" in query_lower or "show all" in query_lower:
            df, df_name = select_dataframe(query)
            if "unmatched" in query_lower and "ricbl" in query_lower:
                return st.session_state.unmatched_ricb_df.to_string()
            elif "unmatched" in query_lower and "bob" in query_lower:
                return st.session_state.unmatched_bob_df.to_string()
            elif "matched" in query_lower or "verified matches" in query_lower:
                return st.session_state.matched_df.to_string()
        
        # Use LangChain Pandas Agent for other queries
        df, df_name = select_dataframe(query)
        agent = create_data_agent(df)
        response = agent.run(query)
        
        # If the response is a DataFrame, convert it to a string for display
        if isinstance(response, pd.DataFrame):
            return response.to_string()
        return str(response)
    except Exception as e:
        return f"Error processing data query: {str(e)}"

# Handle general queries using OpenAI API
def handle_general_query(query: str) -> str:
    try:
        prompt = f"Answer the following question in a concise and informative way, using plain text without any LaTeX or special formatting:\n{query}"
        response = openai_llm(prompt)
        # Clean the response to remove LaTeX (as a fallback)
        cleaned_response = clean_latex(response)
        return cleaned_response.strip()
    except Exception as e:
        return f"Error processing general query: {str(e)}"

# Main chatbot function
def process_query(query: str) -> str:
    query_type = route_query(query)
    
    if query_type == "faq":
        return handle_faq_query(query)
    elif query_type == "data":
        return handle_data_query(query)
    else:
        return handle_general_query(query)