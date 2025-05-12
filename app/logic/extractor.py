import pdfplumber
import pandas as pd

def extract_tables_from_pdf(uploaded_file) -> pd.DataFrame:
    """
    Extracts all tables from a PDF and combines them into one single DataFrame.
    """
    all_dataframes = []
    with pdfplumber.open(uploaded_file) as pdf:
        for page in pdf.pages:
            tables = page.extract_tables()
            for table in tables:
                if table:
                    df = pd.DataFrame(table[1:], columns=table[0])
                    all_dataframes.append(df)

    # Combine all DataFrames into one
    if all_dataframes:
        combined_df = pd.concat(all_dataframes, ignore_index=True)
        return combined_df
    else:
        return pd.DataFrame()  # return empty DF if no tables
