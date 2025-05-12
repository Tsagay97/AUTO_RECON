import pandas as pd
import re

def extract_policy_id(narration: str) -> str:
    if not isinstance(narration, str):
        return None

    pattern = r'[A-Z]{2,}[A-Z0-9]*/\d{4}/[\dA-Z]+(?:/\d+)*'
    match = re.search(pattern, narration)
    if match:
        return match.group(0).strip()
    return None

def drop_useless_columns(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [str(col).strip() for col in df.columns]
    df = df.loc[:, ~df.columns.str.match(r'^(Unnamed.*|None|nan|\s*)$', case=False)]
    df = df.dropna(axis=1, how='all')
    df = df.loc[:, ~df.apply(lambda col: col.astype(str).str.strip().replace('-', '').eq('').all())]
    return df

def normalize(col: str) -> str:
    return (
        str(col)
        .replace('\n', ' ')
        .replace('\r', ' ')
        .replace('\t', ' ')
        .strip()
        .upper()
        .replace("  ", " ")
    )

def clean_bob_data(df: pd.DataFrame) -> pd.DataFrame:
    df = drop_useless_columns(df)
    columns_to_drop = ['TRAN DESC', 'DEBIT', 'BALANCE', 'TOTAL']
    normalized_cols = {col: normalize(col) for col in df.columns}
    drop_cols = [col for col, norm in normalized_cols.items() if norm in columns_to_drop]
    df = df.drop(columns=drop_cols, errors='ignore')

    if 'CREDIT' in df.columns:
        df['CREDIT'] = (
            df['CREDIT']
            .astype(str)
            .str.replace(",", "", regex=False)
            .str.strip()
            .replace("", None)
            .astype(float)
        )

    if 'NARRATION' in df.columns:
        df['EXTRACTED_POLICY'] = df['NARRATION'].apply(extract_policy_id)
        df = df.dropna(subset=['EXTRACTED_POLICY'])

    return df

def clean_ricb_data(df: pd.DataFrame) -> pd.DataFrame:
    df = drop_useless_columns(df)
    fuzzy_keys = ['TRANSACTION STATUS', 'DEPARTMENT', 'ERR LOG', 'JOURNAL NO']
    normalized_cols = {col: normalize(col) for col in df.columns}
    cols_to_drop = [col for col, norm in normalized_cols.items() if norm in fuzzy_keys]
    df = df.drop(columns=cols_to_drop, errors='ignore')

    for col in df.columns:
        if "DATE" in normalize(col):
            df[col] = df[col].astype(str).str.extract(r'(\d{2}/\d{2}/\d{4})')
            break

    if 'AMOUNT' in df.columns:
        df['AMOUNT'] = (
            df['AMOUNT']
            .astype(str)
            .str.replace(",", "", regex=False)
            .str.strip()
            .replace("", None)
            .astype(float)
        )

    return df
