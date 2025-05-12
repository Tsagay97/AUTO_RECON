from rapidfuzz import fuzz
import pandas as pd

def verify_policy_fuzzy(amount_matched_df: pd.DataFrame, policy_col_ricb: str, narration_col_bob: str, threshold: int = 85):
    verified_matches = []
    flagged_matches = []

    for _, row in amount_matched_df.iterrows():
        policy = str(row[policy_col_ricb])
        narration = str(row[narration_col_bob])
        score = fuzz.partial_ratio(policy, narration)

        result = row.to_dict()
        result['Fuzzy Score'] = score

        if score >= threshold:
            verified_matches.append(result)
        else:
            flagged_matches.append(result)

    verified_df = pd.DataFrame(verified_matches)
    flagged_df = pd.DataFrame(flagged_matches)

    return verified_df, flagged_df
