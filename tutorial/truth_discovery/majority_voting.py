import pandas as pd
in_path = "dataset//book.txt"
out_path = "result//majority_voting.txt"
df = pd.read_csv(in_path,sep='\t',names=["source","isbn","book","author"])

def majority_voting(df,key_col="isbn",answer_col="author"):
    df_mv = pd.DataFrame(columns=[key_col,answer_col])
    keys = df[key_col].unique()
    for key in keys:
        indices = df[key_col]==key
        candidates = df[indices][answer_col].unique()
        max_candidate = ""
        max_count=0
        for candidate in candidates:
            indices_support = (df[key_col]==key)&(df[answer_col]==candidate)
            count = sum(indices_support)
            if count>max_count:
                max_count = count
                max_candidate = candidate
        df_mv = df_mv.append({key_col:key,answer_col:max_candidate},ignore_index=True)
    return df_mv

df_mv = majority_voting(df)
df_mv.to_csv(out_path,sep='\t',index=False)