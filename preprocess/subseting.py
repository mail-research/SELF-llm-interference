import pandas as pd
from sklearn.utils import shuffle


if __name__=="__main__":
    df = pd.read_parquet("data/deepscaler/deepscaler_train.parquet")
    df = shuffle(df, random_state=1234).reset_index()
    df = df.head(512)
    df.to_parquet("data/deepscaler/deepscaler_prob.parquet")