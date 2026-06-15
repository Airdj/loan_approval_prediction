import pandas as pd
from pathlib import Path

cols_path = Path('src/utils/columns.csv')

def load_data(train_path='data/raw/train.csv',test_path='data/raw/test.csv',
              crd_df='data/raw/credit_risk_dataset.csv'):
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)
    crd_df = pd.read_csv(crd_df)

    crd_df['id'] = list(range(0, len(crd_df)))
    full_train_df = pd.concat([df_train, crd_df], axis=0).reset_index(drop=True)

    df_train = full_train_df.iloc[:80000,:]
    df_eval = full_train_df.iloc[80000:,:]

    if not cols_path.exists():
        cols = pd.DataFrame(df_test.columns)
        cols.to_csv('src/utils/columns.csv', index=False, header=False)

    return df_train, df_test, df_eval

