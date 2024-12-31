import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

def time_based_split(df, date_column, n_splits=3):
    """
    Split the dataset into train, validation, and test sets based on time.
    """
    df = df.sort_values(by=date_column)
    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    for train_index, test_index in tscv.split(df):
        train_df = df.iloc[train_index]
        test_df = df.iloc[test_index]
    
    # The last split will be our final split
    train_df, val_df = train_df[:-len(test_df)], train_df[-len(test_df):]
    
    return train_df, val_df, test_df

if __name__ == '__main__':
    df = pd.read_csv('data/processed/feature_set.csv')
    train_df, val_df, test_df = time_based_split(df, 'match_date')
    
    train_df.to_csv('data/processed/train_set.csv', index=False)
    val_df.to_csv('data/processed/validation_set.csv', index=False)
    test_df.to_csv('data/processed/test_set.csv', index=False)
    
    print(f"Train set size: {len(train_df)}")
    print(f"Validation set size: {len(val_df)}")
    print(f"Test set size: {len(test_df)}")