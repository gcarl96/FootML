import pandas as pd
from src.utils.config import Config
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

def preprocess_data(train_df, test_df, config: Config):
    # Preprocess data
    train_inputs = train_df[config.data_config.feature_list]
    test_inputs = test_df[config.data_config.feature_list]

    # Standardize data
    if config.model_config.model_name == 'logistic_regression':
        scaler = StandardScaler()
        train_inputs = scaler.fit_transform(train_inputs)
        test_inputs = scaler.transform(test_inputs)

    return train_inputs, test_inputs

def train_model(train_df, test_df, config: Config):
    train_inputs, test_inputs = preprocess_data(train_df, test_df, config)
    train_target = train_df[config.data_config.target_column]

    # Train model
    if config.model_config.model_name == 'logistic_regression':
        model = LogisticRegression(random_state=config.model_config.random_state)
        model.fit(train_inputs, train_target)
        
    elif config.model_config.model_name == 'xgboost':
        model = xgb.XGBClassifier(random_state=config.model_config.random_state)
        model.fit(train_inputs, train_target)

    # Make predictions
    test_pred_proba = model.predict_proba(test_inputs)

    return model, test_pred_proba


if __name__ == '__main__':
    config = Config()
    train_model(config)

