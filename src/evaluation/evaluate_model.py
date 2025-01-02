import pandas as pd
from src.utils.config import Config
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, roc_curve, auc
import os

def evaluate_model(test_df, test_pred_probas, config: Config, experiment_dir="data/outputs/"):

    # Create evaluation dataframe
    eval_df = pd.DataFrame({
        'Season': test_df['season'],
        'Home_Team': test_df['home_team'],
        'Away_Team': test_df['away_team'],
        'Date': test_df['date'],
        'Actual_Result': test_df[config.data_config.target_column],
        'Predicted_Result': (test_pred_probas[:, 1] > 0.5).astype(int),
        'Predicted_Result_Prob': test_pred_probas[:, 1],
        'Home_Goals': test_df['home_goals'],
        'Away_Goals': test_df['away_goals'],
        'Home_xG': test_df['home_xG'],
        'Away_xG': test_df['away_xG'],
        'Bookmaker_Home_Odds': test_df['B365H'],
        'Bookmaker_Draw_Odds': test_df['B365D'],
        'Bookmaker_Away_Odds': test_df['B365A']
    })

    # Convert odds to probabilities
    eval_df['Bookmaker_Home_Prob'] = 1 / eval_df['Bookmaker_Home_Odds']
    eval_df['Bookmaker_Draw_Prob'] = 1 / eval_df['Bookmaker_Draw_Odds']
    eval_df['Bookmaker_Away_Prob'] = 1 / eval_df['Bookmaker_Away_Odds']

    eval_df['Bookmaker_Result'] = (eval_df['Bookmaker_Home_Prob'] > 0.5).astype(int)

    # Normalize probabilities to sum to 1
    prob_cols = ['Bookmaker_Home_Prob', 'Bookmaker_Draw_Prob', 'Bookmaker_Away_Prob']
    eval_df[prob_cols] = eval_df[prob_cols].div(eval_df[prob_cols].sum(axis=1), axis=0)

    # Calculate metrics
    accuracy = accuracy_score(eval_df['Actual_Result'], eval_df['Predicted_Result'])
    model_class_report = classification_report(eval_df['Actual_Result'], eval_df['Predicted_Result'], output_dict=True)
    bookmaker_class_report = classification_report(eval_df['Actual_Result'], eval_df['Bookmaker_Result'], output_dict=True)
    model_result_prob = eval_df['Predicted_Result_Prob'].mean()
    bookmaker_result_prob = eval_df['Bookmaker_Home_Prob'].mean()

    # Plot ROC curves for both model and bookmaker
    plt.figure(figsize=(10, 6))
    # Model ROC
    fpr_model, tpr_model, _ = roc_curve(eval_df['Actual_Result'], eval_df['Predicted_Result_Prob'])
    roc_auc_model = auc(fpr_model, tpr_model)
    plt.plot(fpr_model, tpr_model, color='darkorange', lw=2, 
            label=f'Model ROC (AUC = {roc_auc_model:.2f})')
    

    # Bookmaker ROC
    fpr_bookie, tpr_bookie, _ = roc_curve(eval_df['Actual_Result'], eval_df['Bookmaker_Home_Prob'])
    roc_auc_bookie = auc(fpr_bookie, tpr_bookie)
    plt.plot(fpr_bookie, tpr_bookie, color='green', lw=2,
            label=f'Bookmaker ROC (AUC = {roc_auc_bookie:.2f})')

    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves: Model vs Bookmaker')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(experiment_dir, 'roc_curves.png'))
    plt.close()

    # Return metrics dictionary
    metrics = {
        'accuracy': accuracy,
        'model_precision': model_class_report['1']['precision'],
        'model_recall': model_class_report['1']['recall'],
        'model_f1_score': model_class_report['1']['f1-score'],
        'model_support': model_class_report['1']['support'],
        'bookmaker_precision': bookmaker_class_report['1']['precision'],
        'bookmaker_recall': bookmaker_class_report['1']['recall'],
        'bookmaker_f1_score': bookmaker_class_report['1']['f1-score'],
        'bookmaker_support': bookmaker_class_report['1']['support'],
        'avg_model_prob': model_result_prob,
        'avg_bookmaker_prob': bookmaker_result_prob,
        'model_roc_auc': roc_auc_model,
        'bookmaker_roc_auc': roc_auc_bookie
    }

    return metrics


if __name__ == '__main__':
    config = Config()
    evaluate_model(config)