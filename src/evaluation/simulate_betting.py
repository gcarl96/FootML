import pandas as pd
import json
import os
from src.utils.config import Config

def simulate_betting(predictions_df, config: Config):
    # Initialize variables for simulation
    current_bankroll = config.betting_simulation_config.starting_bankroll
    bankroll_history = [current_bankroll]
    bets_placed = 0
    
    # Sort matches chronologically 
    predictions_df = predictions_df.sort_values('date', ascending=True)
    
    # Get model probabilities for home wins
    model_probs = predictions_df['Pred_Prob']
    
    # Iterate through each match
    for idx, row in predictions_df.iterrows():
        model_prob = model_probs[idx]
        market_prob = 1 / row['B365H']  # Convert odds to probability
        
        # Only bet if model probability exceeds market probability
        if model_prob > market_prob:
            # Calculate Kelly stake
            edge = (row['B365H'] - 1) * model_prob - (1 - model_prob)
            kelly_fraction = edge / (row['B365H'] - 1)
            
            # Apply Kelly fraction limit
            kelly_fraction = min(kelly_fraction,
                                  config.betting_simulation_config.max_kelly_fraction)
            
            # Calculate bet amount
            bet_amount = round(current_bankroll * kelly_fraction)
            
            # Update bankroll based on match result
            if row['result'] == 1:  # Home win
                current_bankroll += bet_amount * (row['B365H'] - 1)
            else:
                current_bankroll -= bet_amount
                
            bets_placed += 1

            print(f"Bet {bets_placed} - {row['home_team']} vs {row['away_team']} - {row['date']} - {row['result']} - {bet_amount} - {current_bankroll:.2f}")
            
        bankroll_history.append(current_bankroll)
        
    # Calculate betting metrics
    roi = (current_bankroll - config.betting_simulation_config.starting_bankroll) / \
          config.betting_simulation_config.starting_bankroll * 100
          
    results = {
        'final_bankroll': current_bankroll,
        'roi_percent': roi,
        'bets_placed': bets_placed,
        'bankroll_history': bankroll_history
    }
    
    return results

if __name__ == "__main__":

    # Find latest experiment folder
    experiment_folders = [f for f in os.listdir('experiments') if os.path.isdir(os.path.join('experiments', f))]
    latest_experiment = max(experiment_folders)
    experiment_path = os.path.join('experiments', latest_experiment)

    # Load config
    config = Config()

    # Load model predictions from predictions.csv
    predictions_df = pd.read_csv(os.path.join(experiment_path, 'predictions.csv'))
        
    # Run betting simulation
    betting_results = simulate_betting(predictions_df, config)
    
    print(f"Betting Simulation Results:")
    print(f"Final Bankroll: ${betting_results['final_bankroll']:.2f}")
    print(f"ROI: {betting_results['roi_percent']:.1f}%") 
    print(f"Bets Placed: {betting_results['bets_placed']}")