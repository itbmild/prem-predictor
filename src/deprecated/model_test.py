from models.modules import NeuralNet
from models.dataset import PLDataModule
from models.trainer import Trainer

from utils import TRAIN_DATA_PATH, VAL_DATA_PATH, TEST_DATA_PATH, MODEL_PATH
from models.constants import PREM_COLS_TO_DROP, PREM_EVAL_LABELS

import torch.nn as nn
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR

import numpy as np
import pandas as pd
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = NeuralNet(29, 64, 2)
    loader_manager = PLDataModule(TRAIN_DATA_PATH, VAL_DATA_PATH, TEST_DATA_PATH, scale=True)
    
    train_loader = loader_manager.get_train_loader()
    val_loader = loader_manager.get_val_loader()
    test_loader = loader_manager.get_test_loader()


    learning_rate = 0.0001
    weight_decay = 0
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    num_epochs=50
    scheduler = CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-6)
    

    trainer = Trainer(model, train_loader, val_loader, criterion, optimizer, device, scheduler=scheduler)

    trainer.train(num_epochs)
    trainer.save_model(MODEL_PATH)
    trained_model = trainer.get_model()



    # loss = evaluate(model, test_loader)

    # 1. Run evaluation
    # Pass the scaler from your loader_manager if it exists
    scaler = loader_manager.scaler if hasattr(loader_manager, 'scaler') else None
    preds, actuals = evaluate(model, test_loader, device, scaler=scaler)

    # 2. Create a DataFrame for easy viewing
    results_df = pd.DataFrame({
        'Actual_Home': actuals[:, 0],
        'Actual_Away': actuals[:, 1],
        'Pred_Home': preds[:, 0],
        'Pred_Away': preds[:, 1]
    })

    # 3. Save to CSV
    results_df.to_csv('test_predictions.csv', index=False)
    print("Results saved to test_predictions.csv")
    
    # 4. Quick sanity check: Average Predicted Goals
    print(f"Avg Predicted Home Goals: {results_df['Pred_Home'].mean():.2f}")
    print(f"Avg Actual Home Goals:    {results_df['Actual_Home'].mean():.2f}")

def evaluate(model, test_loader, device='cpu', scaler=None):
    model.eval()
    model.to(device)
    
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for features, targets in test_loader:
            features = features.to(device)
            
            outputs = model(features)
            
            # Move to CPU and convert to numpy
            all_preds.append(outputs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    # Stack all batches into large arrays
    predictions = np.vstack(all_preds)
    actuals = np.vstack(all_targets)

    # # If you used a scaler, inverse transform to get actual goal numbers
    # if scaler:
    #     # Assuming the scaler was fit on [home_goals, away_goals]
    #     predictions = scaler.inverse_transform(predictions)
    #     actuals = scaler.inverse_transform(actuals)
    
    return predictions, actuals



if __name__ == "__main__":
    main()