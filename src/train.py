""" Entrypoint for model training """
from models.trainer import Trainer
from torch.utils.data import DataLoader
from models.dataset import PremierLeagueDataset, PLDataModule
from models.modules import NeuralNet
from models.loss import JointPoissonLoss
from torch.optim.lr_scheduler import CosineAnnealingLR

from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.linear_model import PoissonRegressor
from sklearn.preprocessing import StandardScaler


from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn


import pandas as pd
import numpy as np
import torch



from utils import TRAIN_DATA_PATH, VAL_DATA_PATH, TEST_DATA_PATH, MODEL_PATH
from models.constants import PREM_COLS_TO_DROP, PREM_EVAL_LABELS

def main():
    # select device so pytorch functions can be parallelised if desired
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 
    loader_manager = PLDataModule(TRAIN_DATA_PATH, VAL_DATA_PATH, TEST_DATA_PATH, scale=True)
    train_loader = loader_manager.get_train_loader()
    val_loader = loader_manager.get_val_loader()


    # criterion = JointPoissonLoss()
    # criterion = torch.nn.PoissonNLLLoss(log_input=False, full=False, reduction='mean')

    criterion = torch.nn.MSELoss()
    learning_rate = 0.0005
    weight_decay = 0
    

    # need in dims, inter dims and out dims
    model = NeuralNet(29,64,2)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    # optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    # optimizer = torch.optim.LBFGS(model.parameters(), lr=learning_rate)

    num_epochs=100
    
    scheduler = CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-6)
    # scheduler= None

    trainer = Trainer(model, train_loader, val_loader, criterion, optimizer, device, scheduler=scheduler)

    trainer.train(num_epochs)
    trainer.save_model(MODEL_PATH)

    # need a way to validate how the model performs on test data in terms of actually predicting the correct outcome

def test_other(X_train, y_train):

    test_df = pd.read_csv(TEST_DATA_PATH)
    X = test_df.drop(columns=PREM_COLS_TO_DROP)

    y = test_df[PREM_EVAL_LABELS]

    X_test = X.to_numpy(dtype=np.float32)
    y_test = y.to_numpy(dtype=np.float32)


    mlp = MLPRegressor(hidden_layer_sizes=(64,64),
                       activation='relu',
                       max_iter=500,
                       random_state=42
                       )
    
    mlp.fit(X_train, y_train)

    y_pred = mlp.predict(X_test)

    # Compute absolute error per sample (home + away)
    abs_errors = np.abs(y_pred - y_test)  # shape = (num_samples, 2)
    avg_abs_error_per_team = np.mean(abs_errors, axis=0)  # average per column
    overall_mae = np.mean(abs_errors)  # overall average error

    # Compute RMSE for additional context
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    # Print results
    print("Predictions for first 20 samples:\n", y_pred[:20])
    print("\nAbsolute errors for first 20 samples:\n", abs_errors[:20])
    print(f"\nAverage absolute error per team: Home = {avg_abs_error_per_team[0]:.3f}, Away = {avg_abs_error_per_team[1]:.3f}")
    print(f"Overall MAE: {overall_mae:.3f}")
    print(f"RMSE: {rmse:.3f}")

    return y_pred, abs_errors


def get_data():
    df = pd.read_csv(TRAIN_DATA_PATH)
    X = df.drop(columns=PREM_COLS_TO_DROP)

    y = df[PREM_EVAL_LABELS]

    X_train = X.to_numpy(dtype=np.float32)
    y_train = y.to_numpy(dtype=np.float32)
    return X_train, y_train

def test_custom(X_train, y_train, X_test, y_test):
    # Convert to tensors
    X_tensor = torch.from_numpy(X_train).float()
    y_tensor = torch.from_numpy(y_train).float()

    X_test_tensor = torch.from_numpy(X_test).float()
    y_test_tensor = torch.from_numpy(y_test).float()

    # Create DataLoader for training
    dataset = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Simple MLP model
    model = nn.Sequential(
        nn.Linear(X_train.shape[1], 64),
        nn.ReLU(),
        nn.Linear(64, 2)
    )

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    for epoch in range(100):
        epoch_loss = 0.0
        for X_batch, y_batch in loader:
            optimizer.zero_grad()
            preds = model(X_batch)
            loss = criterion(preds, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * X_batch.size(0)
        
        epoch_loss /= len(loader.dataset)
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/100], Loss: {epoch_loss:.4f}")

    # Inference on test set
    model.eval()
    with torch.no_grad():
        test_preds = model(X_test_tensor).numpy()
        test_labels = y_test_tensor.numpy()

    # --- Metric reporting ---
    home_error = np.abs(test_preds[:, 0] - test_labels[:, 0])
    away_error = np.abs(test_preds[:, 1] - test_labels[:, 1])

    print(f"Average absolute error per team: Home = {home_error.mean():.3f}, Away = {away_error.mean():.3f}")
    
    overall_mae = np.mean(np.abs(test_preds - test_labels))
    rmse = np.sqrt(np.mean((test_preds - test_labels) ** 2))

    print(f"Overall MAE: {overall_mae:.3f}")
    print(f"Overall RMSE: {rmse:.3f}")

    return test_preds, test_labels

def get_test_data():
    df = pd.read_csv(TEST_DATA_PATH)
    X = df.drop(columns=PREM_COLS_TO_DROP)

    y = df[PREM_EVAL_LABELS]

    X_test = X.to_numpy(dtype=np.float32)
    y_test = y.to_numpy(dtype=np.float32)
    return X_test, y_test

def test_poisson(X_train, y_train):

    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)


    test_df = pd.read_csv(TEST_DATA_PATH)
    X_test = test_df.drop(columns=PREM_COLS_TO_DROP).to_numpy(dtype=np.float32)
    y_test = test_df[PREM_EVAL_LABELS].to_numpy(dtype=np.float32)


    X_test = scaler.transform(X_test)


    poisson_home = PoissonRegressor(alpha=1.0, max_iter=300)
    poisson_away = PoissonRegressor(alpha=1.0, max_iter=300)

    poisson_home.fit(X_train, y_train[:,0])
    poisson_away.fit(X_train, y_train[:,1])

    y_pred_home = poisson_home.predict(X_test)
    y_pred_away = poisson_away.predict(X_test)

    y_pred = np.column_stack([y_pred_home, y_pred_away])

   # Evaluate
    mae_home = mean_absolute_error(y_test[:, 0], y_pred_home)
    mae_away = mean_absolute_error(y_test[:, 1], y_pred_away)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    print(f"Average absolute error per team: Home = {mae_home:.3f}, Away = {mae_away:.3f}")
    print(f"Overall RMSE: {rmse:.3f}")


if __name__ == "__main__":


    X_train, y_train = get_data()

    X_test, y_test = get_test_data()

    # test_other(X_train, y_train)
    # test_poisson(X_train, y_train)

    preds, labels = test_custom(X_train, y_train, X_test, y_test)

    # print(preds[0:20])

    # main()