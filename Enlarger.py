import pandas as pd
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from sklearn.metrics import r2_score
from tqdm import tqdm
from Initializer import IsMatrix

class ConvDeconvModel(nn.Module):
    def __init__(self):
        super(ConvDeconvModel, self).__init__()
        self.convs = nn.ModuleList(
            [nn.Conv2d(1, 1, kernel_size=16, padding=0) for _ in range(11)])
        self.convtrans = nn.ModuleList([nn.ConvTranspose2d(
            1, 1, kernel_size=16, padding=0) for _ in range(11)])
        self.conv1d_3 = nn.ModuleList(
            [nn.Conv1d(1, 1, kernel_size=64) for _ in range(11)])
        self.conv1d_8 = nn.ModuleList(
            [nn.Conv1d(1, 1, kernel_size=64) for _ in range(11)])
        self.fc = nn.Linear(386, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        batch_size = x.size(0)
        conv_out = []
        deconv_out = []

        for i in range(11):
            # Shape: [batch_size, 1, 2, 2]
            conv_result = self.convs[i](x[:, i:i + 1, :, :])
            # Shape: [batch_size, 1, 4]
            conv_out.append(conv_result.reshape(batch_size, 1, -1))

            deconv_result = self.convtrans[i](
                x[:, i:i + 1, :, :])  
            # Shape: [batch_size, 1, 4, 4]
            # Extract middle 3x3, Shape: [batch_size, 1, 3, 3]
            # deconv_middle = deconv_result[:, :, 1:4, 1:4]
            deconv_middle = deconv_result[:, :, 15:31, 15:31]
            # Shape: [batch_size, 1, 9]
            deconv_out.append(deconv_middle.reshape(batch_size, 1, -1))

        conv_out = torch.cat(conv_out, dim=1)  # Shape: [batch_size, 4, 4]
        deconv_out = torch.cat(deconv_out, dim=1)  # Shape: [batch_size, 4, 9]

        conv_out_processed = []
        deconv_out_processed = []
        for i in range(11):
            conv_out_processed.append(self.conv1d_3[i](
                conv_out[:, i:i + 1, :]))  # Shape: [batch_size, 1, 3]
            deconv_out_processed.append(self.conv1d_8[i](
                deconv_out[:, i:i + 1, :]))  # Shape: [batch_size, 1, 8]

        # Shape: [batch_size, 4, 3]
        conv_out = torch.cat(conv_out_processed, dim=1)
        # Shape: [batch_size, 4, 8]
        deconv_out = torch.cat(deconv_out_processed, dim=1)

        # Shape: [batch_size, 4, 11]
        flat_features = torch.cat([conv_out, deconv_out], dim=2)

        output = self.fc(flat_features).reshape(
            batch_size, 11)  # Shape: [batch_size, 4, 1]

        output = self.sigmoid(output)
        # output[:, 1] = torch.round(output[:, 1])
        # output[:, 2] = torch.round(output[:, 2])
        return output

# def calculate_r2(dataloader, model):
#     all_true = []
#     all_pred = []

#     model.eval()
#     with torch.no_grad():
#         for data, target in tqdm(dataloader):
#             output = model(data)
#             all_pred.append(output.numpy())
#             all_true.append(target.numpy())

#     all_true = np.vstack(all_true)
#     all_pred = np.vstack(all_pred)

#     r2 = r2_score(all_true, all_pred)
#     return r2


def main(mode, file_path):
    # args
    if mode == 1:
        train_start = 1
        train_end = 7000
        val_start = 7001
        val_end = 9000
        test_start = 9001
        test_end = 10000
        counter = -1
        species = 1
    elif mode == 2:
        train_start = 10001
        train_end = 17000
        val_start = 17001
        val_end = 19000
        test_start = 19001
        test_end = 20000
        counter = -1001
        species = 2

    # Init
    model = ConvDeconvModel()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train Dataset
    print("Training...")
    dataset = IsMatrix('dataset/matrix', train_start, train_end)
    dataloader = DataLoader(dataset, batch_size=256, shuffle=True)

    # Training
    num_epochs = 50
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    for epoch in range(num_epochs):
        running_loss = 0.0
        with tqdm(dataloader, unit="its") as tepoch:
            tepoch.set_description(f"Epoch {epoch + 1}/{num_epochs}")
            for train_data, target_data in tepoch:
                train_data, target_data = train_data.to(
                    device), target_data.to(device)
                optimizer.zero_grad()
                output = model(train_data)
                loss = criterion(output, target_data)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                tepoch.set_postfix(loss=loss.item())
            print(
                f'Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(dataloader):.4f}')

    model = model.to('cpu')
    dataset = IsMatrix('dataset/matrix', train_start, train_end)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    # r2 = calculate_r2(dataloader, model)
    # print(f"Training complete. Training R2: {r2:.4f}\n")

    # # Validation
    # print("Validating...")
    # dataset = IsMatrix('dataset/matrix', val_start, val_end)
    # dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    # r2 = calculate_r2(dataloader, model)
    # print(f"Validation R2: {r2:.4f}\n")

    # Test Dataset
    print("Generating...")
    dataset = IsMatrix('dataset/matrix', test_start, test_end)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    df = pd.read_csv(file_path)

    # Testing
    model.eval()
    with torch.no_grad():
        for test_data, target_data in tqdm(dataloader):
            output = model(test_data)
            new_row = {
                'loan_id': counter,
                'no_of_dependents': round(output[0][0].item(), 4),
                'education': round(output[0][1].item()),
                'self_employed': round(output[0][2].item()),
                'income_annum': round(output[0][3].item(), 4),
                'loan_amount': round(output[0][4].item(), 4),
                'loan_term': round(output[0][5].item(), 4),
                'cibil_score': round(output[0][6].item(), 4),
                'residential_assets_value': round(output[0][7].item(), 4),
                'commercial_assets_value': round(output[0][8].item(), 4),
                'luxury_assets_value': round(output[0][9].item(), 4),
                'bank_asset_value': round(output[0][10].item(), 4),
                'loan_status': species
            }
            df = pd.concat([df, pd.DataFrame([new_row])])
            counter -= 1

    os.makedirs('dataset/generate', exist_ok=True)
    df.to_csv('dataset/generate/train.csv', index=False)
    print("Generate complete.")

    if mode == 1:
        print()


if __name__ == '__main__':
    # main(1, 'dataset/origin/train.csv')
    # main(2, 'dataset/generate/train.csv')

    # # shuffle
    # df = pd.read_csv('dataset/generate/train.csv')
    # df = pd.concat([
    #     df.iloc[:4269],
    #     df.iloc[4269:].sample(frac=1).reset_index(drop=True)
    # ])
    # df.to_csv('dataset/generate/train.csv', index=False)