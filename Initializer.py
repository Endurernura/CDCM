import torch
import pandas as pd
# import os
from torch.utils.data import Dataset

# dir = os.path.dirname(os.path.abspath(__file__))
# 这个目录初始化没什么用，但是我就得把它在这放着。

class IsMatrix(Dataset):
    def __init__(self, data_folder, start, end):
        self.data = []
        for i in range(start, end+1):
            df = pd.read_csv(f'{data_folder}/{i}.csv')
            features = df.drop(['loan_id', 'loan_status'],axis=1
            ).values.reshape(11, 31, 31)
            # 这里reshape的意义是将数据转换为11个31*31的矩阵，每个矩阵表示一个特征。
            self.data.append(features)
        self.train_data = []
        self.test_data = []
        for features in self.data:
            train = features.copy()
            test = train[:, 15:16, 15:16].copy()
            train[:, 15:16, 15:16] = 0
            # test和train中的掩码在矩阵中的位置是(15, 15)，即第16行第16列的元素。
            # 掩码的现实意义待定。
            self.train_data.append((train, test.flatten()))

    def __len__(self):
        return len(self.train_data)

    def __getitem__(self, idx):
        return torch.tensor(self.train_data[idx][0], dtype=torch.float32), torch.tensor(self.train_data[idx][1], dtype=torch.float32)

# if __name__ == "__main__":
#     dataset = IsMatrix("data", 1, 100000)
#     print(dataset[0])