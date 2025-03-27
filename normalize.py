from sklearn.preprocessing import MinMaxScaler
import pandas as pd

# 这玩意在后续代码里似乎一点也没用到。

input_file = 'loan.csv'
output_file = 'loan_normal.csv'

with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', encoding='utf-8') as outfile:
    for line in infile:
        processed_line = [element.strip() for element in line.split(',')]

        for i in range(len(processed_line)):
            if processed_line[i] == 'Graduate':
                processed_line[i] = '1'
            elif processed_line[i] == 'Not Graduate':
                processed_line[i] = '0'

            if processed_line[i] == 'Yes':
                processed_line[i] = '1'
            elif processed_line[i] == 'No':
                processed_line[i] = '0'

        outfile.write(','.join(processed_line) + '\n')

df = pd.read_csv('loan_normal.csv')
scaler = MinMaxScaler()
df.iloc[:, 1:-1] = scaler.fit_transform(df.iloc[:, 1:-1]).round(4)

df.to_csv('loan_normal.csv', index=False)
