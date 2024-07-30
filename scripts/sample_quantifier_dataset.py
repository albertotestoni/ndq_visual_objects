import random
import pandas as pd
import os
random.seed(42)

# transform old dataset into proper dataframe

new_dataset = 'data/datasets/quantifiers.csv'

if not os.path.exists(new_dataset):
    dataset_path = 'data/datasets/quantifiers_unordered.csv'
    data = pd.read_csv(dataset_path)

    start_index = int(list(data['0Y'])[-2])
    end_index = int(list(data['X0'])[-1])
    ratio_columns = list(data.columns)[2:]
    quantifiers = list(data['quantifier'])[:9]


    indices_images = [{'index':i, 'image_file': f'data/images/quantifier/test_{i}.png'} for i in range(start_index, end_index + 1)]
    ratios = [ratio_columns[i // 200] for i in range(end_index + 1)]

    new_dataframe = pd.DataFrame(indices_images)
    new_dataframe['ratio'] = ratios

    for quantifier in quantifiers:
        ratios = []
        for idx, row in new_dataframe.iterrows():
            result = data.loc[data['quantifier'] == quantifier, row['ratio']].values[0]
            ratios.append(result)
        new_dataframe[quantifier] = ratios

    new_dataframe.to_csv(new_dataset, index=False)

# get subset of dataset
df = pd.read_csv(new_dataset)
df.rename({'image_file': 'image'}, axis=1, inplace=True)
df = df.groupby('ratio').sample(n=6, random_state=42)
df.iloc[:, :2].to_csv('data/datasets/quantifiers_ablation.csv', index=False)