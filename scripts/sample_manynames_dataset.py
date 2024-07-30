import pandas as pd
from utils.utils import concatenate_human_nouns

dataset_path = 'data/datasets/manynames.tsv'
data = pd.read_csv(dataset_path, sep='\t')
data.rename({'link_mn': 'image'}, axis=1, inplace=True)
data = concatenate_human_nouns(data)
data = data[data['human_nouns'].apply(len) != 1]
data = data.groupby('vg_domain').sample(n=15, random_state=42)
data.to_csv('data/datasets/manynames_ablation.csv', index=False)