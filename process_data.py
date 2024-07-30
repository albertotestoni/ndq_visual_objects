import ast
import pandas as pd
from utils import filter_outputs, utils
from scipy import stats

def process_data(dataset_path, dataset, model):
    if dataset == 'noun':
        # noun dataset
        data = pd.read_csv(dataset_path)
        data[model + '_output'] = data[model + '_output'].apply(ast.literal_eval)
        data[model + '_output'] = data[model + '_output'].apply(lambda x: [item for sublist in x for item in sublist])
        data[model + '_saliencies'] = data[model + '_output'].apply(lambda lst: [filter_outputs.check_colors_and_textures(item) for item in lst])
        data[model + '_saliency_rate'] = data[model + '_saliencies'].apply(filter_outputs.calculate_saliency_rate)
        dataset_path = dataset_path.replace(".csv", "_saliency.csv")

    # manynames dataset
    if dataset == 'manynames':
        data = pd.read_csv(dataset_path)
        data = utils.extract_nouns(data, model_name=model)
        data = utils.create_one_hot_vectors(data, model_name=model)
        data['pearsonr'] = data.apply(lambda row: stats.pearsonr(list(row['human_nouns'].values()), list(row[f'{model}_extracted_nouns'].values()))[0], axis=1)
        dataset_path = dataset_path.replace(".csv", "_pearsonr.csv")

    # quantifiers dataset
    if dataset == 'quantifiers':
        data = pd.read_csv(dataset_path)
        data[f'{model}_filtered_output'] = data.apply(lambda row: filter_outputs.map_options(row[f'{model}_output'], row['shuffled_options'], row['prompted_questions'], model), axis=1)
        data = filter_outputs.sample_quantifiers(data, model, threshold=5)
        data = data.iloc[:,[0, 1, -1]]
        dataset_path = dataset_path.replace(".csv", "_filtered.csv")
    
    data.to_csv(dataset_path, index=False)
        
if __name__ == '__main__':
    model = 'blip2'
    dataset = 'quantifiers'
    # list containing strings to look for the csv files
    keywords = 'ablation'
    # directory where csv files are stored
    directory = f'results/{model}/{dataset}/dataframes/'
    
    for dataset_path in utils.find_files(directory, keywords):
        process_data(dataset_path, dataset, model)