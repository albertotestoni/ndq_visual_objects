import ast
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from scipy import stats
from tqdm import tqdm
import time



def mn_pearsonr(data):

    # Group by 'vg_domain' and sum the values
    grouped_df = data.groupby('vg_domain')['pearsonr'].mean().reset_index()

    # Plot using Plotly
    fig = px.bar(grouped_df, x='vg_domain', y='pearsonr',
                text='pearsonr', title=f'{model}: Average pearsonr correlation',
                labels={'pearsonr': 'Mean of Pearsonr correlation value', 'vg_domain': 'VG Domain'},
                color='vg_domain', height=800)

    # Add data labels
    fig.update_traces(texttemplate='%{text}', textposition='outside', width=0.3)

    fig.update_layout(xaxis_title='VG Domain', yaxis_title='Mean of Values',
                    barmode='group', showlegend=True, bargroupgap=0.01, width=900, yaxis=dict(range=[-1, 1]))

    # Show the plot
    fig.write_image(f'data/{model}_mn_pearsonr.png')

def saliency_pearsonr(data, saliency_type='color', model_name='llava'):
    
    idx = 0 if saliency_type == 'color' else 1
    
    data[f'model_{saliency_type}_saliency'] = data[f'{model_name}_saliency_rate'].apply(
        lambda x: ast.literal_eval(x)[idx]
    )

    # Scatter plot
    plt.scatter(data[f'{saliency_type} saliency'], data[f'model_{saliency_type}_saliency'])

    # Linear regression
    X = data[f'{saliency_type} saliency'].values.reshape(-1, 1)
    y = data[f'model_{saliency_type}_saliency'].values
    model = LinearRegression().fit(X, y)
    pearsonr_coef = stats.pearsonr(data[f"{saliency_type} saliency"].values, y)[0]
    data[f'{saliency_type}_pearsonr'] = pearsonr_coef
    # Get the slope (m) and intercept (b) of the line
    m = model.coef_[0]
    b = model.intercept_

    # Plot the regression line
    plt.plot(
        data[f"{saliency_type} saliency"],
        m * data[f"{saliency_type} saliency"] + b,
        color="red",
    )

    # Adding labels and title
    plt.xlabel(f'Human {saliency_type} saliency')
    plt.ylabel(f'Model {saliency_type} saliency')
    plt.title(f"{model_name} {saliency_type} saliency using top_p = 0.9, pearsonr = {pearsonr_coef}")
    # Display the plot
    plt.savefig(f'data/{model_name}_{saliency_type}_pearsonr_{time.strftime("%Y%m%d-%H%M%S")}.png')
    
def quantifier_freq_dist(data, model_name='blip2'):
    magic_number = 17 # total number of proportions
    indices = data['Index'].tolist()
    proportions = np.linspace(0, 1, magic_number)
    indice_mapping = {200 * (i + 1): proportions[i] for i in range(17)}
    x_values = []
    for idx in indices:
        for indice_map in indice_mapping.keys():
            if idx < indice_map:
                x_values.append(indice_mapping[indice_map])
                break
    dfs = []
    data['proportions'] = x_values
    for _, row in tqdm(data.iterrows()):
        proportion = row['proportions']
        categories = row[f'{model_name}_filtered_output']
        
        # Repeat the row for each category
        for category in categories:
            df_flat = pd.DataFrame({'proportions': [proportion], f'{model_name}_filtered_output': [category]})
            dfs.append(df_flat)

    print('concatenating.')
    # Concatenate all DataFrames into a single DataFrame
    df_concatenated = pd.concat(dfs, ignore_index=True)
    print('plotting...')
    # Create a grouped bar chart
    fig = px.histogram(df_concatenated, f'{model_name}_filtered_output', color=f'{model_name}_filtered_output', title='Frequency Distribution',
                    labels={f'{model_name}_filtered_output': 'Categories', 'count': 'Frequency'},
                    category_orders={f'{model_name}_filtered_output': df_concatenated[f'{model_name}_filtered_output'].unique()},
                    barmode='group')
        # Show the plot
    fig.write_image(f'data/{model}_quantifier_dist.png')

model = 'llava'
dataset = 'noun'
data = pd.read_csv(f'results/{model}/{dataset}/dataframes/{model}_noun_saliency.csv')
saliency_pearsonr(data, 'color',  model)

