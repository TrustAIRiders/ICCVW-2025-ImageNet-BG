import os
import numpy as np
import pandas as pd
import json

json_file_path = 'imagenet_class_index.json'
with open(json_file_path, 'r') as file:
    imagenet_class_index = json.load(file)

dfs = []

files_path = "./model_results/"
files = [os.path.join(files_path, f) for f in os.listdir(files_path) if f.endswith(".pickle")]
for file in files:
    model_name = file.split("/")[-1].split(".")[0]
    print(f"> {model_name}")
    
    df = pd.read_pickle(file)
    df["prediction"] = df["logits"].apply(np.argmax)
    df["model"] = model_name
    dfs.append(df)
    
df = pd.concat(dfs, ignore_index=True)

def calculate_accuracy(group):
    return (group['label'] == group['prediction']).sum() / group.shape[0]

grouped = df.groupby(['quality', 'type', 'background_real_type', 'background', 'label', 'model'])
df = grouped.apply(calculate_accuracy).reset_index(name='accuracy')

df['quality'] = df['quality'].replace({'bad': 'N/A', 'questionable': '*', 'ok': '**', 'good': '***', 'v_good': '****'})
df['background_real_type'] = df['background_real_type'].replace({'ah': 'approach_1', 'gah': 'approach_2', 'sau': 'approach_3'})

model_order = [
    'AlexNet', 'VGG19_BN', 'Inception_V3', 'ResNet152',
    'ShuffleNet_V2_X1_5', 'MobileNet_v3_L', 'RegNet_Y_800MF',
    'EfficientNet_B3', 'ConvNeXt_L', 'EfficientNet_V2_M',
    'ViT_L_16', 'MaxVit_T', 'Swin_V2_B'
]


def calculate_model_performance(df):
    mean_accuracy = df.groupby(['type', 'model']).accuracy.mean().unstack()
    mean_accuracy = mean_accuracy.reindex(['unaltered'] + [bg for bg in mean_accuracy.index if bg != 'unaltered'])

    delta_accuracy = mean_accuracy.subtract(mean_accuracy.loc['unaltered'], axis=1)
    
    for model in model_order:
        print("\n> {:<20s} |".format(model), end=' ')
        for type in ['unaltered', 'abstract', 'real']:
            if type in mean_accuracy[model]:
                print("{}: {:.01f} ∆ {:+.01f}  \t".format(type, mean_accuracy[model].loc[type]*100, delta_accuracy[model].loc[type]*100), end=' ')


calculate_model_performance(df[df['quality'].isin(['***', '****'])])

#calculate_model_performance(df)

#calculate_model_performance(df[df['quality'].isin(['****'])])

#calculate_model_performance(df[
#    (df['quality'].isin(['***', '****'])) &
#    ((df['background_real_type'] == 'approach_1') | (df['type'] == 'unaltered'))
#])

def get_extreme_classes(df):
    df_copy = df.copy()
    models = df_copy["model"].unique()

    df_copy['name'] = df_copy['model'].astype(str) + "_" + df_copy['type'].astype(str)
    df_copy['value'] = df_copy['accuracy']

    df_reduced = df_copy[['name', 'label', 'value']].reset_index(drop=True)

    pivot_df = df_reduced.pivot_table(index='name', columns='label', values='value')

    for model in models:
        pivot_df.loc[model + '_abstract'] -= pivot_df.loc[model + '_unaltered']
        pivot_df.loc[model + '_real'] -= pivot_df.loc[model + '_unaltered']
        pivot_df = pivot_df.drop(model + '_unaltered')

    mean_values = pivot_df.mean()
    sorted_labels = mean_values.sort_values().index

    def print_labels(labels, title):
        print(f"\n\n{title}")
        for label in labels:
            sub_df = df[df['label'] == label]
            HAQA = sub_df['quality'].unique()[0]

            pivot_sub_df = sub_df.pivot_table(index='type', values='accuracy')
            delta_accuracy = pivot_sub_df.subtract(pivot_sub_df.loc['unaltered'], axis=1)

            print("\n> {:<5s} {} {:<20s}:".format(HAQA, *imagenet_class_index[str(label)]), end=" ")            
            for t in ['unaltered', 'abstract', 'real']:
                print(f"{t}: {pivot_sub_df.loc[t].values[0] * 100:.01f} ∆ {delta_accuracy.loc[t].values[0] * 100:+.01f}  \t", end=' ')

    print_labels(sorted_labels[:10], title="Classes with drop:")
    print_labels(sorted_labels[-10:], title="Classes with improvement:")     
    print("\n\n -----")
       

df_modern_models = df[df["model"].isin(['EfficientNet_B3', 'ConvNeXt_L', 'EfficientNet_V2_M','ViT_L_16', 'MaxVit_T', 'Swin_V2_B'])]


get_extreme_classes(df_modern_models)
#get_extreme_classes(df)
#get_extreme_classes(df_modern_models[df_modern_models['quality'] == '****'])
#get_extreme_classes(df_modern_models[df_modern_models['quality'] == '***'])
#get_extreme_classes(df_modern_models[df_modern_models['quality'] == '**'])