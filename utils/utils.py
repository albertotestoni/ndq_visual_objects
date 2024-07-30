import ast
import re
import os

import fnmatch
import nltk
from nltk import pos_tag, word_tokenize
from nltk.corpus import stopwords

nltk.download("punkt")
nltk.download("averaged_perceptron_tagger")
nltk.download("stopwords")

from collections import Counter
import argparse


def extract_nouns(df, model_name="blip2"):
    no_tokenize = True if model_name == "llava" else False
    noun_column = f'{model_name}_output'
    df[noun_column] = df[noun_column].apply(lambda x: sentence2noun(x, no_tokenize))
    df[f"{model_name}_extracted_nouns"] = df[noun_column].apply(
        lambda row: dict(Counter(row)))
    print(df[f"{model_name}_extracted_nouns"])
    df = df.drop(noun_column, axis=1)
    return df.copy()


def sentence2noun(sentence, no_tokenize=False):
    sentence = str(sentence)
    words = word_tokenize(sentence.lower())
    tagged_words = pos_tag(words)
    nouns = [
        re.sub(r"[^\w]", "", word)
        for word, pos in tagged_words
        if (("NN" in pos) or ("VB" in pos)) and ("]" not in word)
    ]
    if no_tokenize:
        sentence = ast.literal_eval(sentence)
        return [word[0].lower() for word in sentence]
    filtered_nouns = [noun for noun in nouns if noun != '']
    return filtered_nouns


def concatenate_human_nouns(df):
    human_columns = ["responses", "incorrect", "singletons"]
    df["human_nouns"] = df[human_columns].apply(
        lambda row: {
            key: value for d in row for key, value in ast.literal_eval(d).items()
        },
        axis=1,
    )
    df = df.drop(human_columns, axis=1)
    return df.copy()


def dict2vec(human_dict, model_dict):
    human_dict = ast.literal_eval(human_dict)
    one_hot_columns = set(list(human_dict.keys()) + list(model_dict.keys()))
    vec1 = {
        noun: human_dict.get(noun, 0) / sum(list(human_dict.values()))
        for noun in one_hot_columns
    }
    vec2 = {
        noun: model_dict.get(noun, 0) / sum(list(model_dict.values()))
        for noun in one_hot_columns
    }
    return vec1, vec2


def create_one_hot_vectors(df, model_name):
    df[["human_nouns", f"{model_name}_extracted_nouns"]] = df.apply(
        lambda row: dict2vec(row["human_nouns"], row[f"{model_name}_extracted_nouns"]),
        axis=1,
        result_type="expand",
    )
    return df.copy()

def argument_parser():
    parser = argparse.ArgumentParser(description='Process model inference arguments')
    
    parser.add_argument('--model', choices=['blip2', 'fromage', 'llava'], required=True,
                        help='Name of the model to use (choices: blip2, fromage, llava)')
    parser.add_argument('--dataset', choices=['quantifiers', 'noun', 'manynames'], required=True,
                        help='Name of the dataset to use (choices: quantifiers, noun, manynames)')
    parser.add_argument('--prompt', type=str, default=None,
                        help='Path to save the file')
    parser.add_argument('--top_p', type=float, default=0.8, help='Nucleus sampling top_p value')
    parser.add_argument('--samples', default=10, type=int, help='Amount of times we ask model to caption an image')
    parser.add_argument('--set', choices=['subset', 'ablation', 'full'], default='subset', help='Which dataset variant to use (choices: subset, ablation, full)')
    return parser.parse_args()

def find_files(directory, pattern):
    for root, dirs, files in os.walk(directory):
        for basename in files:
            if pattern in basename:
                filename = os.path.join(root, basename)
                yield filename