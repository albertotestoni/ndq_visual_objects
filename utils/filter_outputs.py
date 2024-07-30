import pandas as pd
import plotly.graph_objects as go
import numpy as np
import ast
import re
import random

colors = ["Red",    "Orange",    "Yellow",    "Green",    "Blue",    "Purple",
                  "Pink",    "Brown",    "Gray",    "Black",    "White",    "Beige",
                          "Turquoise",    "Teal",    "Magenta",    "Lavender",
                                  "Indigo",    "Maroon",    "Gold",    "Silver", 
                                        "Bronze",    "Copper",    "Olive",    "Navy",
                                            "Sky blue",
          "Cream",    "Peach",    "Rose",    "Fuchsia",    "Coral",    "Mint",
            "Chartreuse",    "Salmon",    "Sienna",    "Slate",    "Tan",
              "Crimson",    "Ivory",    "Khaki",    "Lilac",    "Mauve",
                "Mustard",    "Rust",    "Scarlet",    "Tangerine",
                "Vermilion",    "Violet",    "Wheat",    "Brick red",    "Caramel"]


textures = ["Smooth",    "Rough",    "Fuzzy",    "Soft", "Hard",
                  "Bumpy",    "Slick",    "Sticky",    "Grainy",
                      "Sandy",    "Slippery",    "Jagged",    "Sharp",
                        "Coarse",    "Silky",    "Velvety",    "Wet",
                        "Dry",    "Glossy",    "Matte",    "Sparkly",
                        "Metallic",    "Wooden",    "Leathery",
                        "Plastic",    "Rubber",    "Furry", "Woolly",    "Feathery",
                        "Smooth",    "Satin",    "Lace",    "Crochet",    "Knitted",
                        "Embroidered",    "Linen",    "Silk",    "Velvet",    "Suede",
                        "Corduroy",    "Denim",    "Felt",    "Tweed",    "Mesh",
            "Hairy",    "Crisp",    "Crumbly",    "Flaky",    "Puffy",    "Spongy", 
                "Crunchy",    "Chewy",    "Gummy",    "Slimy",    "Starchy",    "Syrupy",
                        "Icy",    "Rocky",    "Stony",    "Sandy",    "Peppery",    "Salty",
                            "Sour",    "Sweet",    "Tangy",    "Tart",    "Spicy", 
                                      "Herbaceous",    "Earthy",    "Mossy",    "Woody", 
                                "Smoky",    "Smokey",    "Rusty",    "Corroded",    "Weathered",
                            "Rugged",    "Smooth",    "Polished",    "Shiny",    "Gleaming",
                        "Dull",    "Muddy",    "Cloudy",    "Milky",    "Transparent",
                                "Translucent",    "Opaque"]

def check_colors_and_textures(string, colors=colors, textures=textures):
    """
    Function for checking if any colors or textures are present in the given string.

    Args:
        string (str): A string to check for the presence of colors or textures.

    Returns:
        A tuple containing two lists of strings:
        - The first list contains the colors present in the string.
        - The second list contains the textures present in the string.
    """



    # Initialize empty lists for colors and textures present in the string
    colors_present = []
    textures_present = []
    # Check if any of the colors are present in the string
    for color in colors:
        if color.lower() in string.lower():
            if (color.lower() != "white") | ((color.lower() == "white") & ("background" not in string.lower())):
                colors_present.append(color)
    # Check if any of the textures are present in the string
    for texture in textures:
        if texture.lower() in string.lower():
            textures_present.append(texture)
    # Return boolean values indicating whether any colors or textures are present,
    # as well as the list of colors and textures present in the string
    return colors_present, textures_present

def colors_to_boolean(string, colors=colors):
    for color in colors:
        if color.lower() in string.lower():
            return True
    return False

def textures_to_boolean(string, textures=textures):
    for texture in textures:
        if texture.lower() in string.lower():
            return True
    return False

def calculate_saliency_rate(tuple_of_lists):
    result = []
    for tup in list(zip(*tuple_of_lists)):
        empty_lists = sum(1 for sublist in tup if sublist)
        total = len(tup)
        result.append(int(empty_lists/total * 100))
    return tuple(result)

def find_option(s):
    pattern = r'(?<![a-zA-Z])[a-zA-Z](?![a-zA-Z])'
    matches = re.findall(pattern, s)
    return matches

def find_string_option(s, options):
    matches = []
    for option in options.values():
        pattern = r'\b' + re.escape(option) + r'\b'
        matches += re.findall(pattern, s)
    return matches

def filter_quantifiers(outputs, options, questions):
    filtered_output = []
    outputs = [item for sublist in outputs for item in sublist]
    for output, randomized_option, question in zip(outputs, options, questions):
        question_matching = re.findall(r'([A-Z]):\s*([^A-Z]*)', question)
        print('question matching:', question_matching)
        options_mapping = {key.lower(): value.strip() for key, value in question_matching}
        matches = find_option(output)
        if len(matches) == 1:
            print('matches ', matches)
            matched_option = matches[0].lower()
            if matched_option in options_mapping.keys():
                filtered_output.append(options_mapping[matched_option])

        else:
            matches = find_string_option(output, options_mapping)
            print('matches ', matches)
            if len(matches) == 1:
                filtered_output.append(matches[0])
    return filtered_output

def map_options(output, map_dict, questions, model):
    output = ast.literal_eval(output)
    map_dicts = ast.literal_eval(map_dict)
    questions = ast.literal_eval(questions)
    return filter_quantifiers(output, map_dicts, questions)
    # result = []
    # print(output)
    # for out, map_dict in zip(output, map_dicts):
    #     print(out)
    #     print(map_dict)
    #     result.append(map_dict[out.lower()])
    # return result

def sample_quantifiers(data, model, threshold = 5):
    # Calculate the minimum length
    min_length = int(data[f'{model}_filtered_output'].apply(len).min())

    num_below_min_length = data[f'{model}_filtered_output'].apply(lambda x: len(x) < threshold).sum()

    print(f'Minimum length for filtered prompts is {min_length}')
    print(f'Model returned {num_below_min_length} outputs below threshold {threshold}')
    print(f'Truncating to length {max(min_length, threshold)}')

    threshold = max(min_length, threshold)

    data = data[data[f'{model}_filtered_output'].apply(len) >= threshold]
    # Truncate the lists to the minimum length
    data[f'{model}_filtered_output'] = data[f'{model}_filtered_output'].apply(lambda x: random.sample(x, threshold))
    
    # Count the number of rows below the minimum length
    
    return data