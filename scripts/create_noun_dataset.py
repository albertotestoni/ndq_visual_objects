import os
import csv
from noun_data import true_name, fam_scores, name_scores, color_scores, texture_scores

folder_path = 'data/images/noun'

label = [i for i in range(2001, 2065)]

with open('../data/datasets/noun_dataset.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['image_path', 'number label', 'actual name', 'familiarity score', 'nameability score', 'color saliency', 'texture saliency'])
    for filename, label, true, fam, name, color, tex in zip(sorted(os.listdir(folder_path)), label, true_name, fam_scores, name_scores, color_scores, texture_scores):
        if filename.endswith('.jpg'):
            image_path = os.path.join(folder_path, filename)
            writer.writerow([image_path, label, true, fam, name, color, tex])
