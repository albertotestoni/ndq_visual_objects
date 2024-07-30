import sys
from tqdm import tqdm

import fromage.fromage
import fromage.fromage.models
from inference import BLIP2InferenceClass, FromageInferenceClass, LLaVAInferenceClass
from utils.utils import argument_parser

# blip2
from transformers import Blip2ForConditionalGeneration

# llava
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path

# fromage
import fromage

args = argument_parser()

possible_prompts = {
    "noun": "Q: What do you see in the image? \nA:",
    "manynames": "Q: Please name the object in the red box with the first name that comes to mind. \nA:",
    "quantifiers": "quantifiers",
}

PROMPT = args.prompt
if PROMPT is None:
    PROMPT = possible_prompts[args.dataset]


DATASET_PATH = f"data/datasets/{args.dataset}"

if args.set == "subset":
    DATASET_PATH += "_subset.csv"
elif args.set == "ablation":
    DATASET_PATH += "_ablation.csv"
else:
    DATASET_PATH += ".csv"

print(f'Loading from {DATASET_PATH}')

# remove newline and double dot, replace space with dash

filename_prompt = PROMPT.replace('? \\nA:', '').replace(" ", "-").replace(":", "").lower()

model_name = args.model
SAVE_INTERVAL = 10
SAVE_PATH = (
    f"results/{model_name}/{args.dataset}/dataframes/"
    f"{model_name}_{args.dataset}_{args.set}_"
    f"{'top_p=' + str(args.top_p)}_"
    f"{'samples=' + str(args.samples)}_"
    f"prompt={filename_prompt}.csv"
)

print(f"Saving to {SAVE_PATH}")

print("loading model...")

if model_name == "blip2":
    loaded_model = Blip2ForConditionalGeneration.from_pretrained(
        "Salesforce/blip2-opt-2.7b", load_in_8bit=True, device_map="auto"
    )
    inference = BLIP2InferenceClass(loaded_model, DATASET_PATH, dataset=args.dataset)
elif model_name == "llava":
    model_path = "liuhaotian/llava-v1.5-7b"
    base_model_name = get_model_name_from_path(model_path)
    loaded_model = load_pretrained_model(model_path, None, base_model_name)
    inference = LLaVAInferenceClass(loaded_model, DATASET_PATH, dataset=args.dataset)
elif model_name == "fromage":
    MODEL_DIR = "/fromage/fromage_model/"

    loaded_model = fromage.fromage.models.load_fromage(MODEL_DIR)
    inference = FromageInferenceClass(loaded_model, DATASET_PATH, dataset=args.dataset)

print(f'Performing inference with model class {inference}')

print(f"generating {args.samples} outputs per image, starting at image index {inference.start_idx}")
for i in tqdm(range(inference.start_idx, len(inference.images))):
    inference.generate_text(
        inference.images[i], PROMPT, i, top_p=args.top_p, samples=args.samples
    )
    if (i + 1) % SAVE_INTERVAL == 0:
        inference.save_results_to_csv(SAVE_PATH)

inference.save_results_to_csv(SAVE_PATH)
