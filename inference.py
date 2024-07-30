import ast
import random
from abc import ABC, abstractmethod
from io import BytesIO
from typing import Any

import pandas as pd
import requests
import torch

# llava imports
from llava.constants import (
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IMAGE_TOKEN,
    IMAGE_TOKEN_INDEX,
)
from llava.conversation import SeparatorStyle, conv_templates
from llava.mm_utils import (
    KeywordsStoppingCriteria,
    process_images,
    tokenizer_image_token,
)

from llava.utils import disable_torch_init
from PIL import Image

# blip2 imports
from transformers import AutoProcessor

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class InferenceClass(ABC):
    """Abstract base class for inference classes."""

    def __init__(
        self,
        model: Any,
        data_path: str,
        model_name: str = "blip2",
        dataset: str = "noun",
        image_path: str = "data/images",
        manual_inference=False,
    ):
        """
        Initialize the inference class.

        Args:
            model (Any): The model for inference.
            data_path (str): The path to the data.
            model_name (str): The name of the model.
            dataset (str): The dataset.
            image_path (str): The path to the images.
            manual_inference (bool): Whether manual inference is enabled.
        """
        self.data = pd.read_csv(data_path)
        self.image_path = image_path
        self.model_name = model_name
        self.dataset = dataset

        self.model = model
        self.images = self.load_images()
        if not manual_inference:
            self.get_columns()

    def get_image_from_url(self, url):
        """Load image from URL."""
        response = requests.get(url, timeout=10)
        img = Image.open(BytesIO(response.content))
        img = img.resize((224, 224))
        img = img.convert("RGB")
        return img

    def load_images(self):
        """Load images."""
        images_string = self.data["image"]
        return list(images_string.str.cat(sep=",").split(","))
        
        

    def save_results_to_csv(self, output_path):
        """Save results to CSV."""
        # Merge results with original data
        output_data = self.data.copy()
        output_data[f"{self.model_name}_output"] = self.results
        if self.dataset == "quantifiers":
            output_data["shuffled_options"] = self.shuffled_options
            output_data["prompted_questions"] = self.prompted_questions
        # Save to CSV
        output_data.to_csv(output_path, index=False)

    def get_columns(self):
        """Get columns."""
        self.results = (
            self.data[f"{self.model_name}_output"].apply(ast.literal_eval)
            if f"{self.model_name}_output" in self.data.columns
            else [[] for _ in range(len(self.data.index))]
        )
        if self.dataset == 'quantifiers':
            self.shuffled_options = (
                self.data["shuffled_options"].apply(ast.literal_eval)
                if "shuffled_options" in self.data.columns
                else [[] for _ in range(len(self.data.index))]
            )
            self.prompted_questions = (
                self.data["prompted_questions"].apply(ast.literal_eval)
                if "prompted_questions" in self.data.columns
                else [[] for _ in range(len(self.data.index))]
        )
        for idx, sublist in enumerate(self.results):
            if len(sublist) == 0:
                self.start_idx = idx
                break

    def get_prompt(self, prompt, samples, idx):
        """Get prompt."""
        if prompt != "quantifiers":
            return [prompt for _ in range(samples)]

        options = [
            "all",
            "almost all",
            "most",
            "few",
            "none",
            "the smaller part",
            "some",
            "many",
            "almost none",
        ]
        prompt_options = []
        prompt_questions = []
        for _ in range(samples):
            random.shuffle(options)
            prompt_options.append(
                {
                    "a": options[0],
                    "b": options[1],
                    "c": options[2],
                    "d": options[3],
                    "e": options[4],
                    "f": options[5],
                    "g": options[6],
                    "h": options[7],
                    "i": options[8],
                }
            )
            prompt_questions.append(
                f"Question: Carefully examine the image. Can you determine the proportion of animals present, compared to objects?"
                f"Please select the most accurate answer from the options below:\n"
                f"A: {options[0]} \nB: {options[1]} \nC: {options[2]} \n"
                f"D: {options[3]} \nE: {options[4]} \nF: {options[5]} \n"
                f"G: {options[6]} \nH: {options[7]} \nI: {options[8]} \n"
                f"Your selection is:"
            )
        self.shuffled_options[idx] = prompt_options
        self.prompted_questions[idx] = prompt_questions
        return prompt_questions

    @abstractmethod
    def get_model_outputs(self, prompts, samples, **kwargs) -> list:
        """Get model outputs."""
        return []

    def generate_text(
        self,
        img,
        question,
        row_idx,
        top_p=0.8,
        temperature=0.5,
        samples=10,
        download_image=False,
    ):
        """Generate text."""
        if 'http' in img:
            raw_image = self.get_image_from_url(img)
        else:
            raw_image = Image.open(img.replace("\\", "/")).convert("RGB")

        questions = self.get_prompt(question, samples, row_idx)

        prompts = [[raw_image, questions[i]] for i in range(samples)]
        self.results[row_idx] = self.get_model_outputs(
            prompts, samples, top_p=top_p, temperature=temperature
        )


class BLIP2InferenceClass(InferenceClass):
    """BLIP2 inference class."""

    def __init__(
        self,
        model,
        data_path: str,
        model_name: str = "blip2",
        dataset: str = "noun",
        image_path: str = "data/images",
    ):
        """Initialize the BLIP2 inference class."""
        super().__init__(model, data_path, model_name, dataset, image_path)
        self.processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")

    def get_model_outputs(self, prompts, samples, **kwargs) -> list:
        """Get model outputs."""
        images, questions = map(list, zip(*prompts))
        inputs = [
            self.processor(images, text=questions, return_tensors="pt").to(
                DEVICE, torch.float16
            )
        ]

        generated_ids = [
            self.model.generate(
                **input_, do_sample=True, top_p=kwargs['top_p'], max_new_tokens=20
            )
            for input_ in inputs
        ]
        generated_texts = [
            self.processor.batch_decode(generated_id, skip_special_tokens=True)
            for generated_id in generated_ids
        ]
        return [[text.strip().lower()] for text in generated_texts[0]]


class FromageInferenceClass(InferenceClass):
    """Fromage inference class."""

    def __init__(
        self,
        model,
        data_path: str,
        model_name: str = "fromage",
        dataset: str = "noun",
        image_path: str = "data/images",
    ):
        """Initialize the Fromage inference class."""
        super().__init__(model, data_path, model_name, dataset, image_path)

    def get_model_outputs(self, prompts, samples, **kwargs) -> list:
        """Get model outputs."""
        generated_output = [
            self.model.generate_for_images_and_texts(
                prompt, max_num_rets=0, num_words=32, top_p=kwargs['top_p'], temperature=0.5
            )
            for prompt in prompts
        ]
        return [[text[0].strip().lower()] for text in generated_output]


class LLaVAInferenceClass(InferenceClass):
    """LLaVA inference class."""

    def __init__(
        self,
        model,
        data_path: str,
        model_name: str = "llava",
        dataset: str = "noun",
        image_path: str = "data/images",
    ):
        """Initialize the LLaVA inference class."""
        super().__init__(model, data_path, model_name, dataset, image_path)
        disable_torch_init()
        self.tokenizer, self.model, self.image_processor, self.context_len = self.model
        self.conv_mode = "llava_v0"
        self.stop_str = None

    def get_model_outputs(self, prompts, samples, **kwargs) -> list:
        """Get model outputs."""
        # TODO: add argparser for hyperparameters
        top_p = kwargs['top_p']
        temperature = 0.5
        results = []
        images, questions = map(list, zip(*prompts))
        for image, question in zip(images, questions):
            image = process_images([image], self.image_processor, self.model.config).to(
                self.model.device, dtype=torch.float16
            )
            query, criteria = self.process_query(question)
            with torch.inference_mode():
                output_ids = self.model.generate(
                    query,
                    images=image,
                    do_sample=True if temperature > 0 else False,
                    temperature=temperature,
                    top_p=top_p,
                    num_beams=1,
                    max_new_tokens=20,
                    use_cache=True,
                    stopping_criteria=[criteria],
                )

            input_token_len = query.shape[1]
            outputs = self.tokenizer.batch_decode(
                output_ids[:, input_token_len:], skip_special_tokens=True
            )[0]
            outputs = outputs.strip()
            if outputs.endswith(self.stop_str):
                outputs = outputs[: -len(self.stop_str)]
            outputs = outputs.strip()
            results.append([outputs])
        return results

    def process_query(self, query):
        """Process query."""
        image_token_se = (
            DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
        )
        if self.model.config.mm_use_im_start_end:
            query = image_token_se + "\n" + query
        else:
            query = DEFAULT_IMAGE_TOKEN + "\n" + query

        conv = conv_templates[self.conv_mode].copy()
        conv.append_message(conv.roles[0], query)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        self.stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2

        input_ids = (
            tokenizer_image_token(
                prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
            )
            .unsqueeze(0)
            .cuda()
        )

        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(
            keywords, self.tokenizer, input_ids
        )

        return input_ids, stopping_criteria
