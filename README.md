# Repository for "Naming, Describing, and Quantifying Visual Objects in Humans and LLMs"

Paper link: https://arxiv.org/abs/2403.06935


## Installation

Install the required packages (preferrably through a virtual environment) through:

```
pip install -r requiements.txt
```

For installing LLaVA, please follow the instructions on the original github repository [here](https://github.com/haotian-liu/LLaVA). Do keep in mind that while LLaVA-v1.6 exists, this project was done using LLaVA-v1.5. To use the newer version, change the model name in main.py.

For FROMAGe, model weights can be obtained [here](https://github.com/kohjingyu/fromage).

## Usage

Runs can be performed by executing main.py with command line prompts. As an example, running on the NOUN dataset using BLIP2, with 3 samples per image and a top-p of 0.7 for nucleus sampling, use the following line:

```
python -m --model blip2 --dataset noun --top_p 0.7 --samples 3
```

Results are saved in the results folder under the respective model and dataset combination. 

The data can be further processed for analysis using process_data.py and plot_results.py.


## Acknowledgements

We would like to thank the authors of the [LLaVA](https://arxiv.org/abs/2310.03744), [BLIP-2](https://arxiv.org/abs/2301.12597) and [FRoMAGe](https://arxiv.org/abs/2301.13823) papers for their outstanding work on their respective models. The codebase for their models can be found [here](https://github.com/haotian-liu/LLaVA), [here](https://github.com/salesforce/LAVIS/tree/main/projects/blip2) and [here](https://github.com/kohjingyu/fromage).