# FMC: Formalization of Natural Language Mathematical Competition Problems
*Jiaxuan Xie, Chengwu Liu, Ye Yuan, Siqi Li, Zhiping Xiao\*, Ming Zhang\**

This dataset is proposed in the AI for Math Workshop @ICML 2025 paper: [FMC: Formalization of Natural Language Mathematical Competition Problems](http://arxiv.org/abs/2507.11275).

<p align="center">
  📃 <a href="http://arxiv.org/abs/2507.11275" target="_blank">[Paper]</a> • 💻 <a href="https://github.com/JadeXie1205/FMC" target="_blank">[Github]</a> • 🤗 <a href="https://huggingface.co/datasets/JadeXie1205/FMC" target="_blank">[Dataset]</a> • 📋 <a href="https://github.com/JadeXie1205/FMC/blob/main/poster-final.pdf" target="_blank">[Poster]</a>
</p>

Efficient and accurate autoformalization methods, leveraging large-scale databases of natural language mathematical problems to construct formal language datasets, are key to advancing formal mathematical reasoning. In this paper, we propose an autoformalization pipeline based on large language models with error feedback, achieving a fully automatic and training-free formalization approach. Using this pipeline, we establish an Olympiad-level dataset aligning natural language problems with Lean formalizations. The dataset contains 3922 mathematical problems in natural language and 9787 in Lean, of which 64.46% received at least good quality ratings and above, making it suitable as a benchmark for automated theorem provers. Additionally, we investigate the formalization and reasoning capabilities of various general large language models and experimentally demonstrate that few-shot learning, error feedback, and increasing sampling numbers positively impact the autoformalization performance.

## Setup Environment
We recommend using Anaconda to create a new environment and install the required packages. You can create a new environment with the following command:
```bash
conda create -n fmc python=3.8
conda activate fmc
```

The following package versions have been tested and are recommended for compatibility: 

```txt
- torch==2.4.0
- transformers==4.46.3
- xformers==0.0.27.post2
- vllm==0.6.0
- openai==1.61.1
- backoff
```

## Run the Code

First, please add your OpenAI API key,  base URL and model name into the `main.py` file in the following format:
```python
api_key = "your_api_key"
base_url = "your_base_url"
model_name = "your_model_name"
```

Then you can run the inference code using the following command:

```bash
git clone https://github.com/JadeXie1205/FMC
cd FMC/autoformalization_pipeline
python main.py --question-file your_dataset.json --answer-root log
```

## Citation
```bibtex
@inproceedings{xie2025fmc,
	title={{FMC}: Formalization of Natural Language Mathematical Competition Problems},
	author={Jiaxuan Xie and Chengwu Liu and Ye Yuan and Siqi Li and Zhiping Xiao and Ming Zhang},
	booktitle={2nd AI for Math Workshop @ ICML 2025},
	year={2025},
	url={https://openreview.net/forum?id=7oRr1DQ8Gk}
}
```
