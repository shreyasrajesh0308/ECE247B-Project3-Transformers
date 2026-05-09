# ECE247B Project 3: Language Modeling and Transformers

Starter code for Project 3 in UCLA ECE247B, Spring 2026. Students implement a baseline bigram language model and a small GPT-style decoder-only transformer on TinyStories.

This public repository intentionally contains only the student-facing starter files. It does not contain solution code.

## Project

You will:

1. Implement and train a `BigramLanguageModel`.
2. Implement causal self-attention, multi-head attention, a feed-forward layer, layer normalization, transformer layers, and `MiniGPT`.
3. Run the provided tests.
4. Compare generations from the bigram model, your MiniGPT model, and the provided pretrained MiniGPT weights.

The main notebook is `Language_Modelling.ipynb`. The implementation TODOs are in `model.py` and `train.py`.

## Installation

```bash
pip install -r requirements.txt
```

The project uses PyTorch, NumPy, tiktoken, wandb, and einops. Do not use high-level transformer libraries such as `transformers` or `torchtext` for the assignment implementation.

## Data

The `data/` folder contains preprocessed train and test token files for TinyStories using the GPT-2 tokenizer.

Source: TinyStories, Eldan and Li, 2023. Dataset card: https://huggingface.co/datasets/roneneldan/TinyStories

## Pretrained Weights

Pretrained model weights are not committed to this repository because they are large binary files.

Download the pretrained models here:

https://drive.google.com/file/d/1g09qUM9WibdfQVgkj6IAj8K2S3SGwc91/view?usp=sharing

After downloading, place the files in:

```text
pretrained_models/
```

Expected files:

```text
pretrained_models/bigram_tester.pt
pretrained_models/minigpt_tester.pt
pretrained_models/best_train_loss_checkpoint.pth
```

## Running

Open and follow `Language_Modelling.ipynb`, or run training from Python:

```python
from train import solver

solver(model_name="bigram")
solver(model_name="minigpt")
```

## References

- TinyStories: "How Small Can Language Models Be and Still Speak Coherent English?", Eldan and Li, 2023. https://arxiv.org/abs/2305.07759
- "Attention Is All You Need", Vaswani et al., 2017. https://arxiv.org/abs/1706.03762
- "Improving Language Understanding by Generative Pre-Training", Radford et al., 2018. https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf
- "Using the Output Embedding to Improve Language Models", Press and Wolf, 2016. https://arxiv.org/abs/1608.05859
