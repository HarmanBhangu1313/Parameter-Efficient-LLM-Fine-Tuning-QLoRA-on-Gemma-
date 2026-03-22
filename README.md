# Parameter-Efficient LLM Fine-Tuning (QLoRA on Gemma)

## Overview

This project demonstrates parameter-efficient adaptation of a pretrained Large Language Model using **QLoRA-based fine-tuning techniques**.  
The objective is to enable supervised training of large transformer models under **low-resource compute constraints** by leveraging low-bit quantization and lightweight trainable adapters.

The implementation uses the HuggingFace ecosystem to build an end-to-end fine-tuning workflow including dataset preprocessing, tokenizer mapping, training configuration, and prompt-based generation evaluation.

---

## Motivation

Training or fully fine-tuning large language models requires significant GPU memory and compute.  
Parameter-efficient methods such as **LoRA and QLoRA** allow targeted adaptation of pretrained models by updating a small subset of trainable parameters while keeping the base model frozen.

This project explores:

- Efficient LLM adaptation using 4-bit quantization  
- Low memory training workflows  
- Practical supervised fine-tuning pipelines  
- Behavioural changes in generation after adaptation  

---

## Key Features

- Parameter efficient fine tuning using **QLoRA**
- 4-bit model quantization using **bitsandbytes**
- HuggingFace **Transformers + PEFT + TRL SFTTrainer** pipeline
- Custom dataset formatting and tokenization workflow
- Low-resource training configuration with gradient accumulation
- Prompt-based inference experiments after training

---

## Tech Stack

- Python  
- PyTorch  
- HuggingFace Transformers  
- PEFT (LoRA / QLoRA)  
- TRL Trainer  
- bitsandbytes quantization  
- HuggingFace Datasets  

---

## Workflow

1. Load pretrained Gemma model with 4-bit quantization  
2. Configure LoRA adapters for causal language modelling  
3. Preprocess structured text dataset with tokenizer mapping  
4. Perform supervised fine-tuning using SFTTrainer  
5. Generate responses using prompt-based inference for evaluation  

---

## Results & Observations

- Demonstrates successful adaptation of model generation behaviour under constrained compute settings  
- Shows feasibility of low resource LLM training workflows  
- Highlights practical considerations such as sequence formatting, optimizer configuration, and memory handling  

---

## Future Improvements

- Quantitative evaluation using BLEU / ROUGE / perplexity metrics  
- Comparison with base model performance  
- Domain-specific dataset expansion  
- Deployment via inference API  
- Integration with retrieval-augmented generation pipelines  

---

## How to Run

```bash
pip install transformers datasets peft trl bitsandbytes accelerate