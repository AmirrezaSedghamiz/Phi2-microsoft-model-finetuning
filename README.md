# QLoRA Fine-Tuning of Phi-2 for Persian Language (Experimental)

This repository contains an **experimental exploration** of fine-tuning Microsoft’s Phi-2 (2.7B) language model using **QLoRA** techniques on large Persian (Farsi) text datasets.

The goal of this project was to:
- Explore the feasibility of adapting a compact LLM to Persian
- Experiment with resource-efficient training under hardware constraints
- Learn practical trade-offs in large-scale LLM fine-tuning

⚠️ **This project is research/experimental in nature.**
The results are mixed and the model is **not intended for production use**.

## Key Learnings

### What Worked
- Streaming large Persian datasets without exhausting memory
- Applying QLoRA to reduce trainable parameters
- Building a resumable training pipeline with checkpoints
- Handling Persian RTL text during evaluation

### What Didn’t Work Well
- Model quality gains were limited despite large datasets
- Enforcing strict Persian-only responses proved fragile
- Training stability required careful tuning and still varied
- Results did not meet production-quality expectations
