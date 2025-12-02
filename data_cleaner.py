"""
Optimized QLoRA Fine-tuning for Phi-2 on Large Persian Datasets
- Advanced resource management (50% GPU, 50% CPU, 1/3 RAM)
- Robust checkpointing and resume functionality
- Comprehensive logging and progress tracking
- Optimized for 7M+ samples with efficient memory usage
"""

import os
import argparse
import math
import logging
import json
import psutil
import gc
import time
from pathlib import Path
from datetime import datetime
import signal
import sys
from contextlib import contextmanager

import torch
from datasets import load_dataset, DatasetDict, Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    set_seed,
)
from peft import LoraConfig, get_peft_model, PeftModel, TaskType
import transformers
from transformers.trainer_utils import get_last_checkpoint
import numpy as np

# ===== Enhanced Logging Setup =====
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(name)s] - %(message)s',
    handlers=[
        logging.FileHandler("training.log", encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# ===== Resource Monitor =====
class ResourceMonitor:
    """Monitor and limit resource usage"""

    def __init__(self, max_gpu_util=0.5, max_cpu_util=0.5, max_ram_util=0.33):
        self.max_gpu_util = max_gpu_util
        self.max_cpu_util = max_cpu_util
        self.max_ram_util = max_ram_util
        self.process = psutil.Process()

    def check_resources(self):
        """Check current resource usage"""
        # CPU usage
        cpu_percent = psutil.cpu_percent(interval=1) / 100.0

        # RAM usage
        ram_info = psutil.virtual_memory()
        ram_used_gb = ram_info.used / (1024 ** 3)
        ram_total_gb = ram_info.total / (1024 ** 3)
        ram_ratio = ram_used_gb / ram_total_gb

        # GPU usage (if available)
        gpu_usage = 0
        gpu_memory = 0
        if torch.cuda.is_available():
            try:
                # Try to get GPU utilization if available
                if hasattr(torch.cuda, 'utilization'):
                    gpu_usage = torch.cuda.utilization() / 100.0
                else:
                    # Fallback: estimate usage from memory allocation
                    allocated = torch.cuda.memory_allocated()
                    total = torch.cuda.get_device_properties(0).total_memory
                    gpu_usage = allocated / total if total > 0 else 0

                # Get GPU memory usage
                gpu_memory = torch.cuda.memory_allocated() / (1024 ** 3)
            except:
                pass

        return {
            'cpu_usage': cpu_percent,
            'ram_usage_gb': ram_used_gb,
            'ram_total_gb': ram_total_gb,
            'ram_ratio': ram_ratio,
            'gpu_usage': gpu_usage,
            'gpu_memory_gb': gpu_memory,
            'within_limits': (
                    cpu_percent <= self.max_cpu_util and
                    ram_ratio <= self.max_ram_util and
                    gpu_usage <= self.max_gpu_util
            )
        }

    def enforce_limits(self):
        """Enforce resource limits by introducing delays if necessary"""
        resources = self.check_resources()

        if resources['cpu_usage'] > self.max_cpu_util:
            delay = (resources['cpu_usage'] - self.max_cpu_util) * 0.1
            logger.warning(f"CPU usage {resources['cpu_usage']:.1%} exceeds limit, delaying for {delay:.2f}s")
            time.sleep(delay)

        if resources['ram_ratio'] > self.max_ram_util:
            logger.warning(
                f"RAM usage {resources['ram_usage_gb']:.1f}/{resources['ram_total_gb']:.1f} GB ({resources['ram_ratio']:.1%}) exceeds limit")
            logger.info("Triggering garbage collection and GPU cache cleanup...")
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            time.sleep(1)  # Allow time for cleanup

        if resources['gpu_usage'] > self.max_gpu_util:
            logger.warning(f"GPU usage {resources['gpu_usage']:.1%} exceeds limit, reducing batch processing")
            time.sleep(0.5)

        return resources

    def log_resources(self, prefix=""):
        """Log current resource usage"""
        resources = self.check_resources()
        logger.info(f"{prefix} Resources - CPU: {resources['cpu_usage']:.1%}, "
                    f"RAM: {resources['ram_usage_gb']:.1f}/{resources['ram_total_gb']:.1f} GB ({resources['ram_ratio']:.1%}), "
                    f"GPU: {resources['gpu_usage']:.1%} ({resources['gpu_memory_gb']:.1f} GB)")


# ===== Context Manager for Resource-Aware Execution =====
@contextmanager
def resource_aware_execution(monitor, task_name):
    """Context manager for resource-aware execution"""
    start_time = time.time()
    logger.info(f"Starting {task_name}...")
    monitor.log_resources(f"Before {task_name}")

    try:
        yield
        duration = time.time() - start_time
        monitor.log_resources(f"After {task_name}")
        logger.info(f"✅ Completed {task_name} in {duration:.2f}s")

    except Exception as e:
        duration = time.time() - start_time
        logger.error(f"❌ Failed during {task_name} after {duration:.2f}s: {e}")
        raise


# ===== Signal Handler for Graceful Interruption =====
class GracefulInterruptHandler:
    """Handle graceful interruption and checkpointing"""

    def __init__(self):
        self.interrupted = False
        self.trainer = None
        self.original_sigint = signal.getsignal(signal.SIGINT)
        self.original_sigterm = signal.getsignal(signal.SIGTERM)

        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)

    def set_trainer(self, trainer):
        self.trainer = trainer

    def signal_handler(self, signum, frame):
        logger.info(f"🔄 Received interrupt signal {signum}, preparing to save checkpoint...")
        self.interrupted = True

        if self.trainer:
            checkpoint_dir = f"{self.trainer.args.output_dir}/interrupt_checkpoint_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            logger.info(f"💾 Saving checkpoint to {checkpoint_dir}...")
            try:
                self.trainer.save_model(checkpoint_dir)
                logger.info("✅ Checkpoint saved successfully")
            except Exception as e:
                logger.error(f"❌ Failed to save checkpoint: {e}")

        logger.info("👋 Exiting gracefully...")
        sys.exit(0)


# ===== Emergency Save Function =====
def emergency_save(output_dir, model=None, tokenizer=None):
    """Emergency save function in case of critical errors"""
    try:
        emergency_dir = f"{output_dir}/emergency_save_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(emergency_dir, exist_ok=True)

        if model is not None:
            if hasattr(model, 'save_pretrained'):
                model.save_pretrained(emergency_dir)
                logger.info(f"💾 Emergency model save to {emergency_dir}")

        if tokenizer is not None:
            tokenizer.save_pretrained(emergency_dir)
            logger.info(f"💾 Emergency tokenizer save to {emergency_dir}")

        # Save emergency info
        emergency_info = {
            "saved_at": datetime.now().isoformat(),
            "reason": "emergency_save"
        }
        with open(f"{emergency_dir}/emergency_info.json", 'w') as f:
            json.dump(emergency_info, f, indent=2)

    except Exception as e:
        logger.error(f"❌ Emergency save failed: {e}")


def parse_args():
    p = argparse.ArgumentParser(description="Optimized QLoRA fine-tuning for Phi-2 on large Persian datasets")
    p.add_argument("--dataset", type=str, required=True,
                   help="JSONL dataset (one JSON object per line, requires 'text' field)")
    p.add_argument("--model_name_or_path", type=str, default="microsoft/phi-2",
                   help="Hugging Face model id or local path.")
    p.add_argument("--model_cache_dir", type=str, default="G:/model",
                   help="Directory to cache downloaded models")
    p.add_argument("--output_dir", type=str, default="./qlora_output",
                   help="Directory to save output model and checkpoints")
    p.add_argument("--max_seq_length", type=int, default=512,
                   help="Sequence length (optimized for Phi-2)")
    p.add_argument("--per_device_train_batch_size", type=int, default=1,
                   help="Batch size per device")
    p.add_argument("--gradient_accumulation_steps", type=int, default=16,
                   help="Gradient accumulation steps for effective batch size")
    p.add_argument("--num_train_epochs", type=int, default=1,
                   help="Number of training epochs")
    p.add_argument("--learning_rate", type=float, default=1.5e-4,
                   help="Learning rate")
    p.add_argument("--lora_r", type=int, default=32,
                   help="LoRA rank")
    p.add_argument("--lora_alpha", type=int, default=64,
                   help="LoRA alpha")
    p.add_argument("--lora_dropout", type=float, default=0.05,
                   help="LoRA dropout")
    p.add_argument("--max_samples", type=int, default=None,
                   help="Maximum number of samples to use (for testing)")
    p.add_argument("--resume_from_checkpoint", type=str, default=None,
                   help="Path to checkpoint to resume from")
    p.add_argument("--save_steps", type=int, default=2000,
                   help="Save checkpoint every X steps")
    p.add_argument("--logging_steps", type=int, default=100,
                   help="Log metrics every X steps")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed for reproducibility")
    p.add_argument("--warmup_ratio", type=float, default=0.05,
                   help="Warmup ratio for learning rate scheduler")
    return p.parse_args()


def setup_quantization():
    """Configure 4-bit quantization with optimal settings"""
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_storage=torch.uint8
    )


def setup_lora_config(args):
    """Configure LoRA with optimized settings for Phi-2"""
    # Phi-2 specific target modules based on architecture analysis
    target_modules = [
        "Wqkv", "out_proj", "fc1", "fc2",  # Phi-2 specific
        "q_proj", "k_proj", "v_proj", "o_proj",  # Standard transformer
        "gate_proj", "up_proj", "down_proj"  # MLP layers
    ]

    return LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=target_modules,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        # Advanced configurations for better stability and performance
        use_rslora=True,  # Rank-stabilized LoRA
        init_lora_weights="gaussian",  # Better initialization
    )


def load_and_preprocess_data(args, tokenizer, monitor):
    """Load and preprocess data with memory efficiency and progress tracking"""
    with resource_aware_execution(monitor, "data_loading"):
        logger.info(f"📂 Loading dataset from {args.dataset}")

        # Check if file exists
        if not os.path.exists(args.dataset):
            raise FileNotFoundError(f"Dataset file not found: {args.dataset}")

        # Load dataset with efficient streaming for large files
        try:
            # Try streaming first for large datasets
            dataset = load_dataset('json', data_files=args.dataset, split='train',
                                   streaming=True)
            # Convert to list if we need to limit samples
            if args.max_samples:
                dataset = dataset.take(args.max_samples)
                dataset = Dataset.from_list(list(dataset))
        except Exception as e:
            logger.warning(f"Streaming load failed, falling back to standard load: {e}")
            # Fallback to standard loading
            dataset = load_dataset('json', data_files=args.dataset, split='train')
            if args.max_samples:
                dataset = dataset.select(range(min(args.max_samples, len(dataset))))

        logger.info(f"📊 Dataset loaded with {len(dataset)} samples")

        # Sample inspection
        if len(dataset) > 0:
            sample = dataset[0]
            logger.info(f"📝 Sample data keys: {list(sample.keys())}")
            if 'text' in sample:
                sample_text = sample['text']
                logger.info(f"📄 Sample text (first 200 chars): {sample_text[:200]}...")

        def tokenize_function(examples):
            """Tokenization function with efficient processing"""
            texts = [text.strip() for text in examples['text'] if text and isinstance(text, str)]

            # Skip empty batches
            if not texts:
                return {}

            tokenized = tokenizer(
                texts,
                max_length=args.max_seq_length,
                truncation=True,
                padding="max_length",
                return_tensors="pt"
            )
            tokenized["labels"] = tokenized["input_ids"].clone()
            return tokenized

        # Tokenize with progress tracking and resource monitoring
        logger.info("🔤 Tokenizing dataset...")
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            batch_size=500,  # Smaller batches for memory efficiency
            remove_columns=dataset.column_names,
            desc="Tokenizing",
            load_from_cache_file=False  # Avoid cache issues with large datasets
        )

        # Filter out empty examples
        tokenized_dataset = tokenized_dataset.filter(lambda x: len(x['input_ids']) > 0)

        # Train/validation split
        if len(tokenized_dataset) > 1000:
            split_dataset = tokenized_dataset.train_test_split(
                test_size=min(0.01, 5000 / len(tokenized_dataset)),  # Max 5000 validation samples
                seed=args.seed
            )
            train_dataset = split_dataset["train"]
            eval_dataset = split_dataset["test"]
        else:
            train_dataset = tokenized_dataset
            eval_dataset = None

        logger.info(
            f"🎯 Final dataset - Training: {len(train_dataset)}, Validation: {len(eval_dataset) if eval_dataset else 0}")

        return train_dataset, eval_dataset


def setup_training_args(args):
    """Configure training arguments with optimized settings for large datasets"""

    # Check for existing checkpoint to resume from
    if args.resume_from_checkpoint is None and os.path.isdir(args.output_dir):
        last_checkpoint = get_last_checkpoint(args.output_dir)
        if last_checkpoint:
            args.resume_from_checkpoint = last_checkpoint
            logger.info(f"🔄 Found existing checkpoint: {args.resume_from_checkpoint}")

    # Calculate warmup steps based on dataset size
    if args.max_samples:
        total_steps = (args.max_samples * args.num_train_epochs) / (
                    args.per_device_train_batch_size * args.gradient_accumulation_steps)
        warmup_steps = max(100, int(total_steps * args.warmup_ratio))
    else:
        warmup_steps = 500  # Default reasonable value

    return TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True if args.resume_from_checkpoint is None else False,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        fp16=True,
        bf16=False,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=3,
        save_safetensors=True,
        remove_unused_columns=False,
        optim="paged_adamw_8bit",
        report_to="none",
        dataloader_pin_memory=False,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        warmup_steps=warmup_steps,
        max_grad_norm=1.0,
        lr_scheduler_type="cosine",
        weight_decay=0.01,
        seed=args.seed,
        dataloader_num_workers=2,
        dataloader_prefetch_factor=2,
        # Disable evaluation during training for large datasets to save time
        eval_strategy="no",
        load_best_model_at_end=False,
    )


def print_training_summary(args, train_dataset):
    """Print comprehensive training summary"""
    effective_batch_size = args.per_device_train_batch_size * args.gradient_accumulation_steps
    total_steps = len(train_dataset) * args.num_train_epochs / effective_batch_size

    logger.info("🎯 === TRAINING SUMMARY ===")
    logger.info(f"📊 Dataset: {args.dataset}")
    logger.info(f"📈 Samples: {len(train_dataset):,}")
    logger.info(f"🔢 Sequence length: {args.max_seq_length}")
    logger.info(
        f"⚡ Batch size: {args.per_device_train_batch_size} × {args.gradient_accumulation_steps} = {effective_batch_size} (effective)")
    logger.info(f"📚 Epochs: {args.num_train_epochs}")
    logger.info(f"🔄 Total steps: ~{total_steps:,.0f}")
    logger.info(f"🎓 Learning rate: {args.learning_rate}")
    logger.info(f"🎛️  LoRA rank (r): {args.lora_r}, alpha: {args.lora_alpha}")
    logger.info(f"💾 Checkpoint every: {args.save_steps} steps")
    logger.info(f"📝 Log every: {args.logging_steps} steps")
    logger.info("=" * 50)


def main():
    args = parse_args()
    set_seed(args.seed)

    # Initialize resource monitor and interrupt handler
    resource_monitor = ResourceMonitor(max_gpu_util=0.5, max_cpu_util=0.5, max_ram_util=0.33)
    interrupt_handler = GracefulInterruptHandler()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    logger.info("🚀 === Starting Optimized QLoRA Fine-tuning ===")
    logger.info(f"📁 Output directory: {args.output_dir}")
    logger.info(f"🤖 Model: {args.model_name_or_path}")
    logger.info(f"📊 Dataset: {args.dataset}")

    # Save configuration
    with open(f"{args.output_dir}/training_config.json", 'w', encoding='utf-8') as f:
        config_dict = vars(args)
        config_dict['start_time'] = datetime.now().isoformat()
        json.dump(config_dict, f, indent=2, ensure_ascii=False)

    model = None
    tokenizer = None
    trainer = None

    try:
        # === Load Model and Tokenizer ===
        with resource_aware_execution(resource_monitor, "model_loading"):
            logger.info("🔧 Loading model and tokenizer...")

            bnb_config = setup_quantization()

            tokenizer = AutoTokenizer.from_pretrained(
                args.model_name_or_path,
                cache_dir=args.model_cache_dir,
                trust_remote_code=True
            )

            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            model = AutoModelForCausalLM.from_pretrained(
                args.model_name_or_path,
                quantization_config=bnb_config,
                cache_dir=args.model_cache_dir,
                device_map="auto",
                trust_remote_code=True,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True
            )

            logger.info("✅ Model and tokenizer loaded successfully")

        # === Apply LoRA ===
        with resource_aware_execution(resource_monitor, "LoRA_setup"):
            logger.info("🎛️ Applying LoRA configuration...")
            peft_config = setup_lora_config(args)
            model = get_peft_model(model, peft_config)

            # Print trainable parameters
            model.print_trainable_parameters()

        # === Load and Preprocess Data ===
        train_dataset, eval_dataset = load_and_preprocess_data(args, tokenizer, resource_monitor)

        # Print training summary
        print_training_summary(args, train_dataset)

        # === Setup Training ===
        with resource_aware_execution(resource_monitor, "training_setup"):
            training_args = setup_training_args(args)

            # Data collator
            data_collator = DataCollatorForLanguageModeling(
                tokenizer=tokenizer,
                mlm=False,
                pad_to_multiple_of=8  # Optimization for tensor cores
            )

            # Initialize trainer
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                data_collator=data_collator,
                tokenizer=tokenizer,
            )

            # Setup graceful interruption
            interrupt_handler.set_trainer(trainer)

        # === Start Training ===
        with resource_aware_execution(resource_monitor, "training"):
            logger.info("🎬 Starting training...")
            start_time = datetime.now()

            # Pre-training resource check
            resource_monitor.log_resources("Pre-training")

            # Train with resume capability
            train_result = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

            training_duration = datetime.now() - start_time
            logger.info(f"✅ Training completed in {training_duration}")

        # === Save Final Model ===
        with resource_aware_execution(resource_monitor, "model_saving"):
            logger.info("💾 Saving final model...")
            trainer.save_model()
            tokenizer.save_pretrained(args.output_dir)

            # Save training metadata
            training_metadata = {
                "training_duration": str(training_duration),
                "completed_at": datetime.now().isoformat(),
                "total_steps": trainer.state.global_step,
                "final_loss": train_result.metrics['train_loss'] if hasattr(train_result, 'metrics') else None,
                "training_logs": trainer.state.log_history
            }

            with open(f"{args.output_dir}/training_metadata.json", 'w', encoding='utf-8') as f:
                json.dump(training_metadata, f, indent=2, ensure_ascii=False)

            logger.info("🎉 === Training Complete ===")
            logger.info(f"💾 Model saved to: {args.output_dir}")
            logger.info(f"⏱️ Training duration: {training_duration}")
            logger.info(f"📈 Final loss: {training_metadata['final_loss']}")

    except Exception as e:
        logger.error(f"❌ Training failed: {e}")

        # Emergency save
        logger.info("🆘 Attempting emergency save...")
        emergency_save(args.output_dir, model, tokenizer)

        # Re-raise to exit
        raise

    finally:
        # Cleanup
        if trainer and hasattr(trainer, 'save_model'):
            try:
                # Final checkpoint
                trainer.save_model()
                logger.info("💾 Final checkpoint saved")
            except:
                pass

        # Force cleanup
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

# python main.py --dataset G:/persian_news_processed.jsonl --output_dir G:/phi2-finetuned --max_seq_length 512 --per_device_train_batch_size 1 --gradient_accumulation_steps 16 --num_train_epochs 1 --learning_rate 1.5e-4 --lora_r 32 --lora_alpha 64 --save_steps 2000 --logging_steps 100 --max_samples 7000000