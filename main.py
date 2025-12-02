"""
Optimized QLoRA Fine-tuning for Phi-2 on Large Persian Datasets
- Fixed gradient computation issue
- Proper parameter training setup
- Streaming data loading to prevent memory overload
- Uses local model without downloading
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
from typing import Iterator, Dict, List, Optional, Any, Union

import torch
from torch.utils.data import IterableDataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    set_seed,
)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
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

    def __init__(self, max_gpu_util=0.8, max_cpu_util=0.8, max_ram_util=0.8):
        self.max_gpu_util = max_gpu_util
        self.max_cpu_util = max_cpu_util
        self.max_ram_util = max_ram_util
        self.process = psutil.Process()

    def check_resources(self):
        """Check current resource usage"""
        # CPU usage
        cpu_percent = psutil.cpu_percent(interval=0.1) / 100.0

        # RAM usage
        ram_info = psutil.virtual_memory()
        ram_used_gb = ram_info.used / (1024**3)
        ram_total_gb = ram_info.total / (1024**3)
        ram_ratio = ram_info.used / ram_info.total

        # GPU usage (if available)
        gpu_usage = 0
        gpu_memory_used = 0
        gpu_memory_total = 0
        if torch.cuda.is_available():
            try:
                gpu_usage = torch.cuda.utilization() / 100.0 if hasattr(torch.cuda, 'utilization') else 0
                gpu_memory_used = torch.cuda.memory_allocated() / (1024**3)
                gpu_memory_total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            except Exception as e:
                logger.debug(f"GPU monitoring error: {e}")

        return {
            'cpu_usage': cpu_percent,
            'ram_usage_gb': ram_used_gb,
            'ram_total_gb': ram_total_gb,
            'ram_ratio': ram_ratio,
            'gpu_usage': gpu_usage,
            'gpu_memory_used_gb': gpu_memory_used,
            'gpu_memory_total_gb': gpu_memory_total,
        }

    def log_resources(self, prefix=""):
        """Log current resource usage"""
        resources = self.check_resources()
        logger.info(f"{prefix} Resources - CPU: {resources['cpu_usage']:.1%}, "
                   f"RAM: {resources['ram_usage_gb']:.1f}/{resources['ram_total_gb']:.1f} GB ({resources['ram_ratio']:.1%}), "
                   f"GPU: {resources['gpu_usage']:.1%} ({resources['gpu_memory_used_gb']:.1f}/{resources['gpu_memory_total_gb']:.1f} GB)")

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
        if self.interrupted:
            logger.info("Forcing immediate exit...")
            sys.exit(1)

        self.interrupted = True
        logger.info(f"🔄 Received interrupt signal {signum}, preparing to save checkpoint...")

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

    def __del__(self):
        # Restore original signal handlers
        signal.signal(signal.SIGINT, self.original_sigint)
        signal.signal(signal.SIGTERM, self.original_sigterm)

# ===== Emergency Save Function =====
def emergency_save(output_dir, model=None, tokenizer=None):
    """Emergency save function in case of critical errors"""
    try:
        emergency_dir = f"{output_dir}/emergency_save_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(emergency_dir, exist_ok=True)

        if model is not None:
            if hasattr(model, 'save_pretrained'):
                model.save_pretrained(emergency_dir, safe_serialization=True)
                logger.info(f"💾 Emergency model save to {emergency_dir}")

        if tokenizer is not None:
            tokenizer.save_pretrained(emergency_dir)
            logger.info(f"💾 Emergency tokenizer save to {emergency_dir}")

        emergency_info = {
            "timestamp": datetime.now().isoformat(),
            "saved_model": model is not None,
            "saved_tokenizer": tokenizer is not None
        }

        with open(f"{emergency_dir}/emergency_info.json", 'w') as f:
            json.dump(emergency_info, f, indent=2)

    except Exception as e:
        logger.error(f"❌ Emergency save failed: {e}")

# ===== Fixed Streaming Dataset =====
class PersianJSONLStreamingDataset(IterableDataset):
    """Fixed streaming dataset that properly handles tensor formatting"""

    def __init__(self, file_path: str, tokenizer, max_seq_length: int,
                 max_samples: Optional[int] = None, shuffle_buffer_size: int = 10000,
                 text_field: str = 'text'):
        self.file_path = file_path
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.max_samples = max_samples
        self.shuffle_buffer_size = shuffle_buffer_size
        self.text_field = text_field
        self._total_samples = None
        self._epoch = 0

    def set_epoch(self, epoch):
        """Set epoch for distributed training"""
        self._epoch = epoch

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        """Iterate through dataset, yielding properly formatted examples"""
        samples_processed = 0
        buffer = []
        rng = np.random.RandomState(42 + self._epoch)

        with open(self.file_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line_num, line in enumerate(f):
                if self.max_samples and samples_processed >= self.max_samples:
                    break

                line = line.strip()
                if not line:
                    continue

                try:
                    data = json.loads(line)
                    text_content = self._extract_text(data)
                    if text_content and len(text_content.strip()) > 20:
                        buffer.append(text_content)

                        if len(buffer) >= self.shuffle_buffer_size:
                            yield from self._process_buffer(buffer, rng)
                            samples_processed += len(buffer)
                            buffer = []

                except (json.JSONDecodeError, TypeError, ValueError, KeyError) as e:
                    if samples_processed % 1000 == 0:
                        logger.debug(f"Skipping malformed line {line_num}: {e}")
                    continue

        if buffer:
            yield from self._process_buffer(buffer, rng)
            samples_processed += len(buffer)

        logger.info(f"Processed {samples_processed} samples in epoch {self._epoch}")

    def _extract_text(self, data: Dict[str, Any]) -> Optional[str]:
        """Extract text from various possible field names"""
        if self.text_field in data and data[self.text_field]:
            text = str(data[self.text_field]).strip()
            if self._is_valid_persian_text(text):
                return text

        text_fields = ['text', 'content', 'article', 'body', 'paragraph', 'document', 'main_text']

        for field in text_fields:
            if field in data and data[field]:
                text = str(data[field]).strip()
                if self._is_valid_persian_text(text):
                    return text

        for key, value in data.items():
            if isinstance(value, str) and self._is_valid_persian_text(value):
                return value.strip()

        return None

    def _is_valid_persian_text(self, text: str) -> bool:
        """Check if text contains valid Persian content"""
        if not text or len(text.strip()) < 25:
            return False

        persian_chars = set('ابپتثجچحخدذرزژسشصضطظعغفقکگلمنوهی')
        text_chars = set(text)

        if persian_chars.intersection(text_chars):
            return True

        if len(text) > 100 and not text.startswith(('{', '[', '<')) and 'http' not in text:
            return True

        return False

    def _process_buffer(self, buffer: List[str], rng: np.random.RandomState) -> Iterator[Dict[str, torch.Tensor]]:
        """Process a buffer of text samples and yield tokenized examples"""
        if not buffer:
            return

        indices = np.arange(len(buffer))
        rng.shuffle(indices)
        shuffled_buffer = [buffer[i] for i in indices]

        batch_size = min(32, len(shuffled_buffer))

        for i in range(0, len(shuffled_buffer), batch_size):
            batch_texts = shuffled_buffer[i:i + batch_size]

            try:
                tokenized = self.tokenizer(
                    batch_texts,
                    max_length=self.max_seq_length,
                    truncation=True,
                    padding="max_length",
                    return_tensors="pt",
                    return_attention_mask=True,
                    add_special_tokens=True
                )

                for j in range(len(tokenized['input_ids'])):
                    yield {
                        'input_ids': tokenized['input_ids'][j].clone().detach(),
                        'attention_mask': tokenized['attention_mask'][j].clone().detach(),
                        'labels': tokenized['input_ids'][j].clone().detach()
                    }

            except Exception as e:
                logger.warning(f"Tokenization error, skipping batch: {e}")
                continue

    def _count_samples(self) -> int:
        """Count total valid samples in file, stopping at max_samples if specified"""
        count = 0
        try:
            with open(self.file_path, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    if self.max_samples and count >= self.max_samples:
                        logger.info(f"Reached max_samples ({self.max_samples}), stopping count")
                        break

                    line = line.strip()
                    if line:
                        try:
                            data = json.loads(line)
                            if self._extract_text(data):
                                count += 1
                        except:
                            continue

                    if count % 100000 == 0 and count > 0:
                        logger.info(f"Counting samples: {count}...")

        except Exception as e:
            logger.error(f"Error counting samples: {e}")

        return count

    def __len__(self):
        """Return approximate length for progress tracking"""
        if self._total_samples is None:
            self._total_samples = self._count_samples()

        if self.max_samples:
            return min(self._total_samples, self.max_samples)
        return self._total_samples if self._total_samples > 0 else 1000000

# ===== Custom Data Collator =====
class CustomDataCollatorForLanguageModeling(DataCollatorForLanguageModeling):
    """Custom data collator that handles our streaming dataset format"""

    def __call__(self, features: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        processed_features = []
        for feature in features:
            processed_feature = {}
            for key, value in feature.items():
                if hasattr(value, 'clone'):
                    processed_feature[key] = value.clone().detach()
                else:
                    processed_feature[key] = torch.tensor(value)
            processed_features.append(processed_feature)

        return super().__call__(processed_features)

def parse_args():
    p = argparse.ArgumentParser(description="Fixed QLoRA fine-tuning for Phi-2 on large Persian datasets")
    p.add_argument("--dataset", type=str, required=True,
                   help="JSONL dataset (one JSON object per line)")
    p.add_argument("--local_model_path", type=str,
                   default="G:/model/microsoft--phi-2/models--microsoft--phi-2/snapshots/ef382358ec9e382308935a992d908de099b64c23",
                   help="Local path to the base model")
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
    p.add_argument("--learning_rate", type=float, default=2e-4,
                   help="Learning rate")
    p.add_argument("--lora_r", type=int, default=16,
                   help="LoRA rank")
    p.add_argument("--lora_alpha", type=int, default=32,
                   help="LoRA alpha")
    p.add_argument("--lora_dropout", type=float, default=0.05,
                   help="LoRA dropout")
    p.add_argument("--max_samples", type=int, default=None,
                   help="Maximum number of samples to use (for testing)")
    p.add_argument("--resume_from_checkpoint", type=str, default=None,
                   help="Path to checkpoint to resume from")
    p.add_argument("--save_steps", type=int, default=500,
                   help="Save checkpoint every X steps")
    p.add_argument("--logging_steps", type=int, default=50,
                   help="Log metrics every X steps")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed for reproducibility")
    p.add_argument("--shuffle_buffer_size", type=int, default=5000,
                   help="Buffer size for shuffling streaming data")
    p.add_argument("--text_field", type=str, default="text",
                   help="Field name containing text in JSONL")
    p.add_argument("--warmup_steps", type=int, default=100,
                   help="Warmup steps for learning rate")
    p.add_argument("--max_grad_norm", type=float, default=0.3,
                   help="Maximum gradient norm for clipping")

    return p.parse_args()

def setup_quantization():
    """Configure 4-bit quantization with optimal settings"""
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

def setup_lora_config(args):
    """Configure LoRA with optimized settings for Phi-2"""
    target_modules = [
        "q_proj", "k_proj", "v_proj", "dense",
        "fc1", "fc2", "Wqkv", "out_proj"
    ]

    return LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=target_modules,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

def load_and_prepare_model(local_model_path: str, bnb_config: BitsAndBytesConfig) -> tuple:
    """Load model and tokenizer from local path and prepare for QLoRA training"""
    logger.info(f"🔧 Loading model from local path: {local_model_path}")

    if not os.path.exists(local_model_path):
        raise FileNotFoundError(f"Local model path not found: {local_model_path}")

    # Load tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            local_model_path,
            trust_remote_code=True,
            local_files_only=True
        )

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            logger.info("Set pad_token to eos_token")

        tokenizer.padding_side = "right"

    except Exception as e:
        logger.error(f"Failed to load tokenizer: {e}")
        raise

    # Load model
    try:
        model = AutoModelForCausalLM.from_pretrained(
            local_model_path,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            local_files_only=True
        )

        # CRITICAL FIX: Prepare model for k-bit training
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

        # Verify model is ready for training
        model.config.use_cache = False

        # Check if any parameters require gradients
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Trainable parameters: {trainable_params:,}")

        if trainable_params == 0:
            logger.warning("❌ No trainable parameters found! Model won't learn.")

    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise

    logger.info("✅ Model and tokenizer loaded and prepared successfully")
    return model, tokenizer

def create_streaming_dataset(args, tokenizer, monitor) -> IterableDataset:
    """Create optimized streaming dataset"""
    with resource_aware_execution(monitor, "streaming_dataset_setup"):
        logger.info(f"📂 Creating streaming dataset from {args.dataset}")

        if not os.path.exists(args.dataset):
            raise FileNotFoundError(f"Dataset file not found: {args.dataset}")

        dataset = PersianJSONLStreamingDataset(
            file_path=args.dataset,
            tokenizer=tokenizer,
            max_seq_length=args.max_seq_length,
            max_samples=args.max_samples,
            shuffle_buffer_size=args.shuffle_buffer_size,
            text_field=args.text_field
        )

        logger.info("🎯 Streaming dataset created successfully")
        return dataset

def setup_training_args(args, dataset_length: Optional[int] = None) -> TrainingArguments:
    """Configure training arguments with optimized settings"""

    # Calculate training steps
    max_steps = None
    if args.max_samples and dataset_length:
        effective_batch_size = args.per_device_train_batch_size * args.gradient_accumulation_steps
        samples_per_epoch = min(dataset_length, args.max_samples)
        max_steps = (samples_per_epoch * args.num_train_epochs) // effective_batch_size
        logger.info(f"📈 Calculated max_steps: {max_steps:,}")

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True if not args.resume_from_checkpoint else False,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        max_steps=max_steps,
        learning_rate=args.learning_rate,
        fp16=False,
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=2,
        save_safetensors=True,
        remove_unused_columns=False,
        optim="paged_adamw_8bit",
        report_to=[],
        dataloader_pin_memory=False,
        gradient_checkpointing=True,
        warmup_steps=args.warmup_steps,
        max_grad_norm=args.max_grad_norm,
        lr_scheduler_type="cosine",
        weight_decay=0.01,
        seed=args.seed,
        dataloader_num_workers=0,
        evaluation_strategy="no",
        load_best_model_at_end=False,
        disable_tqdm=False,
        dataloader_drop_last=True,
        prediction_loss_only=True,
        group_by_length=False,
    )

    return training_args

def print_training_summary(args, dataset_length: int, model) -> None:
    """Print comprehensive training summary"""
    effective_batch_size = args.per_device_train_batch_size * args.gradient_accumulation_steps

    if dataset_length and args.max_samples:
        total_samples = min(dataset_length, args.max_samples)
    else:
        total_samples = dataset_length or args.max_samples or "unknown"

    # Count trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())

    logger.info("🎯 === TRAINING SUMMARY ===")
    logger.info(f"📊 Dataset: {args.dataset}")
    logger.info(f"📈 Estimated samples: {total_samples}")
    logger.info(f"🔢 Sequence length: {args.max_seq_length}")
    logger.info(f"⚡ Batch size: {args.per_device_train_batch_size} × {args.gradient_accumulation_steps} = {effective_batch_size} (effective)")
    logger.info(f"📚 Epochs: {args.num_train_epochs}")
    logger.info(f"🎓 Learning rate: {args.learning_rate}")
    logger.info(f"🎛️  LoRA rank (r): {args.lora_r}, alpha: {args.lora_alpha}")
    logger.info(f"🔀 Shuffle buffer: {args.shuffle_buffer_size:,} samples")
    logger.info(f"🧠 Trainable parameters: {trainable_params:,} ({trainable_params/total_params*100:.2f}% of {total_params:,})")
    logger.info("💡 Using FIXED QLORA with proper gradient computation")
    logger.info("=" * 50)

def main():
    args = parse_args()
    set_seed(args.seed)

    # Initialize resource monitor and interrupt handler
    resource_monitor = ResourceMonitor()
    interrupt_handler = GracefulInterruptHandler()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    logger.info("🚀 === Starting Fixed QLoRA Fine-tuning with GRADIENT COMPUTATION ===")
    logger.info(f"📁 Output directory: {args.output_dir}")
    logger.info(f"🤖 Local model path: {args.local_model_path}")

    # Save configuration
    config_dict = vars(args)
    config_dict['start_time'] = datetime.now().isoformat()
    config_dict['cuda_available'] = torch.cuda.is_available()

    with open(f"{args.output_dir}/training_config.json", 'w', encoding='utf-8') as f:
        json.dump(config_dict, f, indent=2, ensure_ascii=False)

    model = None
    tokenizer = None
    trainer = None

    try:
        # === Load and Prepare Model ===
        with resource_aware_execution(resource_monitor, "model_loading"):
            bnb_config = setup_quantization()
            model, tokenizer = load_and_prepare_model(args.local_model_path, bnb_config)

        # === Apply LoRA ===
        with resource_aware_execution(resource_monitor, "LoRA_setup"):
            logger.info("🎛️ Applying LoRA configuration...")
            peft_config = setup_lora_config(args)
            model = get_peft_model(model, peft_config)

            # Print detailed parameter information
            model.print_trainable_parameters()

            # Double-check trainable parameters
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            logger.info(f"✅ Final trainable parameters: {trainable_params:,}")

        # === Create Streaming Dataset ===
        train_dataset = create_streaming_dataset(args, tokenizer, resource_monitor)

        # Get approximate dataset length
        dataset_length = len(train_dataset)
        print_training_summary(args, dataset_length, model)

        # === Setup Training ===
        with resource_aware_execution(resource_monitor, "training_setup"):
            training_args = setup_training_args(args, dataset_length)

            data_collator = CustomDataCollatorForLanguageModeling(
                tokenizer=tokenizer,
                mlm=False,
                pad_to_multiple_of=8
            )

            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                data_collator=data_collator,
                tokenizer=tokenizer,
            )

            interrupt_handler.set_trainer(trainer)

        # === Check for resume ===
        if args.resume_from_checkpoint:
            if os.path.exists(args.resume_from_checkpoint):
                logger.info(f"🔄 Resuming from checkpoint: {args.resume_from_checkpoint}")
            else:
                logger.warning(f"Checkpoint not found: {args.resume_from_checkpoint}")

        # === Start Training ===
        with resource_aware_execution(resource_monitor, "training"):
            logger.info("🎬 Starting training with proper gradient computation...")
            start_time = datetime.now()
            resource_monitor.log_resources("Pre-training")

            # Train with proper error handling
            train_result = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

            training_duration = datetime.now() - start_time
            logger.info(f"✅ Training completed in {training_duration}")

            # Log training metrics
            logger.info(f"📊 Final training loss: {train_result.metrics.get('train_loss', 'unknown')}")

        # === Save Final Model ===
        with resource_aware_execution(resource_monitor, "model_saving"):
            logger.info("💾 Saving final model...")

            trainer.save_model()
            tokenizer.save_pretrained(args.output_dir)

            training_metadata = {
                "training_duration": str(training_duration),
                "completed_at": datetime.now().isoformat(),
                "total_steps": trainer.state.global_step,
                "final_loss": train_result.metrics.get('train_loss', 'unknown'),
                "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad)
            }

            with open(f"{args.output_dir}/training_metadata.json", 'w', encoding='utf-8') as f:
                json.dump(training_metadata, f, indent=2, ensure_ascii=False)

            logger.info("🎉 === Training Complete ===")
            logger.info(f"💾 Model saved to: {args.output_dir}")
            logger.info(f"⏱️ Training duration: {training_duration}")

    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")

        logger.info("Attempting emergency save...")
        emergency_save(args.output_dir, model, tokenizer)
        raise

    finally:
        # Cleanup
        logger.info("🧹 Cleaning up resources...")
        if model is not None:
            del model
        if trainer is not None:
            del trainer

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        resource_monitor.log_resources("After cleanup")

if __name__ == "__main__":
    main()