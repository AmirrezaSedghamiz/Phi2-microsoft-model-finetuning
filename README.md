🧠 QLoRA Fine-tuning of Phi-2 for Persian Language Understanding
https://img.shields.io/badge/Python-3.9+-blue.svg
https://img.shields.io/badge/PyTorch-2.0+-red.svg
https://img.shields.io/badge/HuggingFace-Transformers-yellow.svg
https://img.shields.io/badge/Training-QLoRA-green.svg
https://img.shields.io/badge/License-MIT-green.svg
https://img.shields.io/badge/Dataset-7M%252B_Persian-orange.svg

Advanced QLoRA fine-tuning of Microsoft's Phi-2 model on large Persian datasets with resource-efficient training, comprehensive evaluation, and personality-enforced Persian-only responses.

📋 Project Overview
This project implements efficient fine-tuning of Phi-2 (2.7B parameters) using QLoRA (Quantized Low-Rank Adaptation) on 7M+ Persian samples. The model is trained to respond exclusively in Persian with a friendly, helpful personality while maintaining resource efficiency (50% GPU/CPU limits).

🎯 Key Features
✅ Resource-Efficient Training: Dynamic resource monitoring with enforced limits (50% GPU, 50% CPU, 33% RAM)

✅ Streaming Data Pipeline: Handles 7M+ samples without memory overload

✅ Graceful Recovery: Automatic checkpointing and resume functionality

✅ Personality Engineering: Enforces Persian-only responses with friendly AI persona

✅ Comprehensive Evaluation: 8 test categories with HTML reporting

✅ Production Ready: Robust error handling and emergency saves

🏗️ Project Structure
text
QLoRa-FineTuning/
├── 📁 Data/                          # Dataset directory (not in Git - large files)
│   ├── General Data/                # Persian news, blogs, Q&A datasets
│   └── Advising Data/               # Specialized Persian datasets
│
├── 📁 src/                          # Source code
│   ├── main.py                     # 🚀 Main training script (optimized)
│   ├── main_optimized.py           # Original optimized version
│   ├── test_model.py               # 🧪 Interactive model testing
│   ├── testing.py                  # 📊 Comprehensive evaluation suite
│   └── exam.py                     # 🔍 Graph algorithm implementation
│
├── 📁 outputs/                      # Training outputs (not in Git)
│   ├── checkpoints/                # Model checkpoints
│   ├── logs/                       # Training logs
│   └── evaluation_results/         # Evaluation outputs
│
├── 📄 requirements.txt             # Python dependencies
├── 📄 training.log                 # Training progress log
├── 📄 README.md                    # This file
└── 📄 .gitignore                   # Git ignore rules
🚀 Quick Start
1. Installation
bash
# Clone repository
git clone https://github.com/yourusername/QLoRa-FineTuning.git
cd QLoRa-FineTuning

# Install dependencies
pip install -r requirements.txt

# Additional RTL support for Persian display
pip install arabic-reshaper python-bidi
2. Download Base Model
bash
# Download Phi-2 from HuggingFace
python -c "from transformers import AutoModel; AutoModel.from_pretrained('microsoft/phi-2', cache_dir='G:/model')"
3. Run Training (7M samples)
bash
python src/main.py \
  --dataset "G:/persian_news_processed.jsonl" \
  --local_model_path "G:/model/microsoft--phi-2/models--microsoft--phi-2/snapshots/ef382358ec9e382308935a992d908de099b64c23" \
  --output_dir "./phi2-finetuned" \
  --max_seq_length 512 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 16 \
  --num_train_epochs 1 \
  --learning_rate 2e-4 \
  --lora_r 16 \
  --lora_alpha 32 \
  --max_samples 7000000 \
  --save_steps 500 \
  --logging_steps 50
4. Test the Model
bash
# Interactive testing
python src/test_model.py --model_path "./phi2-finetuned"

# Comprehensive evaluation with HTML report
python src/testing.py
🔧 Technical Implementation
🏗️ Architecture
python
Base Model: Microsoft Phi-2 (2.7B parameters)
Fine-tuning: QLoRA (4-bit quantization + Low-Rank Adaptation)
Adapter Rank: r=16, alpha=32
Training Precision: bfloat16/fp16
Sequence Length: 512 tokens
📊 Resource Management
GPU Utilization: Max 50% with dynamic throttling

CPU Utilization: Max 50% with task scheduling

RAM Management: Max 33% with garbage collection

Streaming Pipeline: Processes 7M+ samples without loading all into memory

Checkpointing: Automatic saves every 500 steps

🎭 Personality Engineering
The model is trained with strict system prompts enforcing:

Persian-only responses - No English code-switching

Friendly persona - Warm, helpful AI assistant

Cultural awareness - Understanding of Persian culture

Creative responses - Engaging and thoughtful answers

📈 Evaluation Results
🧪 Test Categories
Category	Samples	Purpose
Basic Language	3	Persian language detection
Conversation	4	Natural Persian dialogue
Translation	3	Bi-directional translation
Cultural Knowledge	4	Persian culture understanding
Complex Reasoning	4	Advanced Persian reasoning
Domain Specific	3	Technical Persian explanations
Creative Writing	3	Persian creative content
Personality	4	Friendly AI behavior
📊 Performance Metrics
yaml
Persian Language Adherence: 100%
Helpful Content Generation: 100%
Friendly Tone: 25%
Creative Elements: 28.6%
Emoji Usage (Appropriate): 7.1%
💾 Dataset Information
📚 Composition
Dataset	Samples	Size	Description
Persian News	~5M	~30GB	Modern Persian news articles
Persian Blogs	~1M	~8GB	Informal Persian blog posts
Persian Q&A	~1M	~6GB	Question-answer pairs
Total	~7M	~44GB	Comprehensive Persian corpus
🔄 Data Processing Pipeline
python
1. Streaming JSONL Loading → 2. Persian Text Extraction → 
3. Quality Filtering → 4. Tokenization → 
5. Sequence Padding → 6. Batch Generation
🛠️ Advanced Usage
Resume Training from Checkpoint
bash
python src/main.py \
  --resume_from_checkpoint "./phi2-finetuned/checkpoint-5000" \
  # ... other parameters
Custom Evaluation
python
from testing import create_persian_test_suite, generate_response

# Create custom test suite
tests = {
    "my_category": ["سوال فارسی ۱", "سوال فارسی ۲"]
}

# Run evaluation
results = {}
for category, prompts in tests.items():
    responses = [generate_response(model, tokenizer, p) for p in prompts]
    results[category] = list(zip(prompts, responses))
Export for Production
python
# Convert to single model
model.save_pretrained("./production_model", safe_serialization=True)
tokenizer.save_pretrained("./production_model")
📝 Key Files Explained
File	Purpose	Key Features
main.py	🚀 Main training	Streaming data, QLoRA, resource limits
test_model.py	🧪 Interactive testing	RTL Persian display, streaming responses
testing.py	📊 Evaluation suite	8 test categories, HTML reports, personality metrics
exam.py	🔍 Algorithm	Graph cycle detection, component analysis
🎯 Technical Challenges Solved
Memory Management: Streaming pipeline for 7M+ samples

Resource Efficiency: Dynamic GPU/CPU/RAM limiting

Persian RTL Support: Proper Arabic script rendering

Personality Enforcement: System prompt engineering for Persian-only responses

Training Stability: Gradient checkpointing, mixed precision, LoRA stabilization

📊 Model Specifications
Parameter	Value	Description
Base Model	Phi-2 (2.7B)	Microsoft's compact LM
Fine-tuning	QLoRA	4-bit quantized LoRA
Trainable Params	~4.1M	0.15% of total parameters
Sequence Length	512	Optimized for Persian
Batch Size	1 × 16	Effective batch size 16
Learning Rate	2e-4	Cosine scheduler
Training Time	~72 hours	On single RTX 3090
🔍 Evaluation Examples
✅ Successful Responses
persian
🎯 **پرسش**: سلام، حالت چطوره؟
🤖 **پاسخ**: سلام عزیزم! 😊 خدا رو شکر خیلی خوبم، امیدوارم تو هم حالت عالی باشه. چه خبرا؟ چطور می‌تونم کمکت کنم امروز؟

🎯 **پرسش**: Explain artificial intelligence in simple Persian
🤖 **پاسخ**: هوش مصنوعی مثل یک بچه باهوشه که می‌تونه چیزای جدید یاد بگیره! 🧠 این فناوری به کامپیوترها کمک می‌کنه تا فکر کنند، یاد بگیرند و تصمیم بگیرند...
📈 Personality Metrics
100% Persian adherence - No English leakage

Natural conversation flow - Contextually appropriate

Cultural relevance - Persian idioms and references

Friendly tone - Warm and approachable

🚨 Troubleshooting
Common Issues
bash
# Out of memory
Solution: Reduce batch size, enable gradient checkpointing

# Slow training
Solution: Increase gradient accumulation steps, optimize data loading

# Poor Persian quality
Solution: Adjust temperature (0.7-0.9), review training data quality

# Checkpoint loading failed
Solution: Ensure same model architecture, check file permissions
Performance Tips
yaml
For faster training:
  - Use flash attention if available
  - Increase gradient accumulation steps
  - Enable mixed precision (bfloat16)
  - Use smaller LoRA rank (r=8)

For better quality:
  - Increase training epochs (2-3)
  - Use larger LoRA rank (r=32)
  - Adjust learning rate (1e-4 to 3e-4)
  - Add more diverse Persian data
📚 References & Citations
Academic Papers
QLoRA: Efficient Fine-tuning of Quantized LLMs

Phi-2: The Surprising Reasoning Power of Small LMs

Low-Rank Adaptation (LoRA) for LLMs

Libraries Used
HuggingFace Transformers

PEFT (Parameter-Efficient Fine-Tuning)

BitsAndBytes

Persian NLP Resources
Persian NLP Datasets

📄 License
This project is licensed under the MIT License - see the LICENSE file for details.

🙏 Acknowledgments
Microsoft Research for the Phi-2 base model

HuggingFace for Transformers and PEFT libraries

Persian NLP Community for datasets and resources

QLoRA authors for efficient fine-tuning method

📧 Contact
Your Name - GitHub Profile

Project Link: https://github.com/yourusername/QLoRa-FineTuning

⭐ Show Your Support
If you find this project useful, please give it a star! ⭐

🎯 Skills Demonstrated
This project showcases expertise in:

Large Language Model fine-tuning with QLoRA

Resource-efficient training on limited hardware

Persian NLP and RTL text processing

Comprehensive evaluation with personality metrics

Production-ready ML pipelines with error handling

Streaming data processing for large datasets

Built with ❤️ for the Persian AI community
HuggingFace Persian Models

