import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel, PeftConfig
import argparse
import sys
import time
from arabic_reshaper import reshape
from bidi.algorithm import get_display
import threading
from queue import Queue, Empty


def is_persian(text):
    """Check if text contains Persian/Arabic characters"""
    persian_chars = set('ابپتثجچحخدذرزژسشصضطظعغفقکگلمنوهی')
    return any(char in persian_chars for char in text)


def format_rtl(text):
    """Format text for RTL display"""
    try:
        # Reshape Arabic/Persian characters for proper display
        reshaped_text = reshape(text)
        # Apply bidirectional algorithm for RTL
        return get_display(reshaped_text)
    except:
        # Fallback if arabic_reshaper or bidi not available
        return text


def print_rtl(text, is_response=False):
    """Print text with RTL formatting if it's Persian"""
    if is_persian(text):
        formatted = format_rtl(text)
        if is_response:
            print(f"📝 {formatted}")
        else:
            print(f"💬 {formatted}")
    else:
        if is_response:
            print(f"📝 {text}")
        else:
            print(f"💬 {text}")


def stream_output(generator, tokenizer, prompt):
    """Stream the output token by token"""
    print("🔄 Generating... ", end="", flush=True)

    generated_text = ""
    for new_token in generator:
        token_text = tokenizer.decode(new_token, skip_special_tokens=True)

        # Remove the prompt from the beginning if it's there
        if not generated_text and token_text.startswith(prompt):
            token_text = token_text[len(prompt):]

        generated_text += token_text

        # Print each token as it comes
        if token_text.strip():
            if is_persian(token_text):
                print(format_rtl(token_text), end="", flush=True)
            else:
                print(token_text, end="", flush=True)

    print()  # New line after streaming
    return generated_text


def test_model(model_path):
    # The correct base model path
    base_model_path = "G:/model/microsoft--phi-2/models--microsoft--phi-2/snapshots/ef382358ec9e382308935a992d908de099b64c23"

    print(f"🔧 Loading base model from: {base_model_path}")
    print(f"🔧 Loading adapter from: {model_path}")

    # Load with 4-bit quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )

    try:
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.float16,
            local_files_only=True
        )
        print("✅ Base model loaded successfully")
    except Exception as e:
        print(f"❌ ERROR loading base model: {e}")
        return

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            base_model_path,
            trust_remote_code=True,
            local_files_only=True
        )
        print("✅ Tokenizer loaded successfully")
    except Exception as e:
        print(f"❌ ERROR loading tokenizer: {e}")
        return

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    try:
        model = PeftModel.from_pretrained(base_model, model_path)
        print("✅ Fine-tuned adapter loaded successfully")
    except Exception as e:
        print(f"❌ ERROR loading adapter: {e}")
        return

    # Enhanced test prompts with prompt engineering
    test_prompts = [
        # Persian instructional prompts
        "لطفاً در مورد هوش مصنوعی توضیح دهید:",
        "متن زیر را به زبان ساده توضیح ده: ماشین لرنینگ چیست؟",
        "به عنوان یک معلم آنلاین، برنامه نویسی پایتون را آموزش بده:",
        "سوال: آب و هوای تهران چگونه است؟ پاسخ:",
        "در مورد تاریخ ایران بنویس:",

        # English instructional prompts
        "Please explain artificial intelligence in simple terms:",
        "As an online instructor, teach me Python programming:",
        "Question: What is machine learning? Answer:",
        "Explain quantum computing like I'm 10 years old:",

        # Mixed language prompts
        "Translate to English: 'هواشناسی امروز چگونه است؟'",
        "Translate to Persian: 'The weather is nice today'",
    ]

    print("\n" + "=" * 70)
    print("🧪 TESTING FINE-TUNED MODEL WITH PROMPT ENGINEERING")
    print("=" * 70)

    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n{'=' * 50}")
        print(f"🎯 TEST {i}/{len(test_prompts)}")
        print(f"{'=' * 50}")

        print_rtl(f"Prompt: {prompt}")
        print("-" * 50)

        try:
            # Tokenize input
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

            # Generate with streaming
            start_time = time.time()

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=150,
                    temperature=0.8,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id,
                    repetition_penalty=1.2,
                    top_p=0.9,
                    early_stopping=True,
                    num_return_sequences=1
                )

            generation_time = time.time() - start_time

            # Decode response
            full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Extract only the generated part
            if full_response.startswith(prompt):
                response = full_response[len(prompt):].strip()
            else:
                response = full_response

            print("✅ Response:")
            print_rtl(response, is_response=True)

            print(f"\n⏱️  Generated in {generation_time:.2f}s")
            print(f"📊 Response length: {len(response)} characters")

        except Exception as e:
            print(f"❌ Generation error: {e}")

        # Pause between tests
        if i < len(test_prompts):
            print("\n⏳ Continuing to next test in 3 seconds...")
            time.sleep(3)

    print("\n" + "=" * 70)
    print("✅ ALL TESTS COMPLETED")
    print("=" * 70)

    # Interactive mode
    print("\n🎮 INTERACTIVE MODE (type 'quit' to exit)")
    print("=" * 50)

    while True:
        try:
            user_input = input("\n💬 You: ").strip()
            if user_input.lower() in ['quit', 'exit', 'خروج']:
                break

            if not user_input:
                continue

            # Add instruction context for better responses
            if is_persian(user_input):
                enhanced_prompt = f"لطفاً به سوال زیر پاسخ دهید: {user_input}"
            else:
                enhanced_prompt = f"Please answer the following question: {user_input}"

            inputs = tokenizer(enhanced_prompt, return_tensors="pt").to(model.device)

            print("🤖 AI: ", end="", flush=True)

            start_time = time.time()
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=200,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id,
                    repetition_penalty=1.1
                )

            full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)

            if full_response.startswith(enhanced_prompt):
                response = full_response[len(enhanced_prompt):].strip()
            else:
                response = full_response

            print_rtl(response, is_response=True)
            print(f"⏱️  {time.time() - start_time:.2f}s")

        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")


if __name__ == "__main__":
    # Install required packages for RTL if not available
    try:
        import arabic_reshaper
        from bidi.algorithm import get_display
    except ImportError:
        print("📦 Installing required packages for RTL support...")
        print("Run: pip install arabic-reshaper python-bidi")
        print("Continuing without RTL formatting...")

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Path to fine-tuned adapter")

    args = parser.parse_args()
    test_model(args.model_path)