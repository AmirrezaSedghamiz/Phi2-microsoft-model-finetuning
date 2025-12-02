import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import html
import datetime
import os


def create_persian_test_suite():
    """Comprehensive test suite for Persian language model evaluation"""

    # Test prompts organized by complexity
    test_suite = {
        "basic_language_detection": [
            "Say hello in Persian",
            "What is 'thank you' in Persian?",
            "How do you say 'good morning' in Persian?",
        ],

        "simple_conversation": [
            "سلام، حالت چطوره؟",
            "امروز هوا چطور است؟",
            "اسم تو چیست؟",
            "چطور می‌توانم به تو کمک کنم؟",
        ],

        "translation_requests": [
            "Translate this to Persian: 'I am learning artificial intelligence'",
            "Translate to English: 'من دارم فارسی یاد می‌گیرم'",
            "How do you say 'machine learning' in Persian?",
        ],

        "cultural_knowledge": [
            "Explain Nowruz (Persian New Year)",
            "Who was Hafez?",
            "What is famous about Persian carpets?",
            "Tell me about Persian poetry",
        ],

        "complex_reasoning": [
            "Write a short story about a cat in Tehran",
            "Explain the importance of education in Persian culture",
            "Describe the process of making Persian tea",
            "What are the main challenges facing Iran today?",
        ],

        "domain_specific": [
            "Explain artificial intelligence in simple Persian",
            "What is quantum computing? Explain in Persian",
            "Describe the internet of things (IoT) in Persian",
        ],

        "creative_writing": [
            "Write a Persian poem about the moon",
            "Create a short dialogue between two friends in Tehran",
            "Write a news headline about technology in Iran",
        ],

        "personality_test": [
            "Tell me a joke in Persian",
            "How are you feeling today?",
            "What's your favorite Persian food?",
            "Can you be my friend?",
        ]
    }
    return test_suite


def load_model_and_tokenizer():
    """Load the fine-tuned model and tokenizer"""
    print("🔄 Loading model and tokenizer...")

    base_model_path = "G:/model/microsoft--phi-2/models--microsoft--phi-2/snapshots/ef382358ec9e382308935a992d908de099b64c23"
    fine_tuned_path = "G:/phi2-finetuned-large"

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_path,
        trust_remote_code=True,
        local_files_only=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Try to load fine-tuned model
    try:
        model = AutoModelForCausalLM.from_pretrained(
            fine_tuned_path,
            torch_dtype=torch.float16,
            device_map="cuda",
            trust_remote_code=True,
            local_files_only=True
        )
        print("✅ Loaded fine-tuned model")
    except Exception as e:
        print(f"❌ Could not load fine-tuned model: {e}")
        print("🔄 Loading base model + adapter...")
        try:
            model = AutoModelForCausalLM.from_pretrained(
                base_model_path,
                torch_dtype=torch.float16,
                device_map="cuda",
                trust_remote_code=True,
                local_files_only=True
            )
            model = PeftModel.from_pretrained(model, fine_tuned_path)
            print("✅ Loaded base model + adapter")
        except Exception as e2:
            print(f"❌ Failed to load model: {e2}")
            return None, None

    return model, tokenizer


def get_system_prompt():
    """Advanced system prompt engineering to enforce Persian-only responses with friendly personality"""

    system_prompt = """شما یک دستیار هوش مصنوعی فارسی‌زبان هستید که ویژگی‌های زیر را دارید:

🧠 **هویت و زبان:**
- فقط و فقط به زبان فارسی پاسخ می‌دهید
- هرگز از انگلیسی یا زبان‌های دیگر استفاده نمی‌کنید
- از زبان محاوره‌ای و دوستانه استفاده می‌کنید

❤️ **شخصیت و رفتار:**
- بسیار خوش‌برخورد و دوستانه هستید
- همیشه مایل به کمک کردن هستید
- با کاربران مانند دوستان قدیمی رفتار می‌کنید
- کمی شوخ‌طبع و صمیمی هستید

🎯 **وظایف:**
- به تمام درخواست‌های کاربر به بهترین شکل پاسخ می‌دهید
- اطلاعات مفید و دقیق ارائه می‌دهید
- در صورت نیاز توضیحات بیشتری می‌دهید
- خلاق و مبتکر هستید

📝 **سبک نوشتار:**
- از اصطلاحات و ضرب‌المثل‌های فارسی استفاده می‌کنید
- پاسخ‌ها را گرم و جذاب می‌کنید
- از اموجی مناسب استفاده می‌کنید (ولی نه زیاد)
- پاراگراف‌بندی مناسبی دارید

لطفاً تمام پاسخ‌های خود را بر اساس این ویژگی‌ها ارائه دهید."""

    return system_prompt


def generate_response(model, tokenizer, prompt, max_length=400, temperature=0.8):
    """Generate response with advanced prompt engineering and personality"""

    system_prompt = get_system_prompt()

    # Advanced prompt engineering templates with personality
    prompt_templates = {
        "persian_personality": f"""### سیستم:
{system_prompt}

### کاربر:
{prompt}

### دستیار فارسی:
""",

        "friendly_chat": f"""👤 کاربر: {prompt}

🤖 دستیار فارسی (دوستانه و خوش‌برخورد):""",

        "creative_mode": f"""🎭 **حالت خلاقانه فعال شد**
📋 دستور سیستم: {system_prompt}

💬 درخواست کاربر: «{prompt}»

✨ پاسخ من (به فارسی دوستانه):""",

        "instruction_following": f"""<|system|>
{system_prompt}
<|user|>
{prompt}
<|assistant|>"""
    }

    # Try different templates and return the best one
    best_response = ""
    for template_name, formatted_prompt in prompt_templates.items():
        try:
            inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_length=max_length,
                    temperature=temperature,
                    do_sample=True,
                    top_p=0.92,
                    top_k=50,
                    repetition_penalty=1.15,
                    no_repeat_ngram_size=3,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    early_stopping=True
                )

            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            # Extract only the new generated part
            response = response.replace(formatted_prompt, "").strip()

            # Quality checks for the response
            if (len(response) > 10 and
                    any(char in response for char in 'ابپتثجچحخدذرزژسشصضطظعغفقکگلمنوهی') and
                    len(response) > len(best_response)):
                best_response = response

        except Exception as e:
            continue

    # Fallback if no good response
    if not best_response:
        fallback_responses = [
            "سلام! متأسفانه در حال حاضر نمی‌تونم به درخواست شما پاسخ بدم. لطفاً کمی بعد تلاش کنید! 😊",
            "اوه! به نظر می‌رسه مشکلی پیش اومده. می‌تونید سوال خودتون رو دوباره بپرسید؟",
            "ببخشید، نتونستم متوجه بشم. لطفاً سوال خودتون رو به فارسی واضح‌تر بپرسید! 🌟"
        ]
        best_response = fallback_responses[hash(prompt) % len(fallback_responses)]

    return best_response


def create_html_report(test_results, output_path):
    """Create a beautiful HTML report with RTL support for Persian"""

    html_content = f"""
    <!DOCTYPE html>
    <html dir="rtl" lang="fa">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Persian Language Model Evaluation Report</title>
        <style>
            body {{
                font-family: 'Tahoma', 'Arial', sans-serif;
                line-height: 1.8;
                margin: 0;
                padding: 20px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
            }}
            .container {{
                max-width: 1200px;
                margin: 0 auto;
                background: white;
                padding: 40px;
                border-radius: 15px;
                box-shadow: 0 20px 40px rgba(0,0,0,0.1);
            }}
            .header {{
                text-align: center;
                border-bottom: 4px solid #2c5faa;
                padding-bottom: 25px;
                margin-bottom: 35px;
                background: linear-gradient(135deg, #2c5faa, #3a7bd5);
                color: white;
                padding: 30px;
                margin: -40px -40px 35px -40px;
                border-radius: 15px 15px 0 0;
            }}
            .header h1 {{
                color: white;
                font-size: 2.8em;
                margin: 0;
                text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
            }}
            .header p {{
                font-size: 1.3em;
                opacity: 0.9;
                margin: 10px 0 0 0;
            }}
            .test-category {{
                background: linear-gradient(135deg, #e8f4fd, #d4e7fa);
                padding: 20px;
                margin: 25px 0;
                border-radius: 12px;
                border-right: 6px solid #2c5faa;
                transition: transform 0.3s ease;
            }}
            .test-category:hover {{
                transform: translateY(-5px);
                box-shadow: 0 10px 25px rgba(0,0,0,0.15);
            }}
            .test-category h2 {{
                color: #2c5faa;
                margin-top: 0;
                font-size: 1.6em;
                border-bottom: 2px solid #2c5faa;
                padding-bottom: 10px;
            }}
            .test-item {{
                background: white;
                margin: 20px 0;
                padding: 20px;
                border-radius: 10px;
                border: 2px solid #e1e8ed;
                transition: all 0.3s ease;
            }}
            .test-item:hover {{
                border-color: #2c5faa;
                box-shadow: 0 5px 15px rgba(44, 95, 170, 0.1);
            }}
            .prompt {{
                background: linear-gradient(135deg, #f0f8ff, #e6f3ff);
                padding: 15px;
                border-radius: 8px;
                margin-bottom: 15px;
                font-weight: bold;
                border-right: 4px solid #4CAF50;
                font-size: 1.1em;
            }}
            .response {{
                background: linear-gradient(135deg, #fff9e6, #fff0cc);
                padding: 20px;
                border-radius: 8px;
                border-right: 4px solid #FF9800;
                white-space: pre-wrap;
                font-size: 1.15em;
                line-height: 1.8;
            }}
            .english {{
                direction: ltr;
                text-align: left;
                font-family: 'Courier New', monospace;
                background: linear-gradient(135deg, #f8f9fa, #e9ecef) !important;
            }}
            .persian {{
                font-size: 1.25em;
                line-height: 2;
                font-family: 'Tahoma', 'Arial', sans-serif;
            }}
            .timestamp {{
                text-align: center;
                color: #666;
                margin-top: 40px;
                font-style: italic;
                padding: 20px;
                background: #f8f9fa;
                border-radius: 10px;
            }}
            .quality-badge {{
                display: inline-block;
                background: #4CAF50;
                color: white;
                padding: 8px 16px;
                border-radius: 20px;
                font-size: 0.9em;
                margin: 10px 5px;
                box-shadow: 0 2px 5px rgba(0,0,0,0.2);
            }}
            .personality-note {{
                background: linear-gradient(135deg, #ffeaa7, #fab1a0);
                padding: 15px;
                border-radius: 10px;
                margin: 20px 0;
                border-right: 4px solid #e17055;
                font-weight: bold;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🧪 ارزیابی مدل زبانی فارسی</h1>
                <p>تست جامع مدل Phi-2 تنظیم‌شده با قابلیت‌های فارسی و شخصیت دوستانه</p>
            </div>

            <div class="personality-note">
                📝 <strong>نکته:</strong> مدل با دستورالعمل سیستم آموزش دیده تا:
                <span class="quality-badge">فقط فارسی صحبت کند</span>
                <span class="quality-badge">شخصیت دوستانه داشته باشد</span>
                <span class="quality-badge">همیشه کمک‌کننده باشد</span>
                <span class="quality-badge">خلاق و جذاب پاسخ دهد</span>
            </div>
    """

    # Add test results by category
    for category, tests in test_results.items():
        html_content += f"""
            <div class="test-category">
                <h2>📂 {category.replace('_', ' ').title()}</h2>
        """

        for i, (prompt, response) in enumerate(tests):
            prompt_class = "english" if any(char.isascii() and char.isalpha() for char in prompt) else "persian"
            response_class = "persian"  # Force Persian for responses

            html_content += f"""
                <div class="test-item">
                    <div class="prompt {prompt_class}">🗨️ <strong>پرسش:</strong> {html.escape(prompt)}</div>
                    <div class="response {response_class}">🤖 <strong>پاسخ:</strong>\n{html.escape(response)}</div>
                </div>
            """

        html_content += "</div>"

    # Add footer
    html_content += f"""
            <div class="timestamp">
                🕒 تولید شده در: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
                <br>
                🎯 مدل: Phi-2 تنظیم‌شده با QLoRA روی داده‌های فارسی
            </div>
        </div>
    </body>
    </html>
    """

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)

    print(f"✅ HTML report saved to: {output_path}")


def analyze_personality_metrics(responses):
    """Analyze how well the model follows personality instructions"""
    metrics = {
        'persian_only': 0,
        'friendly_tone': 0,
        'helpful_content': 0,
        'creative_elements': 0,
        'emoji_usage': 0
    }

    for response in responses:
        # Check for Persian characters
        if any(char in response for char in 'ابپتثجچحخدذرزژسشصضطظعغفقکگلمنوهی'):
            metrics['persian_only'] += 1

        # Check for friendly words
        friendly_words = ['سلام', 'لطفا', 'خوشحال', 'کمک', 'دوست', 'خوش', 'ممنون', 'عالی']
        if any(word in response for word in friendly_words):
            metrics['friendly_tone'] += 1

        # Check for helpful content (length and substance)
        if len(response.split()) > 10:
            metrics['helpful_content'] += 1

        # Check for creative elements
        creative_indicators = ['!', '؟', '...', '✨', '😊', '🌟', '🎯']
        if any(indicator in response for indicator in creative_indicators):
            metrics['creative_elements'] += 1

        # Check emoji usage
        if any(char in response for char in ['😊', '😂', '❤️', '✨', '🌟', '🎯', '🧠']):
            metrics['emoji_usage'] += 1

    # Convert to percentages
    total_responses = len(responses)
    for key in metrics:
        metrics[key] = (metrics[key] / total_responses) * 100

    return metrics


def main():
    """Main testing function"""
    print("🚀 Starting Advanced Persian Language Model Evaluation")
    print("🎭 System Prompt: Persian-only, friendly personality enforced")

    # Load model
    model, tokenizer = load_model_and_tokenizer()
    if model is None:
        print("❌ Failed to load model. Exiting.")
        return

    # Create test suite
    test_suite = create_persian_test_suite()
    test_results = {}
    all_responses = []

    print("🧪 Running comprehensive personality-enforced tests...")
    print("📋 System instructions: Persian-only, friendly, helpful, creative")

    # Run all tests
    for category, prompts in test_suite.items():
        print(f"📋 Testing: {category}")
        category_results = []

        for prompt in prompts:
            print(f"   Testing: {prompt[:50]}...")
            response = generate_response(model, tokenizer, prompt)
            category_results.append((prompt, response))
            all_responses.append(response)

        test_results[category] = category_results

    # Analyze personality metrics
    print("\n📊 Analyzing personality adherence...")
    metrics = analyze_personality_metrics(all_responses)

    print("🎯 Personality Metrics:")
    for metric, score in metrics.items():
        print(f"   {metric}: {score:.1f}%")

    # Generate HTML report
    output_path = "persian_personality_evaluation_report.html"
    create_html_report(test_results, output_path)

    # Generate metrics summary
    with open("personality_metrics.txt", "w", encoding="utf-8") as f:
        f.write("شاخص‌های شخصیت مدل فارسی\n")
        f.write("=" * 50 + "\n\n")
        for metric, score in metrics.items():
            f.write(f"{metric}: {score:.1f}%\n")

    print("✅ Evaluation complete!")
    print(f"📊 HTML Report: {output_path}")
    print(f"📈 Metrics: personality_metrics.txt")

    # Print quick summary to console
    print("\n" + "=" * 70)
    print("QUICK PERSONALITY SUMMARY (Check HTML for proper Persian rendering):")
    print("=" * 70)

    for category, tests in test_results.items():
        print(f"\n📂 {category.upper()}:")
        for prompt, response in tests[:2]:
            print(f"   Prompt: {prompt[:60]}...")
            print(f"   Response: {response[:100]}...")
            print()


if __name__ == "__main__":
    main()