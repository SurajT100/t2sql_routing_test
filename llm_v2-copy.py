import requests
import anthropic
import os
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables from .env file
load_dotenv()


def call_llm(prompt: str, provider: str, prefill: str = None, stop_sequences: list = None):
    """
    Universal LLM caller - routes to appropriate provider
    Returns: (response_text, token_dict)
    
    Args:
        prompt: The prompt text
        provider: LLM provider code
        prefill: Optional prefilled assistant response start (Claude only)
        stop_sequences: Optional list of stop sequences
    
    Supported providers:
    - nvidia_qwen3: NVIDIA Qwen 3 Next 80B (with thinking)
    - vertex_qwen_thinking: Vertex AI Qwen3-Next-80B Thinking (reviewer)
    - o1_mini: OpenAI o1-mini (best reasoning/price)
    - o1: OpenAI o1 (best reasoning, expensive)
    - claude_sonnet: Claude Sonnet 4
    - claude_haiku: Claude Haiku 4.5  
    - groq: Groq Llama 3.3 70B
    - grok: xAI Grok Beta
    - vertex_qwen: Qwen 2.5 Coder 32B
    """
    if provider == "nvidia_qwen3":
        return call_nvidia_qwen3(prompt, stop_sequences)
    elif provider == "vertex_qwen_thinking":
        return call_vertex_qwen_thinking(prompt, stop_sequences)
    elif provider == "o1_mini":
        return call_o1_mini(prompt)
    elif provider == "o1":
        return call_o1(prompt)
    elif provider == "claude_sonnet":
        return call_claude_sonnet(prompt, prefill, stop_sequences)
    elif provider == "claude_opus":
        return call_claude_opus(prompt, prefill, stop_sequences)
    elif provider == "claude_haiku":
        return call_claude_haiku(prompt, prefill, stop_sequences)
    elif provider == "vertex_qwen":
        return call_qwen_vertex(prompt)
    elif provider == "groq":
        return call_groq(prompt, stop_sequences)
    elif provider == "grok":
        return call_grok(prompt, stop_sequences)
    else:
        raise ValueError(f"Unknown provider: {provider}")


def call_o1_mini(prompt: str):
    """
    Call OpenAI o1-mini for complex reasoning
    Best price/performance for reasoning tasks
    Returns: (response_text, token_dict)
    """
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    
    response = client.chat.completions.create(
        model="o1-mini",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )
    
    tokens = {
        "input": response.usage.prompt_tokens,
        "output": response.usage.completion_tokens
    }
    
    return response.choices[0].message.content, tokens


def call_o1(prompt: str):
    """
    Call OpenAI o1 for maximum reasoning capability
    Most expensive but most accurate
    Returns: (response_text, token_dict)
    """
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    
    response = client.chat.completions.create(
        model="o1",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )
    
    tokens = {
        "input": response.usage.prompt_tokens,
        "output": response.usage.completion_tokens
    }
    
    return response.choices[0].message.content, tokens


def call_nvidia_qwen3(prompt: str, stop_sequences: list = None):
    """
    Call NVIDIA Qwen 3 Next 80B with thinking capability
    Excellent reasoning model with extended thinking process
    
    Args:
        prompt: The prompt text
        stop_sequences: Optional list of strings to stop generation
    
    Returns: (response_text, token_dict)
    """
    try:
        client = OpenAI(
            base_url="https://integrate.api.nvidia.com/v1",
            api_key=os.environ.get("NVIDIA_API_KEY")
        )
        
        # Build request parameters
        params = {
            "model": "qwen/qwen3-next-80b-a3b-thinking",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.6,
            "top_p": 0.7,
            "max_tokens": 4096
        }
        
        # Add stop sequences if provided
        if stop_sequences:
            params["stop"] = stop_sequences
        
        response = client.chat.completions.create(**params)
        
        tokens = {
            "input": response.usage.prompt_tokens if response.usage else 0,
            "output": response.usage.completion_tokens if response.usage else 0
        }
        
        # Return the content (thinking is internal, final answer in content)
        return response.choices[0].message.content, tokens
        
    except Exception as e:
        # If NVIDIA fails, provide helpful error
        error_msg = f"NVIDIA API Error: {str(e)}\n\nPlease check:\n1. NVIDIA_API_KEY is set in .env\n2. API key is valid\n3. Model is accessible"
        return error_msg, {"input": 0, "output": 0}



def call_claude_sonnet(prompt: str, prefill: str = None, stop_sequences: list = None):
    """
    Call Claude Sonnet 4.5 for reasoning and analysis
    
    Args:
        prompt: The prompt text
        prefill: Optional prefilled assistant response (e.g., "{" for JSON)
        stop_sequences: Optional list of strings to stop generation
    
    Returns: (response_text, token_dict)
    """
    client = anthropic.Anthropic(
        api_key=os.environ.get("ANTHROPIC_API_KEY")
    )
    
    # Build messages array
    messages = [{"role": "user", "content": prompt}]
    
    # Add prefill if provided
    if prefill:
        messages.append({"role": "assistant", "content": prefill})
    
    # Build request parameters
    params = {
        "model": "claude-sonnet-4-20250514",
        "max_tokens": 4096,
        "temperature": 0.0,
        "messages": messages
    }
    
    # Add stop sequences if provided
    if stop_sequences:
        params["stop_sequences"] = stop_sequences
    
    message = client.messages.create(**params)
    
    tokens = {
        "input": message.usage.input_tokens,
        "output": message.usage.output_tokens
    }
    
    # If prefilled, prepend the prefill to the response
    response_text = message.content[0].text
    if prefill:
        response_text = prefill + response_text
    
    return response_text, tokens


def call_claude_opus(prompt: str, prefill: str = None, stop_sequences: list = None):
    """
    Call Claude Opus 4 for deep validation and review
    Most accurate model - use for critical review tasks
    
    Args:
        prompt: The prompt text
        prefill: Optional prefilled assistant response (e.g., "{" for JSON)
        stop_sequences: Optional list of strings to stop generation
    
    Returns: (response_text, token_dict)
    """
    client = anthropic.Anthropic(
        api_key=os.environ.get("ANTHROPIC_API_KEY")
    )
    
    # Build messages array
    messages = [{"role": "user", "content": prompt}]
    
    # Add prefill if provided
    if prefill:
        messages.append({"role": "assistant", "content": prefill})
    
    # Build request parameters
    params = {
        "model": "claude-opus-4-5-20251101",
        "max_tokens": 4096,
        "temperature": 0.0,
        "messages": messages
    }
    
    # Add stop sequences if provided
    if stop_sequences:
        params["stop_sequences"] = stop_sequences
    
    message = client.messages.create(**params)
    
    tokens = {
        "input": message.usage.input_tokens,
        "output": message.usage.output_tokens
    }
    
    # If prefilled, prepend the prefill to the response
    response_text = message.content[0].text
    if prefill:
        response_text = prefill + response_text
    
    return response_text, tokens


def call_claude_haiku(prompt: str, prefill: str = None, stop_sequences: list = None):
    """
    Call Claude Haiku 4.5 for fast reasoning
    
    Args:
        prompt: The prompt text
        prefill: Optional prefilled assistant response (e.g., "{" for JSON)
        stop_sequences: Optional list of strings to stop generation
    
    Returns: (response_text, token_dict)
    """
    client = anthropic.Anthropic(
        api_key=os.environ.get("ANTHROPIC_API_KEY")
    )
    
    # Build messages array
    messages = [{"role": "user", "content": prompt}]
    
    # Add prefill if provided
    if prefill:
        messages.append({"role": "assistant", "content": prefill})
    
    # Build request parameters
    params = {
        "model": "claude-haiku-4-5-20251001",
        "max_tokens": 4096,
        "temperature": 0.0,
        "messages": messages
    }
    
    # Add stop sequences if provided
    if stop_sequences:
        params["stop_sequences"] = stop_sequences
    
    message = client.messages.create(**params)
    
    tokens = {
        "input": message.usage.input_tokens,
        "output": message.usage.output_tokens
    }
    
    # If prefilled, prepend the prefill to the response
    response_text = message.content[0].text
    if prefill:
        response_text = prefill + response_text
    
    return response_text, tokens


def call_groq(prompt: str, stop_sequences: list = None):
    """
    Call Groq LLaMA 3.3 70B Versatile
    Fast inference with good reasoning
    
    Args:
        prompt: The prompt text
        stop_sequences: Optional list of strings to stop generation
    
    Returns: (response_text, token_dict)
    """
    api_key = os.environ.get("GROQ_API_KEY")
    
    if not api_key:
        raise ValueError("GROQ_API_KEY environment variable not set")
    
    url = "https://api.groq.com/openai/v1/chat/completions"
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": "llama-3.3-70b-versatile",
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "temperature": 0.0,
        "max_tokens": 4096
    }
    
    # Add stop sequences if provided
    if stop_sequences:
        payload["stop"] = stop_sequences
    
    response = requests.post(url, headers=headers, json=payload, timeout=60)
    response.raise_for_status()
    
    data = response.json()
    
    tokens = {
        "input": data["usage"]["prompt_tokens"],
        "output": data["usage"]["completion_tokens"]
    }
    
    return data["choices"][0]["message"]["content"], tokens


def call_grok(prompt: str, stop_sequences: list = None):
    """
    Call xAI Grok Beta via OpenAI-compatible API
    
    Grok is xAI's conversational AI with real-time knowledge
    Good for reasoning and understanding context
    
    Args:
        prompt: The prompt text
        stop_sequences: Optional list of strings to stop generation
    
    Returns: (response_text, token_dict)
    """
    api_key = os.environ.get("XAI_API_KEY")  # or GROK_API_KEY
    
    if not api_key:
        raise ValueError("XAI_API_KEY (or GROK_API_KEY) environment variable not set")
    
    url = "https://api.x.ai/v1/chat/completions"
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": "grok-beta",  # Latest Grok model
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "temperature": 0.0,
        "max_tokens": 4096
    }
    
    # Add stop sequences if provided
    if stop_sequences:
        payload["stop"] = stop_sequences
    
    response = requests.post(url, headers=headers, json=payload, timeout=60)
    response.raise_for_status()
    
    data = response.json()
    
    # Grok uses OpenAI-compatible response format
    tokens = {
        "input": data["usage"]["prompt_tokens"],
        "output": data["usage"]["completion_tokens"]
    }
    
    return data["choices"][0]["message"]["content"], tokens


def call_qwen_vertex(prompt: str):
    """
    Call Qwen via Vertex AI MaaS endpoint
    Returns: (response_text, token_dict)
    """
    import google.auth.transport.requests
    from google.oauth2 import service_account
    
    PROJECT_ID = "robust-carver-481011-c9"
    LOCATION = "us-south1"
    MODEL_NAME = "qwen3-coder-480b-a35b-instruct"
    SERVICE_ACCOUNT_JSON = (
        r"C:\Users\Dell\Desktop\Test\projectx\testing_app"
        r"\robust-carver-481011-c9-326237439fb7.json"
    )

    credentials = service_account.Credentials.from_service_account_file(
        SERVICE_ACCOUNT_JSON,
        scopes=["https://www.googleapis.com/auth/cloud-platform"]
    )

    auth_req = google.auth.transport.requests.Request()
    credentials.refresh(auth_req)

    url = (
        f"https://{LOCATION}-aiplatform.googleapis.com/v1/"
        f"projects/{PROJECT_ID}/locations/{LOCATION}/"
        f"publishers/qwen/models/{MODEL_NAME}-maas:generateContent"
    )

    headers = {
        "Authorization": f"Bearer {credentials.token}",
        "Content-Type": "application/json"
    }

    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [{"text": prompt}]
            }
        ],
        "generationConfig": {
            "temperature": 0.0,
            "maxOutputTokens": 4096
        }
    }

    response = requests.post(url, headers=headers, json=payload, timeout=60)
    response.raise_for_status()
    
    data = response.json()
    
    # Extract token usage from MaaS response
    usage_metadata = data.get("usageMetadata", {})
    tokens = {
        "input": usage_metadata.get("promptTokenCount", 0),
        "output": usage_metadata.get("candidatesTokenCount", 0)
    }
    
    response_text = data["candidates"][0]["content"]["parts"][0]["text"]
    
    return response_text, tokens


def call_vertex_qwen_thinking(prompt: str, stop_sequences: list = None):
    """
    Call Qwen3-Next-80B Thinking via Vertex AI MaaS endpoint
    Used as alternative reviewer to Opus
    
    Args:
        prompt: The prompt text
        stop_sequences: Optional list of stop sequences (not used in Vertex)
    
    Returns: (response_text, token_dict)
    """
    import google.auth.transport.requests
    from google.oauth2 import service_account
    
    PROJECT_ID = "robust-carver-481011-c9"
    LOCATION = "us-south1"
    MODEL_NAME = "qwen3-next-80b-a3b-thinking"  # Thinking model for review
    SERVICE_ACCOUNT_JSON = (
        r"C:\Users\Dell\Desktop\Test\projectx\testing_app"
        r"\robust-carver-481011-c9-326237439fb7.json"
    )

    try:
        credentials = service_account.Credentials.from_service_account_file(
            SERVICE_ACCOUNT_JSON,
            scopes=["https://www.googleapis.com/auth/cloud-platform"]
        )

        auth_req = google.auth.transport.requests.Request()
        credentials.refresh(auth_req)

        url = (
            f"https://{LOCATION}-aiplatform.googleapis.com/v1/"
            f"projects/{PROJECT_ID}/locations/{LOCATION}/"
            f"publishers/qwen/models/{MODEL_NAME}-maas:generateContent"
        )

        headers = {
            "Authorization": f"Bearer {credentials.token}",
            "Content-Type": "application/json"
        }

        payload = {
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": prompt}]
                }
            ],
            "generationConfig": {
                "temperature": 0.6,  # Allow some reasoning flexibility
                "topP": 0.7,
                "maxOutputTokens": 4096
            }
        }

        response = requests.post(url, headers=headers, json=payload, timeout=90)
        response.raise_for_status()
        
        data = response.json()
        
        # Extract token usage from MaaS response
        usage_metadata = data.get("usageMetadata", {})
        tokens = {
            "input": usage_metadata.get("promptTokenCount", 0),
            "output": usage_metadata.get("candidatesTokenCount", 0)
        }
        
        response_text = data["candidates"][0]["content"]["parts"][0]["text"]
        
        return response_text, tokens
        
    except Exception as e:
        error_msg = f"Vertex AI Qwen Thinking Error: {str(e)}"
        print(f"[ERROR] {error_msg}")
        return error_msg, {"input": 0, "output": 0}


# ============================================================================
# LLM COST CALCULATOR (Optional utility)
# ============================================================================

def calculate_cost(provider: str, input_tokens: int, output_tokens: int) -> float:
    """
    Calculate approximate cost for LLM call
    
    Pricing (as of Jan 2025, subject to change):
    - NVIDIA Qwen 3 Next 80B: $0.50/$1.50 per 1M tokens (estimated)
    - OpenAI o1: $15/$60 per 1M tokens
    - OpenAI o1-mini: $3/$12 per 1M tokens
    - Claude Sonnet 4: $3/$15 per 1M tokens
    - Claude Haiku 4.5: $1/$5 per 1M tokens
    - Groq Llama 3.3: Free tier / $0.59/$0.79 per 1M tokens
    - Grok Beta: Pricing TBD (currently in beta)
    - Qwen Vertex: Variable pricing
    """
    
    pricing = {
        "nvidia_qwen3": {"input": 0.50, "output": 1.50},  # per 1M tokens (estimated)
        "o1": {"input": 15.0, "output": 60.0},
        "o1_mini": {"input": 3.0, "output": 12.0},
        "claude_opus": {"input": 15.0, "output": 75.0},  # Most expensive, most accurate
        "claude_sonnet": {"input": 3.0, "output": 15.0},
        "claude_haiku": {"input": 1.0, "output": 5.0},
        "groq": {"input": 0.59, "output": 0.79},
        "grok": {"input": 0.0, "output": 0.0},  # Beta pricing TBD
        "vertex_qwen": {"input": 0.0, "output": 0.0}  # Variable
    }
    
    if provider not in pricing:
        return 0.0
    
    input_cost = (input_tokens / 1_000_000) * pricing[provider]["input"]
    output_cost = (output_tokens / 1_000_000) * pricing[provider]["output"]
    
    return input_cost + output_cost


if __name__ == "__main__":
    print("\n" + "="*70)
    print("LLM CALLER V2 - Ready!")
    print("="*70)
    print("\nSupported Providers:")
    print("  • claude_sonnet   - Claude Sonnet 4 (best reasoning)")
    print("  • claude_haiku    - Claude Haiku 4.5 (fast)")
    print("  • groq            - Llama 3.3 70B (fast inference)")
    print("  • grok            - xAI Grok Beta (NEW!)")
    print("  • vertex_qwen     - Qwen 480B (specialized coding)")
    print("\nUsage:")
    print('  response, tokens = call_llm(prompt, "grok")')
    print('  cost = calculate_cost("grok", tokens["input"], tokens["output"])')
    print("="*70)