import requests
import anthropic
import os
import concurrent.futures as _cf
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables from .env file
load_dotenv()


def call_llm(
    prompt: str,
    provider: str,
    prefill: str = None,
    stop_sequences: list = None,
    system_prompt: str = None,
):
    """
    Universal LLM caller - routes to appropriate provider.
    Returns: (response_text, token_dict)

    Args:
        prompt: The prompt text (dynamic / query-specific content).
        provider: LLM provider code.
        prefill: Optional prefilled assistant response start (Claude only).
        stop_sequences: Optional list of stop sequences.
        system_prompt: Optional static system prompt (Claude only).
            When provided the content is sent with cache_control so Anthropic
            can cache it across calls — saving cost on repeated schema/rules.
            Non-Claude providers silently ignore this parameter.

    Supported providers:
    - nvidia_qwen3: NVIDIA Qwen 3 Next 80B (with thinking)
    - vertex_qwen_thinking: Vertex AI Qwen3-Next-80B Thinking (reviewer)
    - vertex_kimi_k2_thinking: Vertex AI Kimi K2 Thinking
    - o1_mini: OpenAI o1-mini (best reasoning/price)
    - o1: OpenAI o1 (best reasoning, expensive)
    - claude_sonnet: Claude Sonnet 4.5
    - claude_opus: Claude Opus 4.5
    - claude_haiku: Claude Haiku 4.5
    - groq: Groq Llama 3.3 70B
    - grok: xAI Grok Beta
    - vertex_qwen: Qwen 2.5 Coder 32B
    """
    if provider == "nvidia_qwen3":
        return call_nvidia_qwen3(prompt, stop_sequences)
    elif provider == "vertex_qwen_thinking":
        return call_vertex_qwen_thinking(prompt, stop_sequences)
    elif provider == "vertex_kimi_k2_thinking":
        return call_vertex_kimi_k2_thinking(prompt, stop_sequences, prefill=prefill)
    elif provider == "o1_mini":
        return call_o1_mini(prompt)
    elif provider == "o1":
        return call_o1(prompt)
    elif provider == "claude_sonnet":
        return call_claude_sonnet(prompt, prefill, stop_sequences, system_prompt)
    elif provider == "claude_opus":
        return call_claude_opus(prompt, prefill, stop_sequences, system_prompt)
    elif provider == "claude_haiku":
        return call_claude_haiku(prompt, prefill, stop_sequences, system_prompt)
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



def call_claude_sonnet(
    prompt: str,
    prefill: str = None,
    stop_sequences: list = None,
    system_prompt: str = None,
):
    """
    Call Claude Sonnet 4.5 for reasoning and analysis.

    Args:
        prompt: The prompt text (dynamic content — question, plan, etc.)
        prefill: Optional prefilled assistant response (e.g., "{" for JSON)
        stop_sequences: Optional list of strings to stop generation
        system_prompt: Optional static system prompt sent with cache_control.
            Use for schema + rules + dialect to enable Anthropic prompt caching.

    Returns: (response_text, token_dict)
        token_dict keys: input, output,
                         cache_creation_input_tokens, cache_read_input_tokens
    """
    client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

    messages = [{"role": "user", "content": prompt}]
    if prefill:
        messages.append({"role": "assistant", "content": prefill})

    params = {
        "model": "claude-sonnet-4-5-20250929",
        "max_tokens": 4096,
        "temperature": 0.0,
        "messages": messages,
    }

    # Attach cacheable system prompt when provided
    if system_prompt:
        params["system"] = [
            {
                "type": "text",
                "text": system_prompt,
                "cache_control": {"type": "ephemeral"},
            }
        ]

    if stop_sequences:
        params["stop_sequences"] = stop_sequences

    message = client.messages.create(**params)

    tokens = {
        "input": message.usage.input_tokens,
        "output": message.usage.output_tokens,
        "cache_creation_input_tokens": getattr(message.usage, "cache_creation_input_tokens", 0) or 0,
        "cache_read_input_tokens": getattr(message.usage, "cache_read_input_tokens", 0) or 0,
    }

    response_text = message.content[0].text
    if prefill:
        response_text = prefill + response_text

    return response_text, tokens


def call_claude_opus(
    prompt: str,
    prefill: str = None,
    stop_sequences: list = None,
    system_prompt: str = None,
):
    """
    Call Claude Opus 4.5 for deep validation and review.
    Most accurate model — use for critical review tasks.

    Args:
        prompt: The prompt text (dynamic content)
        prefill: Optional prefilled assistant response (e.g., "{" for JSON)
        stop_sequences: Optional list of strings to stop generation
        system_prompt: Optional static system prompt sent with cache_control.

    Returns: (response_text, token_dict)
        token_dict keys: input, output,
                         cache_creation_input_tokens, cache_read_input_tokens
    """
    client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

    messages = [{"role": "user", "content": prompt}]
    if prefill:
        messages.append({"role": "assistant", "content": prefill})

    params = {
        "model": "claude-opus-4-5-20251101",
        "max_tokens": 4096,
        "temperature": 0.0,
        "messages": messages,
    }

    if system_prompt:
        params["system"] = [
            {
                "type": "text",
                "text": system_prompt,
                "cache_control": {"type": "ephemeral"},
            }
        ]

    if stop_sequences:
        params["stop_sequences"] = stop_sequences

    message = client.messages.create(**params)

    tokens = {
        "input": message.usage.input_tokens,
        "output": message.usage.output_tokens,
        "cache_creation_input_tokens": getattr(message.usage, "cache_creation_input_tokens", 0) or 0,
        "cache_read_input_tokens": getattr(message.usage, "cache_read_input_tokens", 0) or 0,
    }

    response_text = message.content[0].text
    if prefill:
        response_text = prefill + response_text

    return response_text, tokens


def call_claude_haiku(
    prompt: str,
    prefill: str = None,
    stop_sequences: list = None,
    system_prompt: str = None,
):
    """
    Call Claude Haiku 4.5 for fast reasoning.

    Args:
        prompt: The prompt text (dynamic content)
        prefill: Optional prefilled assistant response (e.g., "{" for JSON)
        stop_sequences: Optional list of strings to stop generation
        system_prompt: Optional static system prompt sent with cache_control.

    Returns: (response_text, token_dict)
        token_dict keys: input, output,
                         cache_creation_input_tokens, cache_read_input_tokens
    """
    client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

    messages = [{"role": "user", "content": prompt}]
    if prefill:
        messages.append({"role": "assistant", "content": prefill})

    params = {
        "model": "claude-haiku-4-5-20251001",
        "max_tokens": 4096,
        "temperature": 0.0,
        "messages": messages,
    }

    if system_prompt:
        params["system"] = [
            {
                "type": "text",
                "text": system_prompt,
                "cache_control": {"type": "ephemeral"},
            }
        ]

    if stop_sequences:
        params["stop_sequences"] = stop_sequences

    message = client.messages.create(**params)

    tokens = {
        "input": message.usage.input_tokens,
        "output": message.usage.output_tokens,
        "cache_creation_input_tokens": getattr(message.usage, "cache_creation_input_tokens", 0) or 0,
        "cache_read_input_tokens": getattr(message.usage, "cache_read_input_tokens", 0) or 0,
    }

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
    Call Qwen Coder via Vertex AI MaaS endpoint
    Returns: (response_text, token_dict)
    """
    import google.auth.transport.requests
    from google.oauth2 import service_account
    
    PROJECT_ID = "llm-test-491910"
    LOCATION = "us-south1"
    MODEL_NAME = "qwen3-coder-480b-a35b-instruct"
    SERVICE_ACCOUNT_JSON = (
        r"D:\cDRIVE\Test\project_llm_decision_with_rag_v14_with_analyzer_need_to_test\google_json"
        r"\llm-test-491910-88c51981e0b6.json"
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
    Call Qwen via Vertex AI MaaS endpoint for review/thinking tasks.
    Uses the same Qwen Coder model (which works) with different temperature.
    
    Returns: (response_text, token_dict)
    """
    import google.auth.transport.requests
    from google.oauth2 import service_account
    
    PROJECT_ID = "robust-carver-481011-c9"
    LOCATION = "us-south1"
    # Use same model as Coder - it works and is capable of review
    MODEL_NAME = "qwen3-next-80b-a3b-thinking"
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
            "temperature": 0.6,
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

def call_vertex_kimi_k2_thinking(prompt: str, stop_sequences: list = None, debug: bool = True, prefill: str = None):
    """
    Call Kimi K2 Thinking via Vertex AI MaaS endpoint.
    Returns: (response_text, token_dict)
    """

    import requests
    import json
    import google.auth.transport.requests
    from google.oauth2 import service_account

    PROJECT_ID = "llm-test-491910"
    LOCATION = "global"
    MODEL_NAME = "moonshotai/kimi-k2-thinking-maas"   # ✅ FIXED
    SERVICE_ACCOUNT_JSON = (
        r"D:\cDRIVE\Test\project_llm_decision_with_rag_v14_with_analyzer_need_to_test\google_json"
        r"\llm-test-491910-88c51981e0b6.json"
    )

    try:
        # ---------------- AUTH ----------------
        credentials = service_account.Credentials.from_service_account_file(
            SERVICE_ACCOUNT_JSON,
            scopes=["https://www.googleapis.com/auth/cloud-platform"]
        )

        auth_req = google.auth.transport.requests.Request()
        credentials.refresh(auth_req)

        # ✅ FIXED ENDPOINT (NO publisher path here)
        if LOCATION == "global":
            base_url = "https://aiplatform.googleapis.com"
        else:
            base_url = f"https://{LOCATION}-aiplatform.googleapis.com"

        url = (
            f"{base_url}/v1/"
            f"projects/{PROJECT_ID}/locations/{LOCATION}/"
            f"endpoints/openapi/chat/completions"
        )

        headers = {
            "Authorization": f"Bearer {credentials.token}",
            "Content-Type": "application/json"
        }

        # ✅ FIXED PAYLOAD (OpenAI-style)
        messages = [{"role": "user", "content": prompt}]
        if prefill:
            messages.append({"role": "assistant", "content": prefill})
        payload = {
            "model": MODEL_NAME,
            "messages": messages,
            "temperature": 0.6,
            "max_tokens": 2000
        }

        if stop_sequences:
            payload["stop"] = stop_sequences

        if debug:
            print("\n===== REQUEST DEBUG =====")
            print("URL:", url)
            print(json.dumps(payload, indent=2)[:1000])

        # ---------------- API CALL (with 429 retry + total-time cap) ----------------
        import time as _time
        _max_retries = 3
        _retry_delays = [2, 4, 8]  # seconds between attempts
        _TOTAL_TIMEOUT = 150        # hard cap per attempt (handles slow-streaming hangs)

        for _attempt in range(_max_retries):
            _fut = _cf.ThreadPoolExecutor(max_workers=1).submit(
                requests.post, url, headers=headers, json=payload, timeout=90
            )
            try:
                response = _fut.result(timeout=_TOTAL_TIMEOUT)
            except _cf.TimeoutError:
                print(f"[KIMI K2] Total timeout ({_TOTAL_TIMEOUT}s) on attempt "
                      f"{_attempt + 1}/{_max_retries}")
                return "", {"input": 0, "output": 0, "error": f"total_timeout_{_TOTAL_TIMEOUT}s"}

            if debug:
                print("\n===== RESPONSE STATUS =====")
                print("Status Code:", response.status_code)

            if response.status_code == 200:
                break  # success — proceed to parse

            if response.status_code == 429 and _attempt < _max_retries - 1:
                _wait = _retry_delays[_attempt]
                print(f"[KIMI K2] 429 rate-limit on attempt {_attempt + 1}/{_max_retries}. "
                      f"Retrying in {_wait}s...")
                _time.sleep(_wait)
                continue

            # Non-429 error OR final attempt — log and return empty
            print("\n===== ERROR RESPONSE =====")
            try:
                print(json.dumps(response.json(), indent=2))
            except Exception:
                print(response.text)
            return "", {"input": 0, "output": 0, "error_code": response.status_code}

        data = response.json()

        if debug:
            print("\n===== FULL RESPONSE (trimmed) =====")
            print(json.dumps(data, indent=2)[:2000])

        # ✅ Robust response parsing (OpenAI-style variants)
        response_text = ""
        tokens = {"input": 0, "output": 0}
        content_source = "none"

        choices = data.get("choices", [])
        if choices:
            message = choices[0].get("message", {}) or {}
            content = message.get("content", "")

            # Most common case: plain string content
            if isinstance(content, str) and content.strip():
                response_text = content
                content_source = "content_string"

            # Some providers return list parts:
            # [{"type":"text","text":"..."}, ...] or [{"text":"..."}]
            elif isinstance(content, list):
                parts = []
                for part in content:
                    if isinstance(part, dict):
                        if isinstance(part.get("text"), str) and part.get("text").strip():
                            parts.append(part["text"])
                        elif part.get("type") == "text" and isinstance(part.get("content"), str) and part.get("content").strip():
                            parts.append(part["content"])
                    elif isinstance(part, str) and part.strip():
                        parts.append(part)
                if parts:
                    response_text = "\n".join(parts)
                    content_source = "content_parts"

            # Fallback for thinking models that may emit reasoning in a separate field
            if not response_text:
                reasoning_content = message.get("reasoning_content", "")
                if isinstance(reasoning_content, str) and reasoning_content.strip():
                    response_text = reasoning_content
                    content_source = "reasoning_content"

        usage = data.get("usage", {})
        tokens = {
            "input": usage.get("prompt_tokens", 0),
            "output": usage.get("completion_tokens", 0)
        }

        if debug:
            print(f"[KIMI PARSE] source={content_source}, output_len={len(response_text)}")

        # Always warn when tokens were consumed but output is empty (not just in debug mode)
        if (tokens.get("input", 0) + tokens.get("output", 0)) > 0 and not response_text:
            msg_keys = list((choices[0].get("message", {}) or {}).keys()) if choices else []
            print(f"[KIMI K2 WARNING] tokens>0 but empty output. message_keys={msg_keys}")

        return response_text, tokens

    except Exception as e:
        print("\n===== EXCEPTION =====")
        print("Error Type:", type(e).__name__)
        print("Error Message:", str(e))

        return "", {"input": 0, "output": 0, "error": str(e)}

"""
def call_vertex_kimi_k2_thinking(prompt: str, stop_sequences: list = None):

    Call Kimi K2 Thinking via Vertex AI MaaS endpoint.

    Returns: (response_text, token_dict)

    import google.auth.transport.requests
    from google.oauth2 import service_account

    PROJECT_ID = "llm-test-491910"
    LOCATION = "us-central1"
    MODEL_NAME = "kimi-k2-thinking"
    SERVICE_ACCOUNT_JSON = (
        r"D:\cDRIVE\Test\project_llm_decision_with_rag_v14_with_analyzer_need_to_test\google_json"
        r"\llm-test-491910-88c51981e0b6.json"
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
        f"publishers/moonshotai/models/{MODEL_NAME}-maas:generateContent"
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
            "temperature": 0.6,
            "maxOutputTokens": 4096
        }
    }

    response = requests.post(url, headers=headers, json=payload, timeout=90)
    response.raise_for_status()

    data = response.json()

    usage_metadata = data.get("usageMetadata", {})
    tokens = {
        "input": usage_metadata.get("promptTokenCount", 0),
        "output": usage_metadata.get("candidatesTokenCount", 0)
    }

    response_text = data["candidates"][0]["content"]["parts"][0]["text"]

    return response_text, tokens
"""

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
        "vertex_qwen": {"input": 0.0, "output": 0.0},  # Variable
        "vertex_qwen_thinking": {"input": 0.0, "output": 0.0},  # Variable
        "vertex_kimi_k2_thinking": {"input": 0.0, "output": 0.0},  # Variable
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
    print("  • vertex_qwen_thinking - Qwen3 Next Thinking via Vertex")
    print("  • vertex_kimi_k2_thinking - Kimi K2 Thinking via Vertex")
    print("\nUsage:")
    print('  response, tokens = call_llm(prompt, "grok")')
    print('  cost = calculate_cost("grok", tokens["input"], tokens["output"])')
    print("="*70)
