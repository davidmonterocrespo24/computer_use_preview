"""
LLM Client - Interface to various LLM providers.
Supports OpenAI, Anthropic, and local models (including vision models via Ollama).
"""
from typing import Dict, List, Any, Optional, Literal
import json
import logging
import os
import requests
from dataclasses import dataclass

log = logging.getLogger(__name__)


@dataclass
class LLMResponse:
    """Response from LLM."""
    content: str
    reasoning: Optional[str] = None
    actions: Optional[List[Dict[str, Any]]] = None
    finish_reason: str = "stop"
    elapsed_sec: Optional[float] = None   # wall-clock time for the LLM call
    
    def parse_actions(self) -> List[Dict[str, Any]]:
        """Parse actions from response content."""
        if self.actions:
            return self.actions
        
        # Try to extract JSON from response
        try:
            # Look for JSON block in response
            if "```json" in self.content:
                json_start = self.content.find("```json") + 7
                json_end = self.content.find("```", json_start)
                json_str = self.content[json_start:json_end].strip()
                data = json.loads(json_str)
                if isinstance(data, list):
                    return data
                elif isinstance(data, dict) and "actions" in data:
                    return data["actions"]
        except Exception:
            pass
        
        return []


class LLMClient:
    """Client for interfacing with LLM providers."""
    
    def __init__(
        self,
        provider: Literal["openai", "anthropic", "local"] = "openai",
        model: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 4096
    ):
        """
        Initialize LLM client.
        
        Args:
            provider: LLM provider (openai, anthropic, local)
            model: Model name/ID
            api_key: API key (if not in environment)
            base_url: Base URL for local models
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
        """
        self.provider = provider
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        # Initialize provider-specific client
        if provider == "openai":
            from openai import OpenAI
            self.client = OpenAI(
                api_key=api_key or os.getenv("OPENAI_API_KEY")
            )
        elif provider == "anthropic":
            from anthropic import Anthropic
            self.client = Anthropic(
                api_key=api_key or os.getenv("ANTHROPIC_API_KEY")
            )
        elif provider == "local":
            from openai import OpenAI
            # Use OpenAI-compatible interface for local models
            self.client = OpenAI(
                api_key=api_key or "dummy",
                base_url=base_url or os.getenv("LOCAL_LLM_BASE_URL", "http://localhost:8000/v1")
            )
        else:
            raise ValueError(f"Unknown provider: {provider}")
        log.debug("LLMClient initialized | provider=%s | model=%s | base_url=%s", provider, model, base_url)

    def generate(
        self,
        messages: List[Dict[str, str]],
        system_prompt: Optional[str] = None,
        tools: Optional[List[Dict]] = None
    ) -> LLMResponse:
        """
        Generate completion from LLM.
        
        Args:
            messages: Conversation messages
            system_prompt: System prompt
            tools: Tool definitions for function calling
            
        Returns:
            LLMResponse object
        """
        if self.provider == "anthropic":
            return self._generate_anthropic(messages, system_prompt, tools)
        else:
            # OpenAI and local models use same interface
            return self._generate_openai(messages, system_prompt, tools)
    
    def _generate_openai(
        self,
        messages: List[Dict[str, str]],
        system_prompt: Optional[str] = None,
        tools: Optional[List[Dict]] = None
    ) -> LLMResponse:
        """Generate using OpenAI API."""
        # Add system message if provided
        if system_prompt:
            messages = [{"role": "system", "content": system_prompt}] + messages
        
        kwargs = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens
        }
        
        # Add tools if provided
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"
        
        response = self.client.chat.completions.create(**kwargs)
        
        message = response.choices[0].message
        content = message.content or ""
        
        # Extract tool calls if any
        actions = []
        if hasattr(message, 'tool_calls') and message.tool_calls:
            for tool_call in message.tool_calls:
                actions.append({
                    "name": tool_call.function.name,
                    "arguments": json.loads(tool_call.function.arguments)
                })
        
        return LLMResponse(
            content=content,
            actions=actions if actions else None,
            finish_reason=response.choices[0].finish_reason
        )
    
    def _generate_anthropic(
        self,
        messages: List[Dict[str, str]],
        system_prompt: Optional[str] = None,
        tools: Optional[List[Dict]] = None
    ) -> LLMResponse:
        """Generate using Anthropic API."""
        kwargs = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens
        }
        
        if system_prompt:
            kwargs["system"] = system_prompt
        
        if tools:
            kwargs["tools"] = tools
        
        response = self.client.messages.create(**kwargs)
        
        content = ""
        actions = []
        
        for block in response.content:
            if block.type == "text":
                content += block.text
            elif block.type == "tool_use":
                actions.append({
                    "name": block.name,
                    "arguments": block.input
                })
        
        return LLMResponse(
            content=content,
            actions=actions if actions else None,
            finish_reason=response.stop_reason
        )
    
    def generate_with_retry(
        self,
        messages: List[Dict[str, str]],
        system_prompt: Optional[str] = None,
        tools: Optional[List[Dict]] = None,
        max_retries: int = 3
    ) -> LLMResponse:
        """Generate with automatic retry on failure. Records elapsed_sec on response."""
        import time
        from tenacity import retry, stop_after_attempt, wait_exponential

        @retry(
            stop=stop_after_attempt(max_retries),
            wait=wait_exponential(multiplier=1, min=2, max=10)
        )
        def _generate():
            return self.generate(messages, system_prompt, tools)

        t0 = time.perf_counter()
        response = _generate()
        response.elapsed_sec = time.perf_counter() - t0
        log.debug("Text model response | model=%s | elapsed=%.1fs", self.model, response.elapsed_sec)
        return response

    @property
    def is_vision_model(self) -> bool:
        """Return True if this client's model supports image input."""
        return self.model in {
            "qwen3-vl:4b", "qwen3-vl:8b",
            "granite3.2-vision:latest",
            "moondream:latest",
        }

    def generate_with_image(
        self,
        messages: List[Dict[str, str]],
        system_prompt: Optional[str],
        image_b64: str,
    ) -> LLMResponse:
        """
        Send a request to an Ollama vision model with an annotated screenshot.

        Uses the Ollama native /api/chat endpoint which accepts an 'images' field.
        The image should be a base64-encoded PNG (no data URL prefix).

        Args:
            messages: Conversation messages
            system_prompt: System prompt text
            image_b64: Base64-encoded screenshot (plain base64, not data URL)

        Returns:
            LLMResponse with the model's analysis
        """
        # Derive Ollama base from the OpenAI-compat base_url
        # e.g. "http://localhost:11434/v1" → "http://localhost:11434"
        if hasattr(self, 'client') and hasattr(self.client, 'base_url'):
            raw_base = str(self.client.base_url).rstrip("/")
            if raw_base.endswith("/v1"):
                ollama_base = raw_base[:-3]
            else:
                ollama_base = raw_base
        else:
            ollama_base = os.getenv("LOCAL_LLM_BASE_URL", "http://localhost:11434/v1").replace("/v1", "")

        # Build messages for Ollama API — attach image to the last user message
        ollama_messages = []
        if system_prompt:
            ollama_messages.append({"role": "system", "content": system_prompt})

        for i, msg in enumerate(messages):
            if msg["role"] == "user" and i == len(messages) - 1:
                # Attach image to the last user message
                ollama_messages.append({
                    "role": "user",
                    "content": msg["content"],
                    "images": [image_b64],
                })
            else:
                ollama_messages.append({"role": msg["role"], "content": msg["content"]})

        payload = {
            "model": self.model,
            "messages": ollama_messages,
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_tokens,
            },
        }

        img_kb = len(image_b64) * 3 // 4 // 1024  # approx decoded size
        log.info(
            "Vision API call | model=%s | endpoint=%s/api/chat | image_size~%dKB | messages=%d",
            self.model, ollama_base, img_kb, len(ollama_messages),
        )
        import time
        t0 = time.perf_counter()
        try:
            resp = requests.post(
                f"{ollama_base}/api/chat",
                json=payload,
                timeout=120,
            )
            resp.raise_for_status()
            data = resp.json()
            content = data.get("message", {}).get("content", "")
            elapsed = time.perf_counter() - t0
            log.info(
                "Vision API response | model=%s | elapsed=%.1fs | response_chars=%d | done_reason=%s",
                self.model, elapsed, len(content), data.get("done_reason", "?"),
            )
            return LLMResponse(
                content=content,
                finish_reason=data.get("done_reason", "stop"),
                elapsed_sec=elapsed,
            )
        except requests.RequestException as e:
            elapsed = time.perf_counter() - t0
            log.error("Vision API error | model=%s | elapsed=%.1fs | error=%s", self.model, elapsed, e)
            return LLMResponse(
                content=f"[Vision model error: {e}]",
                finish_reason="error",
                elapsed_sec=elapsed,
            )


def create_browser_tools() -> List[Dict[str, Any]]:
    """Create tool definitions for browser actions."""
    return [
        {
            "type": "function",
            "function": {
                "name": "navigate",
                "description": "Navigate to a URL",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "url": {
                            "type": "string",
                            "description": "The URL to navigate to"
                        }
                    },
                    "required": ["url"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "click",
                "description": "Click on an element by its ID",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "element_id": {
                            "type": "string",
                            "description": "The element ID to click (e.g., 'elem_0')"
                        }
                    },
                    "required": ["element_id"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "type_text",
                "description": "Type text into an input field",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "element_id": {
                            "type": "string",
                            "description": "The element ID to type into"
                        },
                        "text": {
                            "type": "string",
                            "description": "The text to type"
                        },
                        "press_enter": {
                            "type": "boolean",
                            "description": "Whether to press Enter after typing",
                            "default": False
                        }
                    },
                    "required": ["element_id", "text"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "scroll",
                "description": "Scroll the page in a direction",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "direction": {
                            "type": "string",
                            "enum": ["up", "down", "left", "right"],
                            "description": "Direction to scroll"
                        }
                    },
                    "required": ["direction"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "go_back",
                "description": "Go back in browser history",
                "parameters": {
                    "type": "object",
                    "properties": {}
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "wait",
                "description": "Wait for a specified number of seconds",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "seconds": {
                            "type": "integer",
                            "description": "Number of seconds to wait",
                            "default": 2
                        }
                    }
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "task_complete",
                "description": "Mark the task as complete and provide the final result",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "result": {
                            "type": "string",
                            "description": "The result or answer to the user's query"
                        },
                        "success": {
                            "type": "boolean",
                            "description": "Whether the task was completed successfully"
                        }
                    },
                    "required": ["result", "success"]
                }
            }
        }
    ]
