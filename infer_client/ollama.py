"""Ollama client for local LLM inference."""

from typing import Any, Dict, List, Optional

import requests


class OllamaClient:
    """Client for Ollama local LLM inference.

    Usage::

        client = OllamaClient()  # defaults to localhost:11434
        response = client.generate("llama3", "What is machine learning?")
        print(response)
    """

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        timeout: int = 600,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._session = requests.Session()

    def models(self) -> List[Dict[str, Any]]:
        """List available models."""
        r = self._session.get(
            f"{self.base_url}/api/tags",
            timeout=self.timeout,
        )
        r.raise_for_status()
        return r.json().get("models", [])

    def generate(
        self,
        model: str,
        prompt: str,
        system: Optional[str] = None,
        stream: bool = False,
        options: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Generate text using a local LLM.

        Args:
            model: Model name (e.g., "llama3", "mistral", "phi3").
            prompt: The prompt to send to the model.
            system: Optional system message.
            stream: Whether to stream the response (default False).
            options: Optional model parameters (temperature, top_p, etc.).

        Returns:
            The generated text.
        """
        payload: Dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "stream": stream,
        }

        if system is not None:
            payload["system"] = system

        if options is not None:
            payload["options"] = options

        r = self._session.post(
            f"{self.base_url}/api/generate",
            json=payload,
            timeout=self.timeout,
        )
        r.raise_for_status()
        return r.json().get("response", "")

    def chat(
        self,
        model: str,
        messages: List[Dict[str, str]],
        stream: bool = False,
        options: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Chat with a local LLM (multi-turn conversation).

        Args:
            model: Model name.
            messages: List of message dicts with 'role' and 'content' keys.
            stream: Whether to stream the response.
            options: Optional model parameters.

        Returns:
            Dict with 'response' (text), 'messages' (updated list), and metadata.
        """
        payload: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": stream,
        }

        if options is not None:
            payload["options"] = options

        r = self._session.post(
            f"{self.base_url}/api/chat",
            json=payload,
            timeout=self.timeout,
        )
        r.raise_for_status()
        result = r.json()

        updated_messages = messages + [result.get("message", {})]

        return {
            "response": result.get("message", {}).get("content", ""),
            "messages": updated_messages,
            "model": result.get("model", model),
            "done": result.get("done", True),
        }

    def pull(self, model: str) -> None:
        """Pull (download) a model from Ollama registry.

        Args:
            model: Model name to pull.
        """
        r = self._session.post(
            f"{self.base_url}/api/pull",
            json={"name": model, "stream": False},
            timeout=600,  # pulling can take a while
        )
        r.raise_for_status()

    def __repr__(self) -> str:
        return f"OllamaClient(base_url={self.base_url!r})"
