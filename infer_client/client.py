"""Inference API client — multi-model support."""

from typing import Any, Callable, Dict, List, Optional, Union

import requests


class InferClient:
    """Client for the Transformer Inference API.

    Usage::

        client = InferClient("https://your-server.example.com", api_key="...")
        result = client.infer("The market is crashing")
        print(result)
    """

    def __init__(
        self,
        base_url: str,
        api_key: str,
        timeout: int = 600,
        verify_ssl: bool = True,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self.verify_ssl = verify_ssl
        self._session = requests.Session()
        self._session.headers.update({
            "X-API-Key": self.api_key,
            "Content-Type": "application/json",
        })
        self._session.verify = self.verify_ssl

    # ──────────────────────────────────────────────
    # Public endpoints (no auth)
    # ──────────────────────────────────────────────

    def health(self) -> Dict[str, Any]:
        """Check API health and list loaded models."""
        r = self._session.get(f"{self.base_url}/health", timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    def models(self) -> List[Dict[str, Any]]:
        """List all available models."""
        r = self._session.get(f"{self.base_url}/models", timeout=self.timeout)
        r.raise_for_status()
        return r.json()["models"]

    def model_info(self, model_id: str) -> Dict[str, Any]:
        """Get full metadata for a specific model."""
        r = self._session.get(f"{self.base_url}/models/{model_id}", timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    def resources(self) -> Dict[str, Any]:
        """Get server resource status (CPU, memory, GPU, capacity)."""
        r = self._session.get(f"{self.base_url}/resources", timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    def capabilities(self) -> Dict[str, Any]:
        """Get full server capabilities: models, Ollama, resources."""
        r = self._session.get(f"{self.base_url}/capabilities", timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    def ollama_status(self) -> Dict[str, Any]:
        """Get Ollama service status from the server."""
        r = self._session.get(f"{self.base_url}/ollama/status", timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    def ollama_models(self) -> List[Dict[str, Any]]:
        """List Ollama models available on the server."""
        r = self._session.get(f"{self.base_url}/ollama/models", timeout=self.timeout)
        r.raise_for_status()
        return r.json().get("models", [])

    # ──────────────────────────────────────────────
    # Inference (auth required)
    # ──────────────────────────────────────────────

    def model_config(self, model_id: str, n_texts: int = 100) -> Dict[str, Any]:
        """Get optimal inference configuration for a model.

        Args:
            model_id: Model ID.
            n_texts: Expected number of texts (affects parallel recommendation).

        Returns:
            Dict with batch_size, device_mode, use_parallel, training_mode, etc.
        """
        r = self._session.get(
            f"{self.base_url}/models/{model_id}/config",
            params={"n_texts": n_texts},
            timeout=self.timeout,
        )
        r.raise_for_status()
        return r.json()

    def infer(
        self,
        text: Optional[str] = None,
        texts: Optional[List[str]] = None,
        model: Optional[str] = None,
        batch_size: int = 32,
        max_length: int = 256,
        temperature: float = 0.7,
        parallel: bool = False,
        device_mode: str = "both",
        threshold: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Run inference on one or more texts.

        Args:
            text: Single text input.
            texts: Batch of texts (mutually exclusive with text).
            model: Model ID. None uses the server default.
            batch_size: Batch size for processing.
            max_length: Max generation length (generative models only).
            temperature: Sampling temperature (generative models only).
            parallel: Use parallel GPU+CPU inference (for large batches).
            device_mode: Device mode for parallel: 'cpu', 'gpu', or 'both'.
            threshold: Override multi-label threshold (0.0-1.0).

        Returns:
            Dict with 'results', 'count', 'model_type', 'model_id',
            and for classification: 'training_mode', 'multi_label',
            'multi_label_threshold', 'num_labels', 'labels'.
        """
        if text is None and texts is None:
            raise ValueError("Provide 'text' or 'texts'")

        payload: Dict[str, Any] = {"batch_size": batch_size}
        if texts is not None:
            payload["texts"] = texts
        else:
            payload["text"] = text
        payload["max_length"] = max_length
        payload["temperature"] = temperature

        # Parallel inference options
        if parallel:
            payload["parallel"] = True
            payload["device_mode"] = device_mode

        # Multi-label threshold override
        if threshold is not None:
            payload["threshold"] = threshold

        if model:
            url = f"{self.base_url}/models/{model}/infer"
        else:
            url = f"{self.base_url}/infer"

        r = self._session.post(url, json=payload, timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    def classify(
        self,
        text: Union[str, List[str]],
        model: Optional[str] = None,
        threshold: Optional[float] = None,
        parallel: bool = False,
    ) -> List[Dict[str, Any]]:
        """Shortcut: classify and return just the results list.

        For multi-label models, each result contains 'labels' (list) instead of 'label'.
        Use threshold to override the model's default multi-label threshold.
        """
        if isinstance(text, str):
            resp = self.infer(text=text, model=model, threshold=threshold, parallel=parallel)
        else:
            resp = self.infer(texts=text, model=model, threshold=threshold, parallel=parallel)
        return resp["results"]

    def segment_sentences(
        self,
        text: Union[str, List[str]],
        model: str = "wtpsplit",
        mode: str = "sentence",
    ) -> List[Dict[str, Any]]:
        """Segment text into sentences using WTPSPLIT.

        WTPSPLIT is a fast multilingual sentence segmentation model
        (https://github.com/segment-any-text/wtpsplit) that supports 85+ languages.

        Args:
            text: Single text or list of texts to segment.
            model: WTPSPLIT model ID (default: "wtpsplit").
            mode: Segmentation mode: 'sentence' (default) or 'newline'.

        Returns:
            List of segmentation results with 'sentences' list for each input text.

        Example:
            >>> client.segment_sentences("First sentence. Second sentence.")
            [{'text': '...', 'sentences': ['First sentence.', 'Second sentence.'], 'sentence_count': 2}]
        """
        if isinstance(text, str):
            payload = {"text": text, "mode": mode}
        else:
            payload = {"texts": text, "mode": mode}

        r = self._session.post(
            f"{self.base_url}/models/{model}/segment",
            json=payload,
            timeout=self.timeout,
        )
        r.raise_for_status()
        return r.json()["results"]

    def extract_entities(
        self,
        text: Union[str, List[str]],
        labels: List[str],
        model: str = "gliner",
        threshold: float = 0.5,
        flat_ner: bool = True,
    ) -> List[Dict[str, Any]]:
        """Extract named entities using GLiNER zero-shot NER.

        GLiNER is a third-party model (https://github.com/urchade/GLiNER) that
        supports multilingual entity extraction with custom labels. It can extract
        ANY entity type without training.

        Args:
            text: Single text or list of texts to analyze.
            labels: Entity types to extract (e.g., ["person", "organization", "location"]).
                    Can be ANY entity type: "political party", "disease", "product", etc.
            model: NER model ID (default: "gliner").
            threshold: Confidence threshold (0.0-1.0, default: 0.5).
            flat_ner: Resolve overlapping entities (default: True).

        Returns:
            List of results, each containing:
            - 'text': Input text
            - 'entities': List of found entities with:
              - 'text': Entity text
              - 'label': Entity type
              - 'start': Start character position
              - 'end': End character position
              - 'score': Confidence score
            - 'entity_count': Number of entities found
            - 'labels_used': Labels that were searched for

        Examples:
            >>> # Extract people and organizations
            >>> results = client.extract_entities(
            ...     "Apple Inc. was founded by Steve Jobs",
            ...     labels=["person", "organization"]
            ... )
            >>> results[0]['entities']
            [
                {'text': 'Apple Inc.', 'label': 'organization', 'start': 0, 'end': 10, 'score': 0.91},
                {'text': 'Steve Jobs', 'label': 'person', 'start': 26, 'end': 36, 'score': 0.99}
            ]

            >>> # Extract custom entity types
            >>> results = client.extract_entities(
            ...     "The Democratic Party won the election",
            ...     labels=["political party", "event"]
            ... )

            >>> # Multilingual support (12+ languages)
            >>> results = client.extract_entities(
            ...     "Emmanuel Macron est président de la France",
            ...     labels=["person", "country", "job title"]
            ... )

        Note:
            - Supports 12+ languages: EN, FR, DE, ES, IT, PT, NL, RU, ZH, JA, AR
            - Context window: 512 tokens
            - GLiNER model credit: urchade/GLiNER (not trained by LLM Tool)
        """
        if not labels:
            raise ValueError("Must provide at least one label")

        texts_list = [text] if isinstance(text, str) else text

        payload = {
            "texts": texts_list,
            "labels": labels,
            "threshold": threshold,
            "flat_ner": flat_ner,
        }

        url = f"{self.base_url}/models/{model}/infer"
        r = self._session.post(url, json=payload, timeout=self.timeout)
        r.raise_for_status()
        return r.json()["results"]

    # ──────────────────────────────────────────────
    # Server-side Ollama (auth required)
    # ──────────────────────────────────────────────

    def ollama_generate(
        self,
        model: str,
        prompt: str,
        system: Optional[str] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Generate text using Ollama on the server.

        Args:
            model: Ollama model name (e.g., "llama3", "mistral").
            prompt: The prompt to send.
            system: Optional system message.
            options: Model options (temperature, top_p, etc.).

        Returns:
            Dict with 'response', 'model', and metadata.
        """
        payload: Dict[str, Any] = {
            "model": model,
            "prompt": prompt,
        }
        if system:
            payload["system"] = system
        if options:
            payload["options"] = options

        r = self._session.post(
            f"{self.base_url}/ollama/generate",
            json=payload,
            timeout=self.timeout,
        )
        r.raise_for_status()
        return r.json()

    def ollama_chat(
        self,
        model: str,
        messages: List[Dict[str, str]],
        options: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Chat with Ollama on the server.

        Args:
            model: Ollama model name.
            messages: List of message dicts with 'role' and 'content'.
            options: Model options.

        Returns:
            Dict with 'response', 'messages', and metadata.
        """
        payload: Dict[str, Any] = {
            "model": model,
            "messages": messages,
        }
        if options:
            payload["options"] = options

        r = self._session.post(
            f"{self.base_url}/ollama/chat",
            json=payload,
            timeout=self.timeout,
        )
        r.raise_for_status()
        return r.json()

    # ──────────────────────────────────────────────
    # DataFrame / CSV helpers (require pandas)
    # ──────────────────────────────────────────────

    def classify_df(
        self,
        df: Any,
        text_column: str,
        model: Optional[str] = None,
        batch_size: int = 32,
        threshold: Optional[float] = None,
        parallel: bool = False,
        on_batch_done: Optional[Callable[[int, int], None]] = None,
    ) -> Any:
        """Classify all rows of a DataFrame.

        For single-label models:
            Adds columns: ``label``, ``confidence``, and ``prob_<label>`` per class.

        For multi-label models:
            Adds columns: ``labels`` (list), ``label_count``, ``threshold``,
            and ``prob_<label>`` per class.

        Requires ``pandas`` (install with ``pip install infer-client[pandas]``).

        Args:
            df: pandas DataFrame.
            text_column: Name of the column containing texts.
            model: Model ID. None uses the server default.
            batch_size: Batch size for API calls.
            threshold: Override multi-label threshold (0.0-1.0).
            parallel: Use parallel GPU+CPU inference.
            on_batch_done: Optional callback ``(processed_so_far, total)``
                called after each batch completes.

        Returns:
            A copy of the DataFrame with prediction columns added.
        """
        try:
            import pandas as pd
        except ImportError:
            raise ImportError(
                "pandas is required for classify_df(). "
                "Install with: pip install infer-client[pandas]"
            )

        texts = df[text_column].astype(str).tolist()
        total = len(texts)

        all_results: List[Dict[str, Any]] = []
        multi_label = False

        for i in range(0, total, batch_size):
            batch = texts[i : i + batch_size]
            resp = self.infer(
                texts=batch,
                model=model,
                batch_size=batch_size,
                threshold=threshold,
                parallel=parallel,
            )
            all_results.extend(resp["results"])
            multi_label = resp.get("multi_label", False)
            if on_batch_done is not None:
                on_batch_done(min(i + len(batch), total), total)

        out = df.copy()

        if multi_label:
            # Multi-label results have 'labels' (list) instead of 'label'
            out["labels"] = [r.get("labels", []) for r in all_results]
            out["label_count"] = [r.get("label_count", 0) for r in all_results]
            out["threshold"] = [r.get("threshold", 0.5) for r in all_results]
        else:
            # Single-label results
            out["label"] = [r.get("label", "") for r in all_results]
            out["confidence"] = [r.get("confidence", 0.0) for r in all_results]

        # Add probability columns for both modes
        if all_results and "probabilities" in all_results[0]:
            prob_keys = sorted(all_results[0]["probabilities"].keys())
            for key in prob_keys:
                out[f"prob_{key}"] = [r["probabilities"].get(key, 0.0) for r in all_results]

        return out

    def classify_csv(
        self,
        input_path: str,
        text_column: str,
        output_path: Optional[str] = None,
        model: Optional[str] = None,
        batch_size: int = 32,
        threshold: Optional[float] = None,
        parallel: bool = False,
        on_batch_done: Optional[Callable[[int, int], None]] = None,
    ) -> str:
        """Classify a CSV file and write results to a new CSV.

        Requires ``pandas`` (install with ``pip install infer-client[pandas]``).

        Args:
            input_path: Path to the input CSV file.
            text_column: Name of the column containing texts.
            output_path: Path for output CSV. Defaults to ``<input>_classified.csv``.
            model: Model ID. None uses the server default.
            batch_size: Batch size for API calls.
            threshold: Override multi-label threshold (0.0-1.0).
            parallel: Use parallel GPU+CPU inference.
            on_batch_done: Optional callback ``(processed_so_far, total)``
                called after each batch completes.

        Returns:
            Path to the output CSV file.
        """
        try:
            import pandas as pd
        except ImportError:
            raise ImportError(
                "pandas is required for classify_csv(). "
                "Install with: pip install infer-client[pandas]"
            )

        df = pd.read_csv(input_path)
        result_df = self.classify_df(
            df,
            text_column,
            model=model,
            batch_size=batch_size,
            threshold=threshold,
            parallel=parallel,
            on_batch_done=on_batch_done,
        )

        if output_path is None:
            stem = input_path.rsplit(".", 1)[0]
            output_path = f"{stem}_classified.csv"

        result_df.to_csv(output_path, index=False)
        return output_path

    def __repr__(self) -> str:
        return f"InferClient(base_url={self.base_url!r})"
