"""Ollama client for local LLM inference."""

from typing import Any, Dict, List, Optional, Tuple

import requests

# TranslateGemma supported languages (base code → English name)
TRANSLATEGEMMA_LANGUAGES: Dict[str, str] = {
    "aa": "Afar", "ab": "Abkhazian", "af": "Afrikaans", "ak": "Akan",
    "am": "Amharic", "an": "Aragonese", "ar": "Arabic", "as": "Assamese",
    "az": "Azerbaijani", "ba": "Bashkir", "be": "Belarusian", "bg": "Bulgarian",
    "bm": "Bambara", "bn": "Bengali", "bo": "Tibetan", "br": "Breton",
    "bs": "Bosnian", "ca": "Catalan", "ce": "Chechen", "co": "Corsican",
    "cs": "Czech", "cv": "Chuvash", "cy": "Welsh", "da": "Danish",
    "de": "German", "dv": "Divehi", "dz": "Dzongkha", "ee": "Ewe",
    "el": "Greek", "en": "English", "eo": "Esperanto", "es": "Spanish",
    "et": "Estonian", "eu": "Basque", "fa": "Persian", "ff": "Fulah",
    "fi": "Finnish", "fil": "Filipino", "fo": "Faroese", "fr": "French",
    "fy": "Western Frisian", "ga": "Irish", "gd": "Scottish Gaelic",
    "gl": "Galician", "gn": "Guarani", "gu": "Gujarati", "gv": "Manx",
    "ha": "Hausa", "he": "Hebrew", "hi": "Hindi", "hr": "Croatian",
    "ht": "Haitian", "hu": "Hungarian", "hy": "Armenian", "ia": "Interlingua",
    "id": "Indonesian", "ie": "Interlingue", "ig": "Igbo", "ii": "Sichuan Yi",
    "ik": "Inupiaq", "io": "Ido", "is": "Icelandic", "it": "Italian",
    "iu": "Inuktitut", "ja": "Japanese", "jv": "Javanese", "ka": "Georgian",
    "ki": "Kikuyu", "kk": "Kazakh", "kl": "Kalaallisut", "km": "Central Khmer",
    "kn": "Kannada", "ko": "Korean", "ks": "Kashmiri", "ku": "Kurdish",
    "kw": "Cornish", "ky": "Kyrgyz", "la": "Latin", "lb": "Luxembourgish",
    "lg": "Ganda", "ln": "Lingala", "lo": "Lao", "lt": "Lithuanian",
    "lu": "Luba-Katanga", "lv": "Latvian", "mg": "Malagasy", "mi": "Maori",
    "mk": "Macedonian", "ml": "Malayalam", "mn": "Mongolian", "mr": "Marathi",
    "ms": "Malay", "mt": "Maltese", "my": "Burmese", "nb": "Norwegian Bokmål",
    "nd": "North Ndebele", "ne": "Nepali", "nl": "Dutch",
    "nn": "Norwegian Nynorsk", "no": "Norwegian", "nr": "South Ndebele",
    "nv": "Navajo", "ny": "Chichewa", "oc": "Occitan", "om": "Oromo",
    "or": "Oriya", "os": "Ossetian", "pa": "Punjabi", "pl": "Polish",
    "ps": "Pashto", "pt": "Portuguese", "qu": "Quechua", "rm": "Romansh",
    "rn": "Rundi", "ro": "Romanian", "ru": "Russian", "rw": "Kinyarwanda",
    "sa": "Sanskrit", "sc": "Sardinian", "sd": "Sindhi",
    "se": "Northern Sami", "sg": "Sango", "si": "Sinhala", "sk": "Slovak",
    "sl": "Slovenian", "sn": "Shona", "so": "Somali", "sq": "Albanian",
    "sr": "Serbian", "ss": "Swati", "st": "Southern Sotho", "su": "Sundanese",
    "sv": "Swedish", "sw": "Swahili", "ta": "Tamil", "te": "Telugu",
    "tg": "Tajik", "th": "Thai", "ti": "Tigrinya", "tk": "Turkmen",
    "tl": "Tagalog", "tn": "Tswana", "to": "Tonga", "tr": "Turkish",
    "ts": "Tsonga", "tt": "Tatar", "ug": "Uyghur", "uk": "Ukrainian",
    "ur": "Urdu", "uz": "Uzbek", "ve": "Venda", "vi": "Vietnamese",
    "vo": "Volapük", "wa": "Walloon", "wo": "Wolof", "xh": "Xhosa",
    "yi": "Yiddish", "yo": "Yoruba", "za": "Zhuang", "zh": "Chinese",
    "zu": "Zulu",
}

_REGIONAL_OVERRIDES: Dict[str, str] = {
    "zh-Hans": "zh", "zh-Hant": "zh", "zh-Latn": "zh",
    "be-tarask": "be", "el-polyton": "el", "hi-Latn": "hi",
}


def _resolve_language(code: str) -> Tuple[str, str]:
    """Resolve a BCP-47 language code to (normalised_code, language_name)."""
    normalised = code.strip().replace("_", "-")
    if normalised in TRANSLATEGEMMA_LANGUAGES:
        return normalised, TRANSLATEGEMMA_LANGUAGES[normalised]
    if normalised in _REGIONAL_OVERRIDES:
        base = _REGIONAL_OVERRIDES[normalised]
        return normalised, TRANSLATEGEMMA_LANGUAGES[base]
    base = normalised.split("-")[0]
    if base in TRANSLATEGEMMA_LANGUAGES:
        return normalised, TRANSLATEGEMMA_LANGUAGES[base]
    raise ValueError(f"Unsupported language code: '{code}'.")


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

    def translate(
        self,
        text: str,
        source_lang: str,
        target_lang: str,
        model: str = "translategemma:12b",
        options: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Translate text using a local TranslateGemma model.

        Args:
            text: The text to translate.
            source_lang: Source language code (e.g. "en", "fr", "zh-Hans").
            target_lang: Target language code.
            model: TranslateGemma model variant (default: "translategemma:12b").
            options: Optional model parameters (temperature, top_p, etc.).

        Returns:
            Dict with 'translation', 'source_lang', 'target_lang', etc.
        """
        src_code, src_name = _resolve_language(source_lang)
        tgt_code, tgt_name = _resolve_language(target_lang)

        prompt = (
            f"You are a professional {src_name} ({src_code}) to "
            f"{tgt_name} ({tgt_code}) translator. Your goal is to "
            f"accurately convey the meaning and nuances of the original "
            f"{src_name} text while adhering to {tgt_name} grammar, "
            f"vocabulary, and cultural sensitivities.\n"
            f"Produce only the {tgt_name} translation, without any "
            f"additional explanations or commentary. Please translate "
            f"the following {src_name} text into {tgt_name}:\n"
            f"\n"
            f"\n"
            f"{text}"
        )

        payload: Dict[str, Any] = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
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
        translation = result.get("message", {}).get("content", "").strip()

        return {
            "translation": translation,
            "source_lang": src_code,
            "source_lang_name": src_name,
            "target_lang": tgt_code,
            "target_lang_name": tgt_name,
            "model": result.get("model", model),
            "done": result.get("done", True),
        }

    @staticmethod
    def translate_languages() -> List[Dict[str, str]]:
        """List languages supported by TranslateGemma.

        Returns:
            List of dicts with 'code' and 'name' keys.
        """
        return [
            {"code": code, "name": name}
            for code, name in sorted(TRANSLATEGEMMA_LANGUAGES.items())
        ]

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
