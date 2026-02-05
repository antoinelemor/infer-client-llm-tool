"""Tests for InferClient (integration — requires running API)."""

import os

import pytest

from infer_client import InferClient

BASE_URL = os.environ.get("INFER_URL", "https://your-server.example.com")
API_KEY = os.environ.get("INFER_KEY", "")


@pytest.fixture
def client():
    if not API_KEY:
        pytest.skip("Set INFER_KEY env var to run integration tests")
    return InferClient(base_url=BASE_URL, api_key=API_KEY)


# ── Health ──

def test_health(client):
    r = client.health()
    assert r["status"] == "ok"
    assert "version" in r
    assert "models_count" in r
    assert r["models_count"] >= 1
    assert "default_model" in r
    assert isinstance(r["models"], list)


# ── Models listing ──

def test_models(client):
    models = client.models()
    assert isinstance(models, list)
    assert len(models) >= 1
    m = models[0]
    assert "model_id" in m
    assert "model_type" in m
    assert "num_labels" in m
    assert "labels" in m


# ── Model info ──

def test_model_info(client):
    info = client.model_info("sentiment")
    assert info["model_id"] == "sentiment"
    assert "base_model" in info
    assert "labels" in info
    assert "metrics" in info
    assert "hyperparameters" in info
    assert "languages" in info


def test_model_info_not_found(client):
    with pytest.raises(Exception):
        client.model_info("nonexistent_model_xyz")


# ── Inference (default model) ──

def test_infer_single(client):
    r = client.infer(text="The economy is booming")
    assert r["count"] == 1
    assert "label" in r["results"][0]
    assert "confidence" in r["results"][0]
    assert "model_id" in r


def test_infer_batch(client):
    texts = ["Good news", "Bad news", "No opinion"]
    r = client.infer(texts=texts)
    assert r["count"] == 3
    for res in r["results"]:
        assert "label" in res
        assert "probabilities" in res


# ── Inference (explicit model) ──

def test_infer_with_model(client):
    r = client.infer(text="Markets are recovering", model="sentiment")
    assert r["count"] == 1
    assert r["model_id"] == "sentiment"
    assert "label" in r["results"][0]


# ── Classify shortcut ──

def test_classify_shortcut(client):
    results = client.classify("Terrible quarter for the company")
    assert len(results) == 1
    assert results[0]["label"].startswith("sentiment_long_")


def test_classify_with_model(client):
    results = client.classify("Great performance", model="sentiment")
    assert len(results) == 1
    assert "label" in results[0]


# ── Input validation ──

def test_no_input_raises():
    c = InferClient(base_url="http://localhost", api_key="x")
    with pytest.raises(ValueError, match="Provide"):
        c.infer()
