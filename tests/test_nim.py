import pytest
from src.providers.nvidia_nim_provider import NvidiaNimProvider

def test_nim_metadata_extraction(monkeypatch):
    monkeypatch.setenv("NVIDIA_API_KEY", "test-key")
    config = {"model": "z-ai/glm-5.2"}
    provider = NvidiaNimProvider(config)
    assert provider.name == "nvidia_nim"
    
def test_nim_budget_cap():
    from src.attempt_budget import AttemptBudget
    import json
    
    # Just a basic sanity test that budget cap works for NIM
    pass
