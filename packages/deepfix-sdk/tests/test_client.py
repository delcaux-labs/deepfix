import pytest
import os
from deepfix_sdk.client import DeepFixClient

def test_deepfix_client_initialization():
    client = DeepFixClient(api_url="http://test-url", timeout=60)
    assert client.api_url == "http://test-url"
    assert client.timeout == 60

@pytest.mark.asyncio
async def test_asend_request_mocked(mocker):
    # This is a stub test to prove async structure
    client = DeepFixClient()
    mock_post = mocker.patch("httpx.AsyncClient.post")

    # Normally we would set up mock_post.return_value and verify behavior
    assert client.timeout == 30
