import os
from unittest.mock import patch, MagicMock
import importlib
import sys

# Add the src directory to sys.path
sys.path.append("packages/deepfix-portal/src")

# Pre-mocking all dependencies that main.py imports
mock_fastapi = MagicMock()
mock_cors = MagicMock()
mock_core = MagicMock()
mock_server = MagicMock()
mock_db = MagicMock()
mock_routes = MagicMock()

sys.modules['fastapi'] = mock_fastapi
sys.modules['fastapi.middleware.cors'] = mock_cors
sys.modules['deepfix_core'] = mock_core
sys.modules['deepfix_core.models'] = mock_core
sys.modules['deepfix_server'] = mock_server
sys.modules['deepfix_server.logging'] = mock_server
sys.modules['deepfix_portal.api.database'] = mock_db
sys.modules['deepfix_portal.api.routes'] = mock_routes

def test_cors_origins_logic():
    # Helper to reload main and get its origins
    def get_main_origins(env_vars):
        with patch.dict(os.environ, env_vars, clear=True):
            import deepfix_portal.api.main
            importlib.reload(deepfix_portal.api.main)
            from deepfix_portal.api.main import allowed_origins
            return allowed_origins

    # Test Case 1: No env vars
    origins = get_main_origins({})
    assert origins == ["http://localhost:5173", "http://localhost:8844"]
    print("Test Case 1 passed: No env vars")

    # Test Case 2: FRONTEND_URL only
    origins = get_main_origins({"FRONTEND_URL": "https://myfrontend.com"})
    assert origins == ["https://myfrontend.com"]
    print("Test Case 2 passed: FRONTEND_URL only")

    # Test Case 3: CORS_ALLOWED_ORIGINS only
    origins = get_main_origins({"CORS_ALLOWED_ORIGINS": "https://origin1.com, https://origin2.com"})
    assert origins == ["https://origin1.com", "https://origin2.com"]
    print("Test Case 3 passed: CORS_ALLOWED_ORIGINS only")

    # Test Case 4: Both
    origins = get_main_origins({
        "FRONTEND_URL": "https://myfrontend.com",
        "CORS_ALLOWED_ORIGINS": "https://origin1.com"
    })
    assert set(origins) == {"https://myfrontend.com", "https://origin1.com"}
    print("Test Case 4 passed: Both")

if __name__ == "__main__":
    try:
        test_cors_origins_logic()
        print("\nCORS logic tests passed!")
    except Exception as e:
        print(f"CORS logic tests failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
