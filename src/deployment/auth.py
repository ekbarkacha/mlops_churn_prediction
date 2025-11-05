"""
Authentication & Rate Limiting
Compatible: Python 3.11.6 + FastAPI 0.109.0 + SlowAPI 0.1.9
"""
from fastapi import HTTPException, Security
from fastapi.security import APIKeyHeader
from starlette.status import HTTP_403_FORBIDDEN
from slowapi import Limiter
from slowapi.util import get_remote_address
import logging

from .config import config

# Logger
logger = logging.getLogger(__name__)


# ========== API KEY AUTHENTICATION ==========

# Define API Key header
api_key_header = APIKeyHeader(
    name="X-API-Key",
    auto_error=False,
    description="API Key for authentication"
)


async def verify_api_key(api_key: str = Security(api_key_header)) -> str:
    """
    Verify API key from request header
    
    Args:
        api_key: API key from X-API-Key header
    
    Returns:
        Validated API key string
    
    Raises:
        HTTPException: 403 if key is invalid or missing
    
    Example:
        @app.get("/protected")
        async def protected_route(api_key: str = Depends(verify_api_key)):
            return {"message": "Access granted"}
    """
    
    # Skip authentication if disabled (dev mode)
    if not config.API_KEY_ENABLED:
        logger.debug("Authentication disabled (dev mode)")
        return "dev_mode"
    
    # Check if API key is provided
    if api_key is None:
        logger.warning("Missing API key in request")
        raise HTTPException(
            status_code=HTTP_403_FORBIDDEN,
            detail={
                "error": "Missing API Key",
                "message": "Please include 'X-API-Key' header in your request"
            }
        )
    
    # Validate API key
    if api_key not in config.API_KEYS:
        logger.warning(f"Invalid API key attempted: {api_key[:8]}***")
        raise HTTPException(
            status_code=HTTP_403_FORBIDDEN,
            detail={
                "error": "Invalid API Key",
                "message": "The provided API key is not valid"
            }
        )
    
    logger.debug(f"API key validated successfully: {api_key[:8]}***")
    return api_key


# ========== RATE LIMITING ==========

# Initialize rate limiter
limiter = Limiter(
    key_func=get_remote_address,
    default_limits=[] if not config.RATE_LIMIT_ENABLED else [
        f"{config.RATE_LIMIT_PER_MINUTE}/minute",
        f"{config.RATE_LIMIT_PER_HOUR}/hour"
    ],
    storage_uri="memory://"
)


# Export
__all__ = ["verify_api_key", "limiter", "api_key_header"]