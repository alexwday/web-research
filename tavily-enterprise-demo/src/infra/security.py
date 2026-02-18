"""Optional corporate SSL certificate setup."""
import logging
from typing import Optional

logger = logging.getLogger(__name__)

try:
    import rbc_security

    _RBC_SECURITY_AVAILABLE = True
except ImportError:
    _RBC_SECURITY_AVAILABLE = False


def configure_rbc_security_certs() -> Optional[str]:
    """Enable corporate certs when the optional package is available."""
    if not _RBC_SECURITY_AVAILABLE:
        logger.info("rbc_security package not available")
        return None

    rbc_security.enable_certs()
    logger.info("rbc_security certificates enabled")
    return "rbc_security"
