"""RBC Security certificate setup. Uses optional rbc_security for SSL when available."""
import logging
from typing import Optional

try:
    import rbc_security
    _RBC_SECURITY_AVAILABLE = True
except ImportError:
    _RBC_SECURITY_AVAILABLE = False

logger = logging.getLogger(__name__)


def configure_rbc_security_certs() -> Optional[str]:
    """Enable RBC SSL certificates if the rbc_security package is available."""
    if not _RBC_SECURITY_AVAILABLE:
        logger.info("rbc_security not available, continuing without SSL certificates")
        return None
    logger.info("Enabling RBC Security certificates...")
    rbc_security.enable_certs()
    logger.info("RBC Security certificates enabled")
    return "rbc_security"
