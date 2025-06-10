"""Request Logging Utility"""

import logging
from datetime import datetime
from typing import Optional
from ..models.schemas import QueryType

logger = logging.getLogger(__name__)


async def log_request(
    client_ip: str,
    question: str,
    query_type: Optional[QueryType] = None,
    processing_time: Optional[float] = None,
    success: bool = True,
    error_message: Optional[str] = None
):
    """Log request for analytics and monitoring"""

    truncated_question = question[:100] + "..." if len(question) > 100 else question

    log_data = {
        "timestamp": datetime.now().isoformat(),
        "client_ip": client_ip,
        "question_preview": truncated_question,
        "query_type": query_type.value if query_type else "unknown",
        "processing_time_seconds": processing_time,
        "success": success,
        "error_message": error_message
    }

    if success:
        logger.info(f"Request completed: {log_data}")
    else:
        logger.error(f"Request failed: {log_data}")
