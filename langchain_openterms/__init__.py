"""LangChain integration for OpenTerms protocol.

Provides permission-aware web interaction for LangChain agents by checking
openterms.json before allowing actions on a domain.

Three integration patterns:

1. OpenTermsGuard — wraps any tool so it checks permissions before executing
2. OpenTermsChecker — standalone tool an agent can call to check permissions
3. OpenTermsCallbackHandler — callback that logs permission checks on tool use
"""

from langchain_openterms.guard import OpenTermsGuard
from langchain_openterms.checker import OpenTermsChecker
from langchain_openterms.callback import OpenTermsCallbackHandler
from langchain_openterms.client import OpenTermsClient

__all__ = [
    "OpenTermsGuard",
    "OpenTermsChecker",
    "OpenTermsCallbackHandler",
    "OpenTermsClient",
]

__version__ = "0.3.1"
