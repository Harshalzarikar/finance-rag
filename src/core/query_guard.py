import re
from dataclasses import dataclass

@dataclass
class QueryGuardResult:
    allowed: bool
    reason: str | None = None

class QueryGuard:
    """Lightweight regex-based query safety guardrail."""
    
    # Patterns that suggest prompt injection, jailbreaking, or ignoring instructions
    _SUSPICIOUS_PATTERNS = [
        re.compile(r"ignore\s+(?:all\s+)?(?:previous\s+)?(?:instructions|directions|prompts)", re.IGNORECASE),
        re.compile(r"disregard\s+(?:all\s+)?(?:previous\s+)?(?:instructions|directions|prompts)", re.IGNORECASE),
        re.compile(r"system\s+prompt", re.IGNORECASE),
        re.compile(r"you\s+are\s+now", re.IGNORECASE),
        re.compile(r"forget\s+(?:all\s+)?(?:previous\s+)?(?:instructions|directions|prompts)", re.IGNORECASE),
        re.compile(r"bypass\s+filters?", re.IGNORECASE),
    ]

    def check(self, query: str) -> QueryGuardResult:
        """
        Evaluate if a query is safe to process.
        Returns a QueryGuardResult indicating if it's allowed.
        """
        for pattern in self._SUSPICIOUS_PATTERNS:
            if pattern.search(query):
                return QueryGuardResult(
                    allowed=False,
                    reason="Query contains potentially adversarial instructions or prompt injection attempts."
                )
                
        return QueryGuardResult(allowed=True)
