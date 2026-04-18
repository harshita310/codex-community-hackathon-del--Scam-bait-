from abc import ABC, abstractmethod

class BaseAgent(ABC):
    """Base interface for all ScamBait AI agents."""
    
    @abstractmethod
    def process(self, text: str, state: dict) -> dict:
        pass
