from abc import ABC, abstractmethod

from schemas.profile_schema import PlatformSnapshot


class PlatformMonitor(ABC):
    @abstractmethod
    def sample(self) -> PlatformSnapshot:
        ...
