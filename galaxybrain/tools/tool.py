from abc import ABC, abstractmethod


class Tool(ABC):
    @abstractmethod
    def description(self) -> str:
        pass

    @abstractmethod
    def examples(self) -> str:
        pass

    @abstractmethod
    def run(self, value: str) -> str:
        pass
