class APIKeyIsNotSetError(Exception):
    def __init__(self) -> None:
        self.message = "LLM API key or base URL is not set."
        super().__init__(self.message)

class APIBaseIsNotSetError(Exception):
    def __init__(self) -> None:
        self.message = "LLM API key or base URL is not set."
        super().__init__(self.message)