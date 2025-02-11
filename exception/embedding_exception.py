class EmbeddingModelNotFoundError(Exception):
    def __init__(self) -> None:
        self.message = "Embedding model not found. You should load an embedding model before using it."
        super().__init__(self.message)