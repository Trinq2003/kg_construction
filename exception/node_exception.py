class NodeNotFoundError(Exception):
    def __init__(self, node_id: str) -> None:
        self.node_id = node_id
        self.message = f"Node with ID: '{node_id}' does not exist in the graph."
        super().__init__(self.message)