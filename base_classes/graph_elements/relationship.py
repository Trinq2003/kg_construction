from datetime import datetime
from py2neo import Relationship
from knowledge_graph.modules.node import *
from typing import Any, Dict, Type, get_origin, get_args, List, Tuple, Set, Union

class GraphRelationship(Relationship):
    def __init__(self, start_node: GraphNode, end_node: GraphNode, rel_type, defaults=None, **properties):
        defaults = defaults or {}
        combined_properties = self._apply_defaults_and_validate(defaults, properties)
        combined_properties.update({
            'created_at': datetime.now(),
            'updated_at': datetime.now(),
        })
        super().__init__(start_node, rel_type, end_node, **combined_properties)

    @staticmethod
    def _apply_defaults_and_validate(defaults: Dict[str, Any], properties: Dict[str, Any]) -> Dict[str, Any]:
        for key, expected_type in defaults.items():
            if key not in properties:
                raise ValueError(f"Missing required property: '{key}'")
            if not GraphNode._is_instance_of_type(properties[key], expected_type):
                raise TypeError(f"Property '{key}' must be of type {expected_type}")
        return {**properties}

    @staticmethod
    def _is_instance_of_type(value: Any, expected_type: Type) -> bool:
        """Helper method to check if a value is an instance of an expected type."""
        origin = get_origin(expected_type)
        args = get_args(expected_type)

        if origin is None:
            return isinstance(value, expected_type)
        if origin in {list, List}:
            if not isinstance(value, list):
                return False
            if args:
                return all(GraphNode._is_instance_of_type(item, args[0]) for item in value)
            return True
        if origin in {dict, Dict}:
            if not isinstance(value, dict):
                return False
            if args and len(args) == 2:
                key_type, value_type = args
                return all(
                    GraphNode._is_instance_of_type(k, key_type) and GraphNode._is_instance_of_type(v, value_type)
                    for k, v in value.items()
                )
            return True
        if origin in {tuple, Tuple}:
            if not isinstance(value, tuple):
                return False
            if len(args) == len(value):
                return all(GraphNode._is_instance_of_type(v, t) for v, t in zip(value, args))
            if len(args) == 2 and args[1] is Ellipsis:
                return all(GraphNode._is_instance_of_type(v, args[0]) for v in value)
            return False
        if origin in {set, Set}:
            if not isinstance(value, set):
                return False
            if args:
                return all(GraphNode._is_instance_of_type(item, args[0]) for item in value)
            return True
        if origin is Union:
            return any(GraphNode._is_instance_of_type(value, arg) for arg in args)
        if origin is None:
            return isinstance(value, expected_type)

        return False