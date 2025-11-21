from typing import List, Optional, Dict, Any

class ConceptNode:
    """
    Represents a concept in the ontology for drawing analysis.
    """
    def __init__(
        self,
        label: str,
        parent: Optional['ConceptNode'] = None,
        attributes: Optional[List[str]] = None
    ):
        self.label = label
        self.parent = parent
        self.attributes = attributes if attributes is not None else []
        self.children: List['ConceptNode'] = []

        if parent:
            parent.add_child(self)

    def add_child(self, child: 'ConceptNode'):
        """Adds a child node to this concept."""
        self.children.append(child)
        child.parent = self

    def to_dict(self) -> Dict[str, Any]:
        """Returns a dictionary representation of the concept node."""
        return {
            "label": self.label,
            "parent": self.parent.label if self.parent else None,
            "attributes": self.attributes,
            "children": [child.to_dict() for child in self.children]
        }

    def __repr__(self):
        return f"ConceptNode(label='{self.label}', attributes={self.attributes})"
