import unittest
from src.core.ontology import ConceptNode

class TestOntology(unittest.TestCase):
    def test_concept_node_creation(self):
        node = ConceptNode("Animal", attributes=["Alive"])
        self.assertEqual(node.label, "Animal")
        self.assertEqual(node.attributes, ["Alive"])
        self.assertIsNone(node.parent)
        self.assertEqual(node.children, [])

    def test_concept_hierarchy(self):
        parent = ConceptNode("Animal")
        child = ConceptNode("Dog", parent=parent)

        self.assertEqual(child.parent, parent)
        self.assertIn(child, parent.children)
        self.assertEqual(len(parent.children), 1)

    def test_to_dict(self):
        parent = ConceptNode("Animal")
        child = ConceptNode("Dog", parent=parent)

        d = parent.to_dict()
        self.assertEqual(d['label'], "Animal")
        self.assertEqual(len(d['children']), 1)
        self.assertEqual(d['children'][0]['label'], "Dog")
        self.assertEqual(d['children'][0]['parent'], "Animal")

if __name__ == '__main__':
    unittest.main()
