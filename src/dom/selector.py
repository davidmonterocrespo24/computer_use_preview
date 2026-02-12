"""
Element Selector - Maps element IDs to browser selectors.
"""
from typing import Dict, List, Optional, Tuple
import re


class ElementSelector:
    """Manages selection of elements using various strategies."""
    
    def __init__(self):
        self.element_map: Dict[str, Dict] = {}
    
    def register_elements(self, elements: List[Dict]):
        """Register elements from DOM extraction."""
        for elem in elements:
            self.element_map[elem['id']] = elem
    
    def get_selector(self, element_id: str) -> Optional[str]:
        """Get CSS selector for an element ID."""
        elem = self.element_map.get(element_id)
        if elem:
            return elem.get('selector')
        return None
    
    def get_xpath(self, element_id: str) -> Optional[str]:
        """Get XPath for an element ID."""
        elem = self.element_map.get(element_id)
        if elem:
            return elem.get('xpath')
        return None
    
    def get_element_info(self, element_id: str) -> Optional[Dict]:
        """Get full element information."""
        return self.element_map.get(element_id)
    
    def find_by_text(self, text: str, tag: Optional[str] = None) -> List[str]:
        """Find elements containing specific text."""
        matching_ids = []
        text_lower = text.lower()
        
        for elem_id, elem in self.element_map.items():
            elem_text = elem.get('text', '').lower()
            aria_label = elem.get('aria_label', '').lower()
            placeholder = elem.get('placeholder', '').lower()
            
            if (text_lower in elem_text or 
                text_lower in aria_label or 
                text_lower in placeholder):
                if tag is None or elem.get('tag') == tag:
                    matching_ids.append(elem_id)
        
        return matching_ids
    
    def find_by_role(self, role: str) -> List[str]:
        """Find elements with specific role."""
        return [
            elem_id for elem_id, elem in self.element_map.items()
            if elem.get('role') == role
        ]
    
    def find_input_by_type(self, input_type: str) -> List[str]:
        """Find input elements by type."""
        return [
            elem_id for elem_id, elem in self.element_map.items()
            if elem.get('tag') == 'input' and elem.get('type') == input_type
        ]
    
    def get_coordinates(self, element_id: str) -> Optional[Tuple[int, int]]:
        """Get center coordinates of an element from its bounding box."""
        elem = self.element_map.get(element_id)
        if elem and elem.get('bbox'):
            bbox = elem['bbox']
            center_x = int(bbox['x'] + bbox['width'] / 2)
            center_y = int(bbox['y'] + bbox['height'] / 2)
            return (center_x, center_y)
        return None
    
    def clear(self):
        """Clear all registered elements."""
        self.element_map.clear()
