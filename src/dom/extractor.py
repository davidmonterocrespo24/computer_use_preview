"""
DOM Extractor - Extracts structured information from web pages.
Focuses on interactive and visible elements to minimize data sent to LLM.
"""
from typing import Dict, List, Optional, Any
from bs4 import BeautifulSoup, Tag
from dataclasses import dataclass, asdict
import json


@dataclass
class ElementInfo:
    """Information about a DOM element."""
    id: str  # Unique identifier for the element
    tag: str
    type: Optional[str] = None
    text: Optional[str] = None
    placeholder: Optional[str] = None
    value: Optional[str] = None
    aria_label: Optional[str] = None
    role: Optional[str] = None
    href: Optional[str] = None
    src: Optional[str] = None
    alt: Optional[str] = None
    title: Optional[str] = None
    classes: List[str] = None
    bbox: Optional[Dict[str, float]] = None  # Bounding box from browser
    is_visible: bool = True
    is_interactive: bool = True
    xpath: Optional[str] = None
    selector: Optional[str] = None
    
    def __post_init__(self):
        if self.classes is None:
            self.classes = []
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary, excluding None values."""
        return {k: v for k, v in asdict(self).items() if v is not None and v != []}


class DOMExtractor:
    """Extracts interactive and visible elements from HTML."""
    
    # Tags that are typically interactive
    INTERACTIVE_TAGS = {
        'a', 'button', 'input', 'textarea', 'select', 
        'option', 'label', 'summary', 'details'
    }
    
    # Roles that indicate interactivity
    INTERACTIVE_ROLES = {
        'button', 'link', 'menuitem', 'tab', 'checkbox',
        'radio', 'slider', 'spinbutton', 'textbox', 'searchbox',
        'combobox', 'listbox', 'option', 'switch'
    }
    
    def __init__(self, max_text_length: int = 200, max_elements: int = 100):
        """
        Initialize DOM extractor.
        
        Args:
            max_text_length: Maximum text length to extract from elements
            max_elements: Maximum number of elements to extract
        """
        self.max_text_length = max_text_length
        self.max_elements = max_elements
    
    def extract(self, html: str, viewport_elements: Optional[List[Dict]] = None) -> Dict[str, Any]:
        """
        Extract structured information from HTML.
        
        Args:
            html: HTML content
            viewport_elements: Optional list of elements with bounding boxes from browser
            
        Returns:
            Dictionary with page structure and interactive elements
        """
        soup = BeautifulSoup(html, 'lxml')
        
        # Extract page metadata BEFORE removing elements
        title = soup.title.string if soup.title else ""
        
        # Remove script, style, and other non-content elements
        for element in soup(['script', 'style', 'noscript', 'meta', 'link']):
            element.decompose()
        
        # Extract interactive elements
        elements = self._extract_interactive_elements(soup, viewport_elements)
        
        # Extract page structure hints
        headings = self._extract_headings(soup)
        
        # Extract forms structure
        forms = self._extract_forms(soup)
        
        return {
            "title": title,
            "url": "",  # Will be set by browser
            "elements": elements[:self.max_elements],
            "headings": headings,
            "forms": forms,
            "total_elements": len(elements),
            "truncated": len(elements) > self.max_elements
        }
    
    def _extract_interactive_elements(
        self, 
        soup: BeautifulSoup, 
        viewport_elements: Optional[List[Dict]] = None
    ) -> List[Dict[str, Any]]:
        """Extract interactive elements from the page."""
        elements = []
        element_id = 0
        
        # Create a map of viewport elements by selector if provided
        viewport_map = {}
        if viewport_elements:
            viewport_map = {elem.get('selector'): elem for elem in viewport_elements}
        
        # Find all potentially interactive elements
        for tag in soup.find_all(True):
            if self._is_interactive(tag):
                element_info = self._extract_element_info(tag, element_id, viewport_map)
                if element_info:
                    elements.append(element_info.to_dict())
                    element_id += 1
        
        return elements
    
    def _is_interactive(self, tag: Tag) -> bool:
        """Check if an element is interactive."""
        # Check tag name
        if tag.name in self.INTERACTIVE_TAGS:
            return True
        
        # Check role attribute
        role = tag.get('role', '').lower()
        if role in self.INTERACTIVE_ROLES:
            return True
        
        # Check for onclick, cursor pointer, etc.
        if tag.get('onclick') or tag.get('ng-click'):
            return True
        
        # Check if it's a clickable div/span (common in modern web apps)
        if tag.name in ['div', 'span']:
            classes = tag.get('class', [])
            if isinstance(classes, list):
                class_str = ' '.join(classes).lower()
                if any(keyword in class_str for keyword in ['button', 'link', 'clickable', 'menu', 'tab']):
                    return True
        
        return False
    
    def _extract_element_info(
        self, 
        tag: Tag, 
        element_id: int,
        viewport_map: Dict
    ) -> Optional[ElementInfo]:
        """Extract information from a single element."""
        # Get text content (truncated)
        text = tag.get_text(strip=True)
        if len(text) > self.max_text_length:
            text = text[:self.max_text_length] + "..."
        
        # Skip if no meaningful content
        if not text and tag.name not in ['input', 'img', 'button']:
            if not tag.get('aria-label') and not tag.get('title'):
                return None
        
        # Generate selector
        selector = self._generate_selector(tag)
        
        # Get bounding box if available
        bbox = viewport_map.get(selector, {}).get('bbox') if viewport_map else None
        is_visible = viewport_map.get(selector, {}).get('is_visible', True) if viewport_map else True
        
        # Skip invisible elements if we have viewport data
        if viewport_map and not is_visible:
            return None
        
        element_info = ElementInfo(
            id=f"elem_{element_id}",
            tag=tag.name,
            type=tag.get('type'),
            text=text if text else None,
            placeholder=tag.get('placeholder'),
            value=tag.get('value'),
            aria_label=tag.get('aria-label'),
            role=tag.get('role'),
            href=tag.get('href'),
            src=tag.get('src'),
            alt=tag.get('alt'),
            title=tag.get('title'),
            classes=tag.get('class', []) if isinstance(tag.get('class'), list) else [],
            bbox=bbox,
            is_visible=is_visible,
            selector=selector,
            xpath=self._generate_xpath(tag)
        )
        
        return element_info
    
    def _generate_selector(self, tag: Tag) -> str:
        """Generate a CSS selector for an element."""
        # Prefer ID
        if tag.get('id'):
            return f"#{tag.get('id')}"
        
        # Use name attribute for inputs
        if tag.name == 'input' and tag.get('name'):
            return f"input[name='{tag.get('name')}']"
        
        # Use combination of tag, class, and text
        classes = tag.get('class', [])
        if classes and isinstance(classes, list):
            class_selector = '.' + '.'.join(classes[:2])  # Use first 2 classes
            return f"{tag.name}{class_selector}"
        
        return tag.name
    
    def _generate_xpath(self, tag: Tag) -> str:
        """Generate a simple XPath for an element."""
        # This is a simplified version - in production you'd want more robust xpath
        if tag.get('id'):
            return f"//*[@id='{tag.get('id')}']"
        
        # Count position among siblings
        siblings = [s for s in tag.parent.children if s.name == tag.name] if tag.parent else []
        if len(siblings) > 1:
            position = siblings.index(tag) + 1
            return f"//{tag.name}[{position}]"
        
        return f"//{tag.name}"
    
    def _extract_headings(self, soup: BeautifulSoup) -> List[Dict[str, str]]:
        """Extract page headings for structure understanding."""
        headings = []
        for level in range(1, 7):
            for heading in soup.find_all(f'h{level}'):
                text = heading.get_text(strip=True)
                if text:
                    headings.append({
                        "level": level,
                        "text": text[:100]
                    })
        return headings
    
    def _extract_forms(self, soup: BeautifulSoup) -> List[Dict[str, Any]]:
        """Extract form structures."""
        forms = []
        for form in soup.find_all('form'):
            form_info = {
                "action": form.get('action'),
                "method": form.get('method', 'get'),
                "fields": []
            }
            
            # Extract form fields
            for input_elem in form.find_all(['input', 'textarea', 'select']):
                field = {
                    "type": input_elem.get('type', 'text'),
                    "name": input_elem.get('name'),
                    "placeholder": input_elem.get('placeholder'),
                    "required": input_elem.has_attr('required')
                }
                form_info["fields"].append(field)
            
            forms.append(form_info)
        
        return forms
    
    def to_json(self, extracted_data: Dict[str, Any]) -> str:
        """Convert extracted data to JSON string."""
        return json.dumps(extracted_data, indent=2, ensure_ascii=False)
