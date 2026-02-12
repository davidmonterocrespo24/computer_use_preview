"""
Tests for element selector.
"""
import pytest
from src.dom.selector import ElementSelector


class TestElementSelector:
    """Test element selector functionality."""
    
    @pytest.fixture
    def selector(self):
        """Create an element selector instance."""
        return ElementSelector()
    
    @pytest.fixture
    def sample_elements(self):
        """Sample elements for testing."""
        return [
            {
                'id': 'elem_0',
                'tag': 'input',
                'type': 'text',
                'placeholder': 'Search',
                'selector': '#search-box',
                'xpath': '//*[@id="search-box"]',
                'bbox': {'x': 10, 'y': 20, 'width': 200, 'height': 30}
            },
            {
                'id': 'elem_1',
                'tag': 'button',
                'text': 'Submit',
                'selector': 'button.submit-btn',
                'xpath': '//button[1]',
                'bbox': {'x': 50, 'y': 60, 'width': 100, 'height': 40}
            },
            {
                'id': 'elem_2',
                'tag': 'a',
                'text': 'Learn More',
                'href': '/about',
                'selector': 'a.link',
                'xpath': '//a[1]'
            },
            {
                'id': 'elem_3',
                'tag': 'input',
                'type': 'email',
                'aria_label': 'Email Address',
                'selector': 'input[type="email"]',
                'xpath': '//input[@type="email"]'
            },
            {
                'id': 'elem_4',
                'tag': 'button',
                'role': 'button',
                'text': 'Click Here',
                'selector': 'button.primary',
                'xpath': '//button[2]'
            }
        ]
    
    def test_register_elements(self, selector, sample_elements):
        """Test registering elements."""
        selector.register_elements(sample_elements)
        
        assert len(selector.element_map) == 5
        assert 'elem_0' in selector.element_map
        assert 'elem_4' in selector.element_map
    
    def test_get_selector(self, selector, sample_elements):
        """Test getting CSS selector for element."""
        selector.register_elements(sample_elements)
        
        css_selector = selector.get_selector('elem_0')
        assert css_selector == '#search-box'
        
        css_selector = selector.get_selector('elem_1')
        assert css_selector == 'button.submit-btn'
    
    def test_get_xpath(self, selector, sample_elements):
        """Test getting XPath for element."""
        selector.register_elements(sample_elements)
        
        xpath = selector.get_xpath('elem_0')
        assert xpath == '//*[@id="search-box"]'
        
        xpath = selector.get_xpath('elem_1')
        assert xpath == '//button[1]'
    
    def test_get_element_info(self, selector, sample_elements):
        """Test getting full element information."""
        selector.register_elements(sample_elements)
        
        info = selector.get_element_info('elem_0')
        assert info is not None
        assert info['tag'] == 'input'
        assert info['type'] == 'text'
        assert info['placeholder'] == 'Search'
    
    def test_find_by_text(self, selector, sample_elements):
        """Test finding elements by text content."""
        selector.register_elements(sample_elements)
        
        # Find by exact text
        results = selector.find_by_text('Submit')
        assert 'elem_1' in results
        
        # Find by partial text (case insensitive)
        results = selector.find_by_text('learn')
        assert 'elem_2' in results
        
        # Find by aria label
        results = selector.find_by_text('email')
        assert 'elem_3' in results
    
    def test_find_by_text_with_tag_filter(self, selector, sample_elements):
        """Test finding elements by text with tag filter."""
        selector.register_elements(sample_elements)
        
        # Find buttons with "Click"
        results = selector.find_by_text('Click', tag='button')
        assert 'elem_4' in results
        assert 'elem_1' not in results  # Doesn't contain "Click"
    
    def test_find_by_role(self, selector, sample_elements):
        """Test finding elements by role."""
        selector.register_elements(sample_elements)
        
        results = selector.find_by_role('button')
        assert 'elem_4' in results
    
    def test_find_input_by_type(self, selector, sample_elements):
        """Test finding input elements by type."""
        selector.register_elements(sample_elements)
        
        # Find text inputs
        results = selector.find_input_by_type('text')
        assert 'elem_0' in results
        
        # Find email inputs
        results = selector.find_input_by_type('email')
        assert 'elem_3' in results
    
    def test_get_coordinates(self, selector, sample_elements):
        """Test getting element coordinates from bounding box."""
        selector.register_elements(sample_elements)
        
        # Element with bbox
        coords = selector.get_coordinates('elem_0')
        assert coords is not None
        assert coords == (110, 35)  # Center: (10 + 200/2, 20 + 30/2)
        
        # Element with bbox
        coords = selector.get_coordinates('elem_1')
        assert coords == (100, 80)  # Center: (50 + 100/2, 60 + 40/2)
        
        # Element without bbox
        coords = selector.get_coordinates('elem_2')
        assert coords is None
    
    def test_get_nonexistent_element(self, selector, sample_elements):
        """Test getting info for non-existent element."""
        selector.register_elements(sample_elements)
        
        assert selector.get_selector('elem_999') is None
        assert selector.get_xpath('elem_999') is None
        assert selector.get_element_info('elem_999') is None
        assert selector.get_coordinates('elem_999') is None
    
    def test_clear(self, selector, sample_elements):
        """Test clearing all registered elements."""
        selector.register_elements(sample_elements)
        assert len(selector.element_map) == 5
        
        selector.clear()
        assert len(selector.element_map) == 0
    
    def test_find_by_text_empty_result(self, selector, sample_elements):
        """Test finding elements with no matches."""
        selector.register_elements(sample_elements)
        
        results = selector.find_by_text('NonexistentText')
        assert len(results) == 0
    
    def test_multiple_matches(self, selector):
        """Test finding multiple matching elements."""
        elements = [
            {'id': 'elem_0', 'tag': 'button', 'text': 'Click'},
            {'id': 'elem_1', 'tag': 'button', 'text': 'Click Here'},
            {'id': 'elem_2', 'tag': 'a', 'text': 'Click Link'}
        ]
        selector.register_elements(elements)
        
        # Should find all elements with "Click"
        results = selector.find_by_text('Click')
        assert len(results) == 3
        
        # Should find only buttons with "Click"
        results = selector.find_by_text('Click', tag='button')
        assert len(results) == 2
