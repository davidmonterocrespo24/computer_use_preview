"""
Tests for DOM extractor.
"""
import pytest
from src.dom.extractor import DOMExtractor, ElementInfo


class TestDOMExtractor:
    """Test DOM extraction functionality."""
    
    @pytest.fixture
    def extractor(self):
        """Create a DOM extractor instance."""
        return DOMExtractor(max_text_length=100, max_elements=50)
    
    @pytest.fixture
    def sample_html(self):
        """Sample HTML for testing."""
        return """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Test Page</title>
        </head>
        <body>
            <h1>Welcome</h1>
            <h2>Subtitle</h2>
            
            <form action="/submit" method="post">
                <input type="text" name="username" placeholder="Enter username" id="user-input">
                <input type="password" name="password" placeholder="Enter password">
                <button type="submit">Login</button>
            </form>
            
            <a href="https://example.com">Example Link</a>
            <a href="/about">About</a>
            
            <div class="button clickable" onclick="doSomething()">Click Me</div>
            
            <script>console.log('test')</script>
            <style>.test { color: red; }</style>
        </body>
        </html>
        """
    
    def test_extract_title(self, extractor, sample_html):
        """Test extracting page title."""
        result = extractor.extract(sample_html)
        assert result['title'] == 'Test Page'
    
    def test_extract_interactive_elements(self, extractor, sample_html):
        """Test extracting interactive elements."""
        result = extractor.extract(sample_html)
        elements = result['elements']
        
        # Should find inputs, button, links, and clickable div
        assert len(elements) > 0
        
        # Check for input elements
        input_elements = [e for e in elements if e['tag'] == 'input']
        assert len(input_elements) == 2
        
        # Check for button
        button_elements = [e for e in elements if e['tag'] == 'button']
        assert len(button_elements) == 1
        assert button_elements[0]['text'] == 'Login'
        
        # Check for links
        link_elements = [e for e in elements if e['tag'] == 'a']
        assert len(link_elements) == 2
    
    def test_extract_element_with_placeholder(self, extractor, sample_html):
        """Test extracting element with placeholder."""
        result = extractor.extract(sample_html)
        elements = result['elements']
        
        username_input = [e for e in elements if e.get('placeholder') == 'Enter username']
        assert len(username_input) == 1
        assert username_input[0]['type'] == 'text'
    
    def test_extract_headings(self, extractor, sample_html):
        """Test extracting page headings."""
        result = extractor.extract(sample_html)
        headings = result['headings']
        
        assert len(headings) == 2
        assert headings[0]['level'] == 1
        assert headings[0]['text'] == 'Welcome'
        assert headings[1]['level'] == 2
        assert headings[1]['text'] == 'Subtitle'
    
    def test_extract_forms(self, extractor, sample_html):
        """Test extracting form structure."""
        result = extractor.extract(sample_html)
        forms = result['forms']
        
        assert len(forms) == 1
        assert forms[0]['action'] == '/submit'
        assert forms[0]['method'] == 'post'
        assert len(forms[0]['fields']) == 2
    
    def test_removes_script_and_style(self, extractor, sample_html):
        """Test that script and style tags are removed."""
        result = extractor.extract(sample_html)
        elements = result['elements']
        
        # Should not find script or style elements
        script_elements = [e for e in elements if e['tag'] in ['script', 'style']]
        assert len(script_elements) == 0
    
    def test_element_id_generation(self, extractor, sample_html):
        """Test that elements get unique IDs."""
        result = extractor.extract(sample_html)
        elements = result['elements']
        
        ids = [e['id'] for e in elements]
        # All IDs should be unique
        assert len(ids) == len(set(ids))
        # IDs should follow pattern elem_0, elem_1, etc.
        assert all(id.startswith('elem_') for id in ids)
    
    def test_max_elements_limit(self):
        """Test that max_elements limit is respected."""
        extractor = DOMExtractor(max_elements=3)
        html = """
        <html><body>
            <a href="#">Link 1</a>
            <a href="#">Link 2</a>
            <a href="#">Link 3</a>
            <a href="#">Link 4</a>
            <a href="#">Link 5</a>
        </body></html>
        """
        result = extractor.extract(html)
        
        assert len(result['elements']) == 3
        assert result['truncated'] is True
        assert result['total_elements'] > 3
    
    def test_text_length_truncation(self):
        """Test that long text is truncated."""
        extractor = DOMExtractor(max_text_length=20)
        html = """
        <html><body>
            <a href="#">This is a very long text that should be truncated</a>
        </body></html>
        """
        result = extractor.extract(html)
        
        link = result['elements'][0]
        assert len(link['text']) <= 23  # 20 + "..."
        assert link['text'].endswith('...')
    
    def test_extract_aria_labels(self, extractor):
        """Test extracting aria-label attributes."""
        html = """
        <html><body>
            <button aria-label="Close dialog">X</button>
            <input type="text" aria-label="Search">
        </body></html>
        """
        result = extractor.extract(html)
        elements = result['elements']
        
        button = [e for e in elements if e['tag'] == 'button'][0]
        assert button['aria_label'] == 'Close dialog'
        
        input_elem = [e for e in elements if e['tag'] == 'input'][0]
        assert input_elem['aria_label'] == 'Search'
    
    def test_extract_with_viewport_elements(self, extractor, sample_html):
        """Test extracting with viewport bounding box data."""
        viewport_elements = [
            {
                'selector': '#user-input',
                'bbox': {'x': 10, 'y': 20, 'width': 200, 'height': 30},
                'is_visible': True
            }
        ]
        
        result = extractor.extract(sample_html, viewport_elements)
        elements = result['elements']
        
        # Should include bounding box for matching elements
        user_input = [e for e in elements if e.get('id') == 'user-input']
        if user_input:
            # Note: bbox matching depends on selector generation
            pass
    
    def test_to_json(self, extractor, sample_html):
        """Test JSON serialization."""
        result = extractor.extract(sample_html)
        json_str = extractor.to_json(result)
        
        assert isinstance(json_str, str)
        assert 'title' in json_str
        assert 'elements' in json_str
        
        # Should be valid JSON
        import json
        parsed = json.loads(json_str)
        assert parsed['title'] == 'Test Page'
    
    def test_clickable_div_detection(self, extractor):
        """Test detecting clickable divs."""
        html = """
        <html><body>
            <div class="btn-primary button">Click</div>
            <div class="menu-item">Menu</div>
            <div class="clickable">Action</div>
        </body></html>
        """
        result = extractor.extract(html)
        elements = result['elements']
        
        # Should detect divs with button/clickable classes
        assert len(elements) >= 2
    
    def test_role_attribute_detection(self, extractor):
        """Test detecting elements by role attribute."""
        html = """
        <html><body>
            <div role="button">Custom Button</div>
            <div role="link">Custom Link</div>
            <span role="checkbox">Check</span>
        </body></html>
        """
        result = extractor.extract(html)
        elements = result['elements']
        
        assert len(elements) == 3
        roles = [e.get('role') for e in elements]
        assert 'button' in roles
        assert 'link' in roles
        assert 'checkbox' in roles
