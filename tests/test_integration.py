"""
Integration tests - End to end testing (still with mocked LLM).
"""
import pytest
from unittest.mock import Mock, patch
from src.agent.browser_agent import BrowserAgent
from src.llm.client import LLMResponse


class TestIntegration:
    """Integration tests combining multiple components."""
    
    @patch('src.browser.playwright_browser.sync_playwright')
    def test_full_search_workflow(self, mock_playwright):
        """Test a complete search workflow with mocked browser."""
        # Setup mock Playwright
        playwright = Mock()
        browser = Mock()
        context = Mock()
        page = Mock()
        
        mock_playwright.return_value.__enter__ = Mock(return_value=playwright)
        mock_playwright.return_value.__exit__ = Mock(return_value=None)
        
        playwright.chromium.launch.return_value = browser
        browser.new_context.return_value = context
        context.new_page.return_value = page
        
        # Setup page responses
        page.url = 'https://google.com'
        page.title.return_value = 'Google'
        page.content.return_value = '''
            <html><body>
                <input type="text" name="q" placeholder="Search">
                <button type="submit">Google Search</button>
            </body></html>
        '''
        page.screenshot.return_value = b'screenshot'
        page.viewport_size = {'width': 1440, 'height': 900}
        page.evaluate.return_value = []
        
        # Create mock LLM
        mock_llm = Mock()
        responses = [
            LLMResponse(
                content="I'll type in the search box",
                actions=[{
                    'name': 'type_text',
                    'arguments': {
                        'element_id': 'elem_0',
                        'text': 'Python tutorials',
                        'press_enter': True
                    }
                }]
            ),
            LLMResponse(
                content="Search completed",
                actions=[{
                    'name': 'task_complete',
                    'arguments': {
                        'result': 'Search results displayed',
                        'success': True
                    }
                }]
            )
        ]
        mock_llm.generate_with_retry.side_effect = responses
        
        # Run integration test
        from src.browser.playwright_browser import PlaywrightBrowser
        
        with PlaywrightBrowser(headless=True) as browser_instance:
            agent = BrowserAgent(browser_instance, mock_llm, max_iterations=5, verbose=False)
            result = agent.run("Search for Python tutorials")
            
            assert result['success'] is True
            assert 'Search results' in result['result']
    
    def test_dom_extraction_with_selector(self):
        """Test DOM extraction integrated with selector."""
        from src.dom.extractor import DOMExtractor
        from src.dom.selector import ElementSelector
        
        html = '''
            <html>
            <body>
                <button id="login-btn">Login</button>
                <input type="text" name="username" placeholder="Username">
                <a href="/help">Help</a>
            </body>
            </html>
        '''
        
        # Extract DOM
        extractor = DOMExtractor()
        dom_data = extractor.extract(html)
        
        # Register with selector
        selector = ElementSelector()
        selector.register_elements(dom_data['elements'])
        
        # Find button by text
        button_ids = selector.find_by_text('Login')
        assert len(button_ids) > 0
        
        # Get selector for button
        css_selector = selector.get_selector(button_ids[0])
        assert css_selector is not None
        
        # Find input by type
        input_ids = selector.find_input_by_type('text')
        assert len(input_ids) > 0
    
    def test_llm_response_parsing_integration(self):
        """Test LLM response parsing with different formats."""
        from src.llm.client import LLMResponse
        
        # Test JSON array format
        response1 = LLMResponse(
            content='''
            I'll navigate to the page:
            ```json
            [
                {"name": "navigate", "arguments": {"url": "https://example.com"}},
                {"name": "wait", "arguments": {"seconds": 2}}
            ]
            ```
            ''',
            actions=None
        )
        
        actions1 = response1.parse_actions()
        assert len(actions1) == 2
        assert actions1[0]['name'] == 'navigate'
        assert actions1[1]['name'] == 'wait'
        
        # Test object format
        response2 = LLMResponse(
            content='''
            ```json
            {
                "actions": [
                    {"name": "click", "arguments": {"element_id": "elem_0"}}
                ]
            }
            ```
            ''',
            actions=None
        )
        
        actions2 = response2.parse_actions()
        assert len(actions2) == 1
        assert actions2[0]['name'] == 'click'
        
        # Test direct actions (function calling)
        response3 = LLMResponse(
            content="Clicking button",
            actions=[
                {'name': 'click', 'arguments': {'element_id': 'elem_1'}}
            ]
        )
        
        actions3 = response3.parse_actions()
        assert len(actions3) == 1
        assert actions3[0]['name'] == 'click'
