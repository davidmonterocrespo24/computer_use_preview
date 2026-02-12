"""
Tests for browser automation with mocked Playwright.
"""
import pytest
from unittest.mock import Mock, patch, MagicMock
from src.browser.playwright_browser import PlaywrightBrowser


class TestPlaywrightBrowser:
    """Test browser automation with mocked Playwright."""
    
    @pytest.fixture
    def mock_playwright(self):
        """Create a mock Playwright instance."""
        with patch('src.browser.playwright_browser.sync_playwright') as mock_pw:
            # Setup mock chain
            playwright = MagicMock()
            browser = MagicMock()
            context = MagicMock()
            page = MagicMock()
            
            mock_pw.return_value.__enter__ = Mock(return_value=playwright)
            mock_pw.return_value.__exit__ = Mock(return_value=None)
            
            playwright.chromium.launch.return_value = browser
            browser.new_context.return_value = context
            context.new_page.return_value = page
            
            # Setup page methods
            page.url = 'https://google.com'
            page.title.return_value = 'Google'
            page.content.return_value = '<html><body><h1>Test</h1></body></html>'
            page.screenshot.return_value = b'fake_screenshot_data'
            page.viewport_size = {'width': 1440, 'height': 900}
            page.evaluate.return_value = []
            
            yield {
                'sync_playwright': mock_pw,
                'playwright': playwright,
                'browser': browser,
                'context': context,
                'page': page
            }
    
    def test_browser_initialization(self, mock_playwright):
        """Test browser initialization."""
        browser = PlaywrightBrowser(headless=True)
        
        assert browser.headless is True
        assert browser.screen_size == (1440, 900)
    
    def test_browser_start(self, mock_playwright):
        """Test starting browser."""
        browser = PlaywrightBrowser(headless=True)
        browser.start()
        
        # Verify Playwright was started
        mock_playwright['playwright'].chromium.launch.assert_called_once()
        
        # Verify browser context was created
        mock_playwright['browser'].new_context.assert_called_once()
        
        # Verify page was created
        mock_playwright['context'].new_page.assert_called_once()
    
    def test_browser_context_manager(self, mock_playwright):
        """Test using browser as context manager."""
        with PlaywrightBrowser(headless=True) as browser:
            assert browser is not None
    
    def test_navigate(self, mock_playwright):
        """Test navigating to a URL."""
        with PlaywrightBrowser(headless=True) as browser:
            mock_playwright['page'].url = 'https://example.com'
            
            result = browser.navigate('https://example.com')
            
            # Verify goto was called
            mock_playwright['page'].goto.assert_called()
            
            # Verify result contains state
            assert 'url' in result
            assert 'dom' in result
    
    def test_navigate_adds_https(self, mock_playwright):
        """Test that navigate adds https:// if missing."""
        with PlaywrightBrowser(headless=True) as browser:
            browser.navigate('example.com')
            
            # Should add https://
            call_args = mock_playwright['page'].goto.call_args
            assert call_args[0][0] == 'https://example.com'
    
    def test_get_state(self, mock_playwright):
        """Test getting current page state."""
        with PlaywrightBrowser(headless=True) as browser:
            state = browser.get_state()
            
            assert 'url' in state
            assert 'title' in state
            assert 'dom' in state
            assert 'screenshot_base64' in state
            assert 'viewport' in state
    
    def test_click_element(self, mock_playwright):
        """Test clicking an element."""
        with PlaywrightBrowser(headless=True) as browser:
            # Register an element
            browser.element_selector.element_map['elem_0'] = {
                'id': 'elem_0',
                'selector': '#submit-btn'
            }
            
            result = browser.click('elem_0')
            
            # Verify click was called
            mock_playwright['page'].click.assert_called_with('#submit-btn', timeout=5000)
            assert 'url' in result
    
    def test_click_at_coordinates(self, mock_playwright):
        """Test clicking at specific coordinates."""
        with PlaywrightBrowser(headless=True) as browser:
            result = browser.click_at(100, 200)
            
            # Verify mouse click
            mock_playwright['page'].mouse.click.assert_called_with(100, 200)
            assert 'url' in result
    
    def test_type_text(self, mock_playwright):
        """Test typing text into an element."""
        with PlaywrightBrowser(headless=True) as browser:
            # Register an element
            browser.element_selector.element_map['elem_0'] = {
                'id': 'elem_0',
                'selector': '#search-input'
            }
            
            result = browser.type_text('elem_0', 'test query', press_enter=True)
            
            # Verify click, fill, and type were called
            mock_playwright['page'].click.assert_called()
            mock_playwright['page'].fill.assert_called_with('#search-input', '')
            mock_playwright['page'].type.assert_called_with('#search-input', 'test query')
            mock_playwright['page'].keyboard.press.assert_called_with('Enter')
    
    def test_type_text_without_clear(self, mock_playwright):
        """Test typing text without clearing first."""
        with PlaywrightBrowser(headless=True) as browser:
            browser.element_selector.element_map['elem_0'] = {
                'id': 'elem_0',
                'selector': '#input'
            }
            
            result = browser.type_text('elem_0', 'text', clear_first=False)
            
            # Verify fill was NOT called (not clearing)
            mock_playwright['page'].fill.assert_not_called()
            mock_playwright['page'].type.assert_called_with('#input', 'text')
    
    def test_scroll_down(self, mock_playwright):
        """Test scrolling down."""
        with PlaywrightBrowser(headless=True) as browser:
            result = browser.scroll('down', amount=500)
            
            # Verify JavaScript scroll was executed
            mock_playwright['page'].evaluate.assert_called()
            call_args = mock_playwright['page'].evaluate.call_args[0][0]
            assert 'scrollBy' in call_args
            assert '500' in call_args
    
    def test_scroll_up(self, mock_playwright):
        """Test scrolling up."""
        with PlaywrightBrowser(headless=True) as browser:
            result = browser.scroll('up', amount=300)
            
            call_args = mock_playwright['page'].evaluate.call_args[0][0]
            assert 'scrollBy' in call_args
            assert '-300' in call_args
    
    def test_go_back(self, mock_playwright):
        """Test going back in history."""
        with PlaywrightBrowser(headless=True) as browser:
            result = browser.go_back()
            
            mock_playwright['page'].go_back.assert_called_once()
            assert 'url' in result
    
    def test_go_forward(self, mock_playwright):
        """Test going forward in history."""
        with PlaywrightBrowser(headless=True) as browser:
            result = browser.go_forward()
            
            mock_playwright['page'].go_forward.assert_called_once()
            assert 'url' in result
    
    def test_wait(self, mock_playwright):
        """Test waiting."""
        with patch('src.browser.playwright_browser.time.sleep') as mock_sleep:
            with PlaywrightBrowser(headless=True) as browser:
                result = browser.wait(3)
                
                mock_sleep.assert_called_with(3)
                assert 'url' in result
    
    def test_hover(self, mock_playwright):
        """Test hovering over element."""
        with PlaywrightBrowser(headless=True) as browser:
            browser.element_selector.element_map['elem_0'] = {
                'id': 'elem_0',
                'selector': '#hover-btn'
            }
            
            result = browser.hover('elem_0')
            
            mock_playwright['page'].hover.assert_called_with('#hover-btn')
    
    def test_press_key(self, mock_playwright):
        """Test pressing a keyboard key."""
        with PlaywrightBrowser(headless=True) as browser:
            result = browser.press_key('Enter')
            
            mock_playwright['page'].keyboard.press.assert_called_with('Enter')
            assert 'url' in result
    
    def test_take_screenshot(self, mock_playwright):
        """Test taking a screenshot."""
        with PlaywrightBrowser(headless=True) as browser:
            screenshot = browser.take_screenshot()
            
            assert screenshot == b'fake_screenshot_data'
            mock_playwright['page'].screenshot.assert_called()
    
    def test_get_url(self, mock_playwright):
        """Test getting current URL."""
        with PlaywrightBrowser(headless=True) as browser:
            url = browser.get_url()
            
            assert url == 'https://google.com'
    
    def test_get_title(self, mock_playwright):
        """Test getting page title."""
        with PlaywrightBrowser(headless=True) as browser:
            title = browser.get_title()
            
            assert title == 'Google'
    
    def test_click_element_not_found(self, mock_playwright):
        """Test clicking non-existent element raises error."""
        with PlaywrightBrowser(headless=True) as browser:
            with pytest.raises(ValueError, match="Element .* not found"):
                browser.click('elem_999')
    
    def test_type_text_element_not_found(self, mock_playwright):
        """Test typing into non-existent element raises error."""
        with PlaywrightBrowser(headless=True) as browser:
            with pytest.raises(ValueError, match="Element .* not found"):
                browser.type_text('elem_999', 'text')
    
    def test_screen_size(self, mock_playwright):
        """Test getting screen size."""
        with PlaywrightBrowser(headless=True, screen_size=(1920, 1080)) as browser:
            size = browser.screen_size()
            
            # Should use viewport size from page
            assert size == (1440, 900)
    
    def test_screen_size_fallback(self, mock_playwright):
        """Test screen size fallback when viewport not available."""
        mock_playwright['page'].viewport_size = None
        
        with PlaywrightBrowser(headless=True, screen_size=(1920, 1080)) as browser:
            size = browser.screen_size()
            
            # Should fallback to provided size
            assert size == (1920, 1080)
