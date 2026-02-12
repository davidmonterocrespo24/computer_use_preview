"""
Tests for browser automation with mocked Playwright.
"""
import pytest
from unittest.mock import Mock, patch, MagicMock, PropertyMock, call
from src.browser.playwright_browser import PlaywrightBrowser


class TestPlaywrightBrowser:
    """Test browser automation with mocked Playwright."""
    
    @pytest.fixture(autouse=True)
    def setup_mocks(self):
        """Setup Playwright mocks for all tests."""
        # Create mock objects
        self.mock_page = MagicMock()
        self.mock_context = MagicMock()
        self.mock_browser = MagicMock()
        self.mock_playwright = MagicMock()
        
        # Configure page mock - CRITICAL: content() must return a string
        self.mock_page.url = 'https://google.com'
        self.mock_page.title.return_value = 'Google'
        self.mock_page.content.return_value = '<html><head><title>Google</title></head><body><h1>Test</h1><input id="search" type="text" placeholder="Search"/><button>Search</button></body></html>'
        self.mock_page.screenshot.return_value = b'fake_screenshot_data'
        self.mock_page.viewport_size = {'width': 1440, 'height': 900}
        self.mock_page.evaluate.return_value = []
        self.mock_page.goto.return_value = None
        self.mock_page.click.return_value = None
        self.mock_page.fill.return_value = None
        self.mock_page.press.return_value = None
        self.mock_page.press.return_value = None
        self.mock_page.wait_for_timeout.return_value = None
        self.mock_page.hover.return_value = None
        self.mock_page.go_back.return_value = None
        self.mock_page.go_forward.return_value = None
        self.mock_page.wait_for_load_state.return_value = None
        self.mock_page.locator.return_value = MagicMock()
        self.mock_page.close.return_value = None
        
        # Configure browser chain
        self.mock_context.new_page.return_value = self.mock_page
        self.mock_context.on = Mock()  # For event handlers
        self.mock_context.close.return_value = None
        self.mock_browser.new_context.return_value = self.mock_context
        self.mock_browser.close.return_value = None
        self.mock_playwright.chromium.launch.return_value = self.mock_browser
        
        # Create the playwright() object that has a .start() method
        # CRITICAL: Code calls sync_playwright().start(), NOT the context manager
        self.mock_pw_obj = MagicMock()
        self.mock_pw_obj.start.return_value = self.mock_playwright
        self.mock_pw_obj.stop.return_value = None
        
        # Patch sync_playwright to return the object with .start()
        self.patcher = patch('src.browser.playwright_browser.sync_playwright', return_value=self.mock_pw_obj)
        self.patcher.start()
        
        yield
        
        # Cleanup
        self.patcher.stop()
    
    def test_browser_initialization(self):
        """Test browser initialization."""
        browser = PlaywrightBrowser(headless=True)
        
        assert browser.headless is True
        assert browser.screen_size == (1440, 900)
    
    def test_browser_start(self):
        """Test starting browser."""
        browser = PlaywrightBrowser(headless=True)
        browser.start()
        
        # Verify Playwright was started
        self.mock_playwright.chromium.launch.assert_called_once()
        
        # Verify browser context was created
        self.mock_browser.new_context.assert_called_once()
        
        # Verify page was created
        self.mock_context.new_page.assert_called_once()
    
    def test_browser_context_manager(self):
        """Test using browser as context manager."""
        with PlaywrightBrowser(headless=True) as browser:
            assert browser is not None
    
    def test_navigate(self):
        """Test navigating to a URL."""
        with PlaywrightBrowser(headless=True) as browser:
            self.mock_page.url = 'https://example.com'
            
            result = browser.navigate('https://example.com')
            
            # Verify goto was called
            self.mock_page.goto.assert_called()
            
            # Verify result contains state
            assert 'url' in result
            assert 'dom' in result
    
    def test_navigate_adds_https(self):
        """Test that navigate adds https:// if missing."""
        with PlaywrightBrowser(headless=True) as browser:
            browser.navigate('example.com')
            
            # Should add https://
            call_args = self.mock_page.goto.call_args
            assert call_args[0][0] == 'https://example.com'
    
    def test_get_state(self):
        """Test getting current page state."""
        with PlaywrightBrowser(headless=True) as browser:
            state = browser.get_state()
            
            assert 'url' in state
            assert 'title' in state
            assert 'dom' in state
            assert 'screenshot_base64' in state
            assert 'viewport' in state
    
    def test_click_element(self):
        """Test clicking an element."""
        with PlaywrightBrowser(headless=True) as browser:
            # Register an element
            browser.element_selector.element_map['elem_0'] = {
                'id': 'elem_0',
                'selector': '#submit-btn'
            }
            
            result = browser.click('elem_0')
            
            # Verify click was called
            self.mock_page.click.assert_called_with('#submit-btn', timeout=5000)
            assert 'url' in result
    
    def test_click_at_coordinates(self):
        """Test clicking at specific coordinates."""
        with PlaywrightBrowser(headless=True) as browser:
            result = browser.click_at(100, 200)
            
            # Verify mouse click
            self.mock_page.mouse.click.assert_called_with(100, 200)
            assert 'url' in result
    
    def test_type_text(self):
        """Test typing text into an element."""
        with PlaywrightBrowser(headless=True) as browser:
            # Register an element
            browser.element_selector.element_map['elem_0'] = {
                'id': 'elem_0',
                'selector': '#search-input'
            }
            
            result = browser.type_text('elem_0', 'test query', press_enter=True)
            
            # Verify click, fill, and type were called
            self.mock_page.click.assert_called()
            self.mock_page.fill.assert_called_with('#search-input', '')
            self.mock_page.type.assert_called_with('#search-input', 'test query')
            self.mock_page.keyboard.press.assert_called_with('Enter')
    
    def test_type_text_without_clear(self):
        """Test typing text without clearing first."""
        with PlaywrightBrowser(headless=True) as browser:
            browser.element_selector.element_map['elem_0'] = {
                'id': 'elem_0',
                'selector': '#input'
            }
            
            result = browser.type_text('elem_0', 'text', clear_first=False)
            
            # Verify fill was NOT called (not clearing)
            self.mock_page.fill.assert_not_called()
            self.mock_page.type.assert_called_with('#input', 'text')
    
    def test_scroll_down(self):
        """Test scrolling down."""
        with PlaywrightBrowser(headless=True) as browser:
            result = browser.scroll('down', amount=500)
            
            # Verify JavaScript scroll was executed
            self.mock_page.evaluate.assert_called()
            call_args = self.mock_page.evaluate.call_args[0][0]
            assert 'scrollBy' in call_args
            assert '500' in call_args
    
    def test_scroll_up(self):
        """Test scrolling up."""
        with PlaywrightBrowser(headless=True) as browser:
            result = browser.scroll('up', amount=300)
            
            call_args = self.mock_page.evaluate.call_args[0][0]
            assert 'scrollBy' in call_args
            assert '-300' in call_args
    
    def test_go_back(self):
        """Test going back in history."""
        with PlaywrightBrowser(headless=True) as browser:
            result = browser.go_back()
            
            self.mock_page.go_back.assert_called_once()
            assert 'url' in result
    
    def test_go_forward(self):
        """Test going forward in history."""
        with PlaywrightBrowser(headless=True) as browser:
            result = browser.go_forward()
            
            self.mock_page.go_forward.assert_called_once()
            assert 'url' in result
    
    def test_wait(self):
        """Test waiting."""
        with patch('src.browser.playwright_browser.time.sleep') as mock_sleep:
            with PlaywrightBrowser(headless=True) as browser:
                result = browser.wait(3)
                
                mock_sleep.assert_called_with(3)
                assert 'url' in result
    
    def test_hover(self):
        """Test hovering over element."""
        with PlaywrightBrowser(headless=True) as browser:
            browser.element_selector.element_map['elem_0'] = {
                'id': 'elem_0',
                'selector': '#hover-btn'
            }
            
            result = browser.hover('elem_0')
            
            self.mock_page.hover.assert_called_with('#hover-btn')
    
    def test_press_key(self):
        """Test pressing a keyboard key."""
        with PlaywrightBrowser(headless=True) as browser:
            result = browser.press_key('Enter')
            
            self.mock_page.keyboard.press.assert_called_with('Enter')
            assert 'url' in result
    
    def test_take_screenshot(self):
        """Test taking a screenshot."""
        with PlaywrightBrowser(headless=True) as browser:
            screenshot = browser.take_screenshot()
            
            assert screenshot == b'fake_screenshot_data'
            self.mock_page.screenshot.assert_called()
    
    def test_get_url(self):
        """Test getting current URL."""
        with PlaywrightBrowser(headless=True) as browser:
            url = browser.get_url()
            
            assert url == 'https://google.com'
    
    def test_get_title(self):
        """Test getting page title."""
        with PlaywrightBrowser(headless=True) as browser:
            title = browser.get_title()
            
            assert title == 'Google'
    
    def test_click_element_not_found(self):
        """Test clicking non-existent element raises error."""
        with PlaywrightBrowser(headless=True) as browser:
            with pytest.raises(ValueError, match="Element .* not found"):
                browser.click('elem_999')
    
    def test_type_text_element_not_found(self):
        """Test typing into non-existent element raises error."""
        with PlaywrightBrowser(headless=True) as browser:
            with pytest.raises(ValueError, match="Element .* not found"):
                browser.type_text('elem_999', 'text')
    
    def test_screen_size(self):
        """Test getting screen size."""
        with PlaywrightBrowser(headless=True, screen_size=(1920, 1080)) as browser:
            size = browser.screen_size()
            
            # Should use viewport size from page
            assert size == (1440, 900)
    
    def test_screen_size_fallback(self):
        """Test screen size fallback when viewport not available."""
        self.mock_page.viewport_size = None
        
        with PlaywrightBrowser(headless=True, screen_size=(1920, 1080)) as browser:
            size = browser.screen_size()
            
            # Should fallback to provided size
            assert size == (1920, 1080)
