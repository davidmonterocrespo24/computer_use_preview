"""
Playwright Browser - Browser automation with DOM extraction.
"""
from typing import Optional, Dict, List, Any, Tuple, Literal
from playwright.sync_api import sync_playwright, Page, Browser, BrowserContext
import time
import base64
from pathlib import Path

from ..dom.extractor import DOMExtractor
from ..dom.selector import ElementSelector


class PlaywrightBrowser:
    """Browser automation using Playwright with DOM extraction."""
    
    def __init__(
        self,
        headless: bool = False,
        screen_size: Tuple[int, int] = (1440, 900),
        initial_url: str = "https://www.google.com",
        dom_extractor: Optional[DOMExtractor] = None
    ):
        """
        Initialize Playwright browser.
        
        Args:
            headless: Run browser in headless mode
            screen_size: Browser viewport size
            initial_url: Initial URL to load
            dom_extractor: DOM extractor instance
        """
        self.headless = headless
        self.screen_size = screen_size
        self.initial_url = initial_url
        self.dom_extractor = dom_extractor or DOMExtractor()
        self.element_selector = ElementSelector()
        
        self._playwright = None
        self._browser: Optional[Browser] = None
        self._context: Optional[BrowserContext] = None
        self._page: Optional[Page] = None
    
    def __enter__(self):
        """Start browser session."""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Close browser session."""
        self.close()
    
    def start(self):
        """Start the browser."""
        print("🚀 Starting browser...")
        self._playwright = sync_playwright().start()
        
        self._browser = self._playwright.chromium.launch(
            headless=self.headless,
            args=[
                "--disable-extensions",
                "--disable-dev-shm-usage",
                "--disable-background-networking",
            ]
        )
        
        self._context = self._browser.new_context(
            viewport={
                "width": self.screen_size[0],
                "height": self.screen_size[1]
            },
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        )
        
        self._page = self._context.new_page()
        
        # Handle new pages (tabs) by redirecting to current page
        self._context.on("page", self._handle_new_page)
        
        # Navigate to initial URL
        self.navigate(self.initial_url)
        print(f"✅ Browser started at {self.initial_url}")
    
    def close(self):
        """Close the browser."""
        if self._page:
            self._page.close()
        if self._context:
            self._context.close()
        if self._browser:
            self._browser.close()
        if self._playwright:
            self._playwright.stop()
        print("🔴 Browser closed")
    
    def _handle_new_page(self, new_page: Page):
        """Handle new tabs by redirecting to current page."""
        new_url = new_page.url
        new_page.close()
        if new_url and new_url != "about:blank":
            self._page.goto(new_url)
    
    def get_state(self) -> Dict[str, Any]:
        """
        Get current page state with DOM information.
        
        Returns:
            Dictionary with page state including DOM elements
        """
        self._page.wait_for_load_state("domcontentloaded")
        time.sleep(0.5)  # Allow dynamic content to render
        
        # Get HTML content
        html = self._page.content()
        
        # Get viewport elements with bounding boxes
        viewport_elements = self._get_viewport_elements()
        
        # Extract DOM structure
        dom_data = self.dom_extractor.extract(html, viewport_elements)
        dom_data['url'] = self._page.url
        
        # Register elements for selection
        self.element_selector.register_elements(dom_data['elements'])
        
        # Optionally include screenshot (as base64)
        screenshot = self._page.screenshot(type="png", full_page=False)
        screenshot_b64 = base64.b64encode(screenshot).decode('utf-8')
        
        return {
            "url": self._page.url,
            "title": dom_data['title'],
            "dom": dom_data,
            "screenshot_base64": screenshot_b64,
            "viewport": {
                "width": self.screen_size[0],
                "height": self.screen_size[1]
            }
        }
    
    def _get_viewport_elements(self) -> List[Dict]:
        """Get elements in viewport with their bounding boxes."""
        script = """
        () => {
            const elements = [];
            const interactiveTags = ['A', 'BUTTON', 'INPUT', 'TEXTAREA', 'SELECT'];
            const interactiveRoles = ['button', 'link', 'textbox', 'checkbox', 'radio'];
            
            function isInteractive(elem) {
                if (interactiveTags.includes(elem.tagName)) return true;
                const role = elem.getAttribute('role');
                if (role && interactiveRoles.includes(role)) return true;
                if (elem.onclick || elem.getAttribute('onclick')) return true;
                return false;
            }
            
            function getSelector(elem) {
                if (elem.id) return '#' + elem.id;
                if (elem.name) return elem.tagName.toLowerCase() + '[name="' + elem.name + '"]';
                return elem.tagName.toLowerCase();
            }
            
            document.querySelectorAll('*').forEach(elem => {
                if (!isInteractive(elem)) return;
                
                const rect = elem.getBoundingClientRect();
                const style = window.getComputedStyle(elem);
                
                // Check if element is visible
                const isVisible = (
                    rect.width > 0 &&
                    rect.height > 0 &&
                    style.display !== 'none' &&
                    style.visibility !== 'hidden' &&
                    style.opacity !== '0'
                );
                
                if (isVisible) {
                    elements.push({
                        selector: getSelector(elem),
                        bbox: {
                            x: rect.x,
                            y: rect.y,
                            width: rect.width,
                            height: rect.height
                        },
                        is_visible: true,
                        in_viewport: (
                            rect.top >= 0 &&
                            rect.left >= 0 &&
                            rect.bottom <= window.innerHeight &&
                            rect.right <= window.innerWidth
                        )
                    });
                }
            });
            
            return elements;
        }
        """
        try:
            return self._page.evaluate(script)
        except Exception as e:
            print(f"⚠️  Error getting viewport elements: {e}")
            return []
    
    # ============ Browser Actions ============
    
    def navigate(self, url: str) -> Dict[str, Any]:
        """Navigate to a URL."""
        if not url.startswith(('http://', 'https://')):
            url = 'https://' + url
        
        print(f"🌐 Navigating to: {url}")
        self._page.goto(url, wait_until="domcontentloaded")
        return self.get_state()
    
    def click(self, element_id: str) -> Dict[str, Any]:
        """Click an element by its ID."""
        selector = self.element_selector.get_selector(element_id)
        if not selector:
            raise ValueError(f"Element {element_id} not found")
        
        print(f"🖱️  Clicking: {selector}")
        try:
            self._page.click(selector, timeout=5000)
            self._page.wait_for_load_state("domcontentloaded")
        except Exception as e:
            print(f"⚠️  Click failed, trying coordinates: {e}")
            # Fallback to coordinate click
            coords = self.element_selector.get_coordinates(element_id)
            if coords:
                self.click_at(*coords)
        
        return self.get_state()
    
    def click_at(self, x: int, y: int) -> Dict[str, Any]:
        """Click at specific coordinates."""
        print(f"🖱️  Clicking at: ({x}, {y})")
        self._page.mouse.click(x, y)
        self._page.wait_for_load_state("domcontentloaded")
        return self.get_state()
    
    def type_text(self, element_id: str, text: str, press_enter: bool = False, 
                  clear_first: bool = True) -> Dict[str, Any]:
        """Type text into an element."""
        selector = self.element_selector.get_selector(element_id)
        if not selector:
            raise ValueError(f"Element {element_id} not found")
        
        print(f"⌨️  Typing '{text}' into: {selector}")
        
        # Click to focus
        self._page.click(selector)
        
        # Clear existing text if requested
        if clear_first:
            self._page.fill(selector, "")
        
        # Type text
        self._page.type(selector, text)
        
        # Press enter if requested
        if press_enter:
            self._page.keyboard.press("Enter")
        
        self._page.wait_for_load_state("domcontentloaded")
        return self.get_state()
    
    def scroll(self, direction: Literal["up", "down", "left", "right"], 
               amount: int = 500) -> Dict[str, Any]:
        """Scroll the page."""
        print(f"📜 Scrolling {direction}")
        
        if direction == "down":
            self._page.evaluate(f"window.scrollBy(0, {amount})")
        elif direction == "up":
            self._page.evaluate(f"window.scrollBy(0, -{amount})")
        elif direction == "right":
            self._page.evaluate(f"window.scrollBy({amount}, 0)")
        elif direction == "left":
            self._page.evaluate(f"window.scrollBy(-{amount}, 0)")
        
        time.sleep(0.3)
        return self.get_state()
    
    def go_back(self) -> Dict[str, Any]:
        """Navigate back in history."""
        print("⬅️  Going back")
        self._page.go_back()
        self._page.wait_for_load_state("domcontentloaded")
        return self.get_state()
    
    def go_forward(self) -> Dict[str, Any]:
        """Navigate forward in history."""
        print("➡️  Going forward")
        self._page.go_forward()
        self._page.wait_for_load_state("domcontentloaded")
        return self.get_state()
    
    def wait(self, seconds: int = 2) -> Dict[str, Any]:
        """Wait for a specified time."""
        print(f"⏳ Waiting {seconds} seconds")
        time.sleep(seconds)
        return self.get_state()
    
    def hover(self, element_id: str) -> Dict[str, Any]:
        """Hover over an element."""
        selector = self.element_selector.get_selector(element_id)
        if not selector:
            raise ValueError(f"Element {element_id} not found")
        
        print(f"👆 Hovering: {selector}")
        self._page.hover(selector)
        time.sleep(0.3)
        return self.get_state()
    
    def press_key(self, key: str) -> Dict[str, Any]:
        """Press a keyboard key."""
        print(f"⌨️  Pressing key: {key}")
        self._page.keyboard.press(key)
        time.sleep(0.2)
        return self.get_state()
    
    def take_screenshot(self, filepath: Optional[str] = None) -> bytes:
        """Take a screenshot."""
        screenshot = self._page.screenshot(type="png", full_page=False)
        if filepath:
            Path(filepath).write_bytes(screenshot)
            print(f"📸 Screenshot saved to: {filepath}")
        return screenshot
    
    def get_url(self) -> str:
        """Get current URL."""
        return self._page.url
    
    def get_title(self) -> str:
        """Get current page title."""
        return self._page.title()
