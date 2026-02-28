"""
Real navigation evaluation tests.

These tests require:
  - Ollama running locally:  ollama serve
  - Playwright installed:    playwright install chromium

Run with:
  pytest tests/test_real_navigation.py -v -s

Skip in CI:
  pytest tests/ -m "not requires_browser"
"""
import pytest
import time
import base64
import numpy as np

from src.llm.orchestrator import ModelOrchestrator, ModelTier, SelectionContext
from src.llm.client import LLMClient, LLMResponse

OLLAMA_BASE_URL = "http://localhost:11434/v1"
OLLAMA_NATIVE_URL = "http://localhost:11434"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def ollama_is_running() -> bool:
    """Check if Ollama is reachable."""
    import requests
    try:
        r = requests.get(f"{OLLAMA_NATIVE_URL}/api/tags", timeout=3)
        return r.status_code == 200
    except Exception:
        return False


def make_ctx(**kwargs) -> SelectionContext:
    defaults = dict(
        query="test",
        dom_element_count=10,
        iteration_number=1,
        current_url="https://example.com",
        stuck_count=0,
        page_has_canvas=False,
        page_has_iframes=False,
        page_has_forms=False,
        last_action_type="navigate",
        screenshot_b64=None,
    )
    defaults.update(kwargs)
    return SelectionContext(**defaults)


# ---------------------------------------------------------------------------
# Tests: Orchestrator model selection (no browser, no real LLM)
# ---------------------------------------------------------------------------

class TestOrchestratorSelectionLogic:
    """Verify orchestrator routing decisions without any real model calls."""

    def test_orchestrator_selects_vision_when_stuck(self):
        orch = ModelOrchestrator()
        ctx = make_ctx(stuck_count=3)
        tier, reason = orch.select_tier(ctx)
        assert tier == ModelTier.FULL_VIS

    def test_orchestrator_selects_vision_on_sparse_dom(self):
        orch = ModelOrchestrator()
        ctx = make_ctx(dom_element_count=2, iteration_number=2)
        tier, reason = orch.select_tier(ctx)
        assert tier == ModelTier.FULL_VIS

    def test_orchestrator_selects_ultra_fast_for_scroll(self):
        orch = ModelOrchestrator()
        ctx = make_ctx(last_action_type="scroll", iteration_number=2)
        tier, _ = orch.select_tier(ctx)
        assert tier == ModelTier.ULTRA_FAST

    def test_orchestrator_selects_quick_vis_for_canvas(self):
        orch = ModelOrchestrator()
        ctx = make_ctx(page_has_canvas=True)
        tier, _ = orch.select_tier(ctx)
        assert tier == ModelTier.QUICK_VIS


# ---------------------------------------------------------------------------
# Tests: Vision annotation (no real LLM, no browser — just image processing)
# ---------------------------------------------------------------------------

class TestAnnotatedScreenshot:
    """Test that screenshots are annotated correctly with DOM element boxes."""

    def test_annotate_screenshot_with_dom_returns_ndarray(self):
        from src.vision.detector import LightweightVisionDetector
        detector = LightweightVisionDetector(use_yolo=False)

        # Create a blank white image (100x200 BGR)
        image = np.ones((200, 400, 3), dtype=np.uint8) * 255

        dom_elements = [
            {"id": "elem_0", "tag": "button", "text": "Click me",
             "bbox": {"x": 10, "y": 10, "width": 100, "height": 30}},
            {"id": "elem_1", "tag": "input",
             "bbox": {"x": 10, "y": 60, "width": 200, "height": 25}},
            {"id": "elem_2", "tag": "a", "text": "Link",
             "bbox": None},  # no bbox — should be skipped
        ]

        annotated = detector.annotate_screenshot_with_dom(image, dom_elements)

        assert isinstance(annotated, np.ndarray)
        assert annotated.shape == image.shape
        # Annotated image should differ from the original (boxes drawn)
        assert not np.array_equal(annotated, image)

    def test_annotate_skips_elements_without_bbox(self):
        from src.vision.detector import LightweightVisionDetector
        detector = LightweightVisionDetector(use_yolo=False)
        image = np.ones((200, 400, 3), dtype=np.uint8) * 255

        # All elements without bbox — image should be unchanged
        dom_elements = [
            {"id": "elem_0", "tag": "button", "bbox": None},
            {"id": "elem_1", "tag": "a"},
        ]
        annotated = detector.annotate_screenshot_with_dom(image, dom_elements)
        assert np.array_equal(annotated, image)

    def test_annotated_image_roundtrip_base64(self):
        """Annotated image can be base64-encoded and decoded."""
        from src.vision.detector import LightweightVisionDetector, image_to_base64, image_from_base64
        detector = LightweightVisionDetector(use_yolo=False)
        image = np.ones((100, 100, 3), dtype=np.uint8) * 200

        annotated = detector.annotate_screenshot_with_dom(image, [])
        b64 = image_to_base64(annotated)
        assert isinstance(b64, str)
        assert len(b64) > 0

        recovered = image_from_base64(b64)
        assert recovered.shape == annotated.shape


# ---------------------------------------------------------------------------
# Tests: Real Ollama text models (require Ollama running)
# ---------------------------------------------------------------------------

@pytest.mark.requires_browser
@pytest.mark.slow
class TestRealOllamaTextModels:
    """Verify installed text models can generate responses via Ollama."""

    @pytest.fixture(autouse=True)
    def check_ollama(self):
        if not ollama_is_running():
            pytest.skip("Ollama is not running. Start with: ollama serve")

    def _make_client(self, model: str) -> LLMClient:
        return LLMClient(
            provider="local",
            model=model,
            base_url=OLLAMA_BASE_URL,
            temperature=0.3,
            max_tokens=256,
        )

    def _simple_prompt(self, client: LLMClient) -> LLMResponse:
        messages = [{"role": "user", "content": "Reply with only: READY"}]
        return client.generate(messages, system_prompt="You are a test assistant.")

    def test_ultra_fast_model_responds(self):
        client = self._make_client("lfm2.5-thinking:latest")
        response = self._simple_prompt(client)
        assert response.content is not None
        assert len(response.content.strip()) > 0

    def test_balanced_model_responds(self):
        client = self._make_client("ministral-3:3b")
        response = self._simple_prompt(client)
        assert response.content is not None
        assert len(response.content.strip()) > 0

    def test_granite_fast_model_responds(self):
        client = self._make_client("granite4:3b")
        response = self._simple_prompt(client)
        assert response.content is not None
        assert len(response.content.strip()) > 0


# ---------------------------------------------------------------------------
# Tests: Real Ollama vision model (require Ollama running)
# ---------------------------------------------------------------------------

@pytest.mark.requires_browser
@pytest.mark.slow
class TestRealOllamaVisionModel:
    """Verify vision model can analyze an annotated screenshot."""

    @pytest.fixture(autouse=True)
    def check_ollama(self):
        if not ollama_is_running():
            pytest.skip("Ollama is not running. Start with: ollama serve")

    def test_vision_model_receives_annotated_screenshot(self):
        """Send a synthetic annotated screenshot and verify the model responds."""
        from src.vision.detector import LightweightVisionDetector, image_to_base64

        # Build a synthetic screenshot with annotated boxes
        image = np.ones((300, 600, 3), dtype=np.uint8) * 240
        detector = LightweightVisionDetector(use_yolo=False)
        dom_elements = [
            {"id": "elem_0", "tag": "button", "text": "Search",
             "bbox": {"x": 20, "y": 20, "width": 120, "height": 35}},
            {"id": "elem_1", "tag": "input", "placeholder": "Search...",
             "bbox": {"x": 20, "y": 80, "width": 300, "height": 30}},
        ]
        annotated = detector.annotate_screenshot_with_dom(image, dom_elements)
        annotated_b64 = image_to_base64(annotated)

        client = LLMClient(
            provider="local",
            model="qwen3-vl:4b",
            base_url=OLLAMA_BASE_URL,
            temperature=0.3,
            max_tokens=256,
        )
        messages = [{
            "role": "user",
            "content": (
                "This is a web page screenshot with highlighted interactive elements. "
                "List the element IDs you can see (boxes labeled with elem_0, elem_1, etc.)."
            )
        }]

        response = client.generate_with_image(messages, system_prompt=None, image_b64=annotated_b64)

        assert response is not None
        assert response.finish_reason != "error", f"Vision model error: {response.content}"
        assert len(response.content.strip()) > 0


# ---------------------------------------------------------------------------
# Tests: Full pipeline (orchestrator + real browser)
# ---------------------------------------------------------------------------

@pytest.mark.requires_browser
@pytest.mark.slow
class TestFullPipelineWithOrchestrator:
    """
    End-to-end tests with a real browser and real Ollama models.
    Verifies that the orchestrator + browser agent pipeline executes without errors.
    """

    @pytest.fixture(autouse=True)
    def check_ollama(self):
        if not ollama_is_running():
            pytest.skip("Ollama is not running. Start with: ollama serve")

    def test_orchestrator_pipeline_simple_navigation(self):
        """Navigate to example.com using the orchestrator pipeline."""
        from src.browser.playwright_browser import PlaywrightBrowser
        from src.agent.browser_agent import BrowserAgent
        from src.dom.extractor import DOMExtractor

        dom_extractor = DOMExtractor(max_text_length=200, max_elements=50)
        llm_client = LLMClient(
            provider="local",
            model="ministral-3:3b",
            base_url=OLLAMA_BASE_URL,
            temperature=0.3,
            max_tokens=512,
        )
        orchestrator = ModelOrchestrator()

        with PlaywrightBrowser(
            headless=True,
            initial_url="about:blank",
            dom_extractor=dom_extractor,
        ) as browser:
            agent = BrowserAgent(
                browser=browser,
                llm_client=llm_client,
                max_iterations=3,
                verbose=False,
                orchestrator=orchestrator,
                vision_enabled=False,
            )
            result = agent.run("Navigate to https://example.com and tell me the page title.")

        # The agent should have run and either completed or hit max iterations
        assert "iterations" in result
        assert result["iterations"] <= 3
        assert "model_selection_log" in result
        assert len(result["model_selection_log"]) > 0

        # Verify log structure
        first_entry = result["model_selection_log"][0]
        assert "tier" in first_entry
        assert "model" in first_entry
        assert "reason" in first_entry

    def test_vision_pipeline_activates_on_sparse_dom(self):
        """Verify vision fallback is invoked when DOM is sparse."""
        from src.browser.playwright_browser import PlaywrightBrowser
        from src.agent.browser_agent import BrowserAgent
        from src.dom.extractor import DOMExtractor

        dom_extractor = DOMExtractor(max_text_length=200, max_elements=50)
        llm_client = LLMClient(
            provider="local",
            model="ministral-3:3b",
            base_url=OLLAMA_BASE_URL,
            temperature=0.3,
            max_tokens=512,
        )
        # Use very high threshold so vision always triggers
        orchestrator = ModelOrchestrator(
            dom_sparse_threshold=1000,   # almost guaranteed to trigger FULL_VIS
            stuck_iteration_threshold=1,
        )

        with PlaywrightBrowser(
            headless=True,
            initial_url="about:blank",
            dom_extractor=dom_extractor,
        ) as browser:
            agent = BrowserAgent(
                browser=browser,
                llm_client=llm_client,
                max_iterations=2,
                verbose=False,
                orchestrator=orchestrator,
                vision_enabled=True,  # enable vision fallback
            )
            result = agent.run("Tell me what you see on this page.")

        # Verify vision tier appears in the log
        log = result.get("model_selection_log", [])
        tiers_used = {entry["tier"] for entry in log}
        # At least one vision tier should have been selected
        assert tiers_used & {"FULL_VIS", "QUICK_VIS"}, (
            f"Expected vision tier in log, got: {tiers_used}"
        )
