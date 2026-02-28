# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

**Setup:**
```bash
# Install dependencies (creates venv automatically)
./install.sh        # Linux/Mac
install.bat         # Windows

# Manual setup
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
playwright install chromium
```

**Run the agent:**
```bash
# With dynamic model orchestrator (Ollama must be running: ollama serve)
python main.py --query "Search for Python tutorials" --provider local --base-url http://localhost:11434/v1
# Without orchestrator (single model)
python main.py --query "..." --provider local --model ministral-3:3b --no-orchestrator
python main.py --query "..." --provider openai
python main.py --query "..." --provider anthropic --headless
python quickstart.py   # Interactive wizard
python examples.py     # Pre-built examples
```

**Testing:**
```bash
pytest                                           # All tests
pytest tests/test_model_orchestrator.py         # Orchestrator unit tests (no Ollama needed)
pytest tests/test_real_navigation.py            # Real Ollama + browser tests (requires: ollama serve)
pytest tests/test_agent.py                      # Single test file
pytest tests/test_agent.py::TestBrowserAgent::test_run_success  # Single test
pytest -m "not requires_browser"                # Skip tests needing real browser/Ollama
pytest --cov=src --cov-report=html              # With coverage
./run_tests.sh      # Linux/Mac runner
run_tests.bat       # Windows runner
```

**Linting/Formatting:**
```bash
black src/ tests/    # Format code
```

## Architecture

The system is an LLM-driven browser automation agent that uses DOM extraction (not vision/screenshots) for efficiency on CPU-only environments.

**Execution flow:**
```
main.py (CLI args) → BrowserAgent.run(query)
  └─ Loop until task_complete or max_iterations:
       1. PlaywrightBrowser.get_state() → DOM snapshot
       2. DOMExtractor.extract() → Interactive element list
       3. LLMClient.generate() → Action decision
       4. Execute action (navigate/click/type/scroll/etc.)
```

**Module roles:**
- [src/agent/browser_agent.py](src/agent/browser_agent.py) — Orchestrates the LLM ↔ browser loop. Accepts optional `ModelOrchestrator` for dynamic model selection. Tracks `stuck_count` and URL history to detect when vision fallback is needed. Actions: `navigate`, `click`, `type_text`, `scroll`, `go_back`, `wait`, `task_complete`.
- [src/llm/orchestrator.py](src/llm/orchestrator.py) — **New.** `ModelOrchestrator` selects the right model tier per iteration based on `SelectionContext` (DOM quality, stuck count, page type). Tiers: ULTRA_FAST (`lfm2.5-thinking`), BALANCED (`ministral-3:3b`), QUICK_VIS (`granite3.2-vision`), FULL_VIS (`qwen3-vl:4b`). Records selection decisions in a log.
- [src/llm/client.py](src/llm/client.py) — Unified interface for OpenAI, Anthropic, and local (Ollama) providers. `generate_with_image()` calls Ollama's native `/api/chat` endpoint with base64 image for vision models.
- [src/vision/detector.py](src/vision/detector.py) — `annotate_screenshot_with_dom()` draws numbered bounding boxes on the screenshot for each DOM element (green=button/link, blue=input, yellow=other). The annotated image is sent to the vision LLM so it can see element positions. Also has OpenCV-based button/field detection and optional YOLO.
- [src/browser/playwright_browser.py](src/browser/playwright_browser.py) — Wraps Playwright (Chromium). Returns page state including screenshot as base64 and HTML (for canvas/iframe detection).
- [src/dom/extractor.py](src/dom/extractor.py) — Parses HTML via BeautifulSoup. Extracts interactive elements with bounding boxes (used by `annotate_screenshot_with_dom`).
- [src/dom/selector.py](src/dom/selector.py) — Maps element numeric IDs to CSS selectors and XPath expressions.

**Model orchestrator selection logic (priority order):**
1. `stuck_count >= threshold` OR sparse DOM → **FULL_VIS** (`qwen3-vl:4b`)
2. Page has canvas or iframes → **QUICK_VIS** (`granite3.2-vision`)
3. Simple/repetitive action (`scroll`, `wait`, `go_back`) after iter 1 → **ULTRA_FAST** (`lfm2.5-thinking`)
4. Form page or auth/checkout URL → **BALANCED** (`ministral-3:3b`)
5. Default → **BALANCED**

**Vision fallback flow:**
```
DOM sparse or stuck → take screenshot → annotate_screenshot_with_dom()
  → numbered colored boxes on screenshot → send to vision LLM (qwen3-vl:4b)
  → vision LLM reasons about what it sees, references elements by ID
```

**Key design decisions:**
- DOM is always tried first (fast, free). Vision is a fallback, not the default.
- When vision is used, DOM bounding boxes are drawn ON the screenshot so the vision LLM knows exactly where each element is — no YOLO needed.
- Orchestrator uses Ollama local models only (no API cost). The single `LLMClient` path still works for OpenAI/Anthropic.
- All orchestrator unit tests run without Ollama or a browser. Real navigation tests require `ollama serve`.

## Configuration

[config.yaml](config.yaml) controls LLM provider/model, browser settings (headless, viewport, user-agent), DOM extraction limits (max elements, text length), agent iteration limit, and optional vision settings. Copy [.env.example](.env.example) to `.env` and add API keys (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`).

## Testing Notes

- Tests use `pytest-mock` — all external dependencies are mocked in [tests/conftest.py](tests/conftest.py).
- `requires_browser` marker flags tests that need a real browser (skipped by default in CI).
- Test files map 1:1 to source modules under `src/`.
