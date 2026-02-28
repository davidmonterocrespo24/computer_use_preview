"""
Browser Agent - Main agent that navigates web pages using DOM extraction and LLM.
Supports dynamic model selection via ModelOrchestrator and vision fallback.
"""
from typing import Dict, List, Any, Optional
import json
import logging
from rich.console import Console
from rich.panel import Panel

log = logging.getLogger(__name__)

from ..browser.playwright_browser import PlaywrightBrowser
from ..llm.client import LLMClient, create_browser_tools, LLMResponse
from ..llm.orchestrator import ModelOrchestrator, ModelTier, SelectionContext


console = Console(legacy_windows=False, highlight=False)


SYSTEM_PROMPT = """You are a browser automation agent controlling a real web browser.

CRITICAL RULES — you MUST follow these:
1. EVERY response MUST call exactly one tool using the JSON format below. No plain text responses.
2. If you are ALREADY on the correct URL, do NOT call navigate() again — read the elements and act.
3. Call task_complete() the MOMENT you have all the information needed. Do not keep browsing.
4. If stuck at the same page with no progress, call scroll("down") to reveal more content.
5. Use element IDs exactly as shown (e.g., "elem_5"), never guess or invent IDs.

You receive:
- Current URL and page title
- Page content snippet (first visible text)
- Interactive elements with IDs (links, buttons, inputs)
- Page headings

RESPONSE FORMAT — always output JSON like this:
{"name": "navigate", "arguments": {"url": "https://example.com"}}
{"name": "click", "arguments": {"element_id": "elem_5"}}
{"name": "task_complete", "arguments": {"result": "your answer", "success": true}}

Available tools:
- navigate(url): Go to a URL (only if you need to change pages)
- click(element_id): Click a link or button by its ID
- type_text(element_id, text, press_enter): Type into an input field
- scroll(direction): "up", "down", "left", "right"
- go_back(): Return to previous page
- wait(seconds): Wait for dynamic content
- task_complete(result, success): FINISH — provide the complete answer

TASK COMPLETION: When you have found all required information, call task_complete() immediately.
Example: {"name": "task_complete", "arguments": {"result": "Python was created by Guido van Rossum, first released in 1991. Guido van Rossum was born in 1956.", "success": true}}
"""

VISION_SYSTEM_PROMPT = """You are a browser automation agent operating in VISUAL MODE.

CRITICAL RULES:
1. EVERY response MUST call exactly one tool. Never output text without calling a tool.
2. If you already have the answer, call task_complete() immediately.
3. Use element IDs exactly as labeled in the screenshot boxes (e.g., "elem_3").

The screenshot has colored numbered boxes over interactive elements:
- Green: buttons and links
- Blue: input fields
- Yellow: other interactive elements

Tools: navigate(url), click(element_id), type_text(element_id, text, press_enter),
scroll(direction), go_back(), wait(seconds), task_complete(result, success)
"""


class BrowserAgent:
    """Agent that uses LLM to control a web browser.

    Supports dynamic model selection via ModelOrchestrator (optional).
    Falls back to a single LLMClient when no orchestrator is provided.
    """

    def __init__(
        self,
        browser: PlaywrightBrowser,
        llm_client: LLMClient,
        max_iterations: int = 20,
        verbose: bool = True,
        orchestrator: Optional[ModelOrchestrator] = None,
        vision_enabled: bool = False,
    ):
        """
        Initialize browser agent.

        Args:
            browser: Playwright browser instance
            llm_client: LLM client (used when orchestrator is None)
            max_iterations: Maximum number of action iterations
            verbose: Whether to print detailed information
            orchestrator: Optional ModelOrchestrator for dynamic model selection
            vision_enabled: Whether to use vision fallback with annotated screenshots
        """
        self.browser = browser
        self.llm = llm_client
        self.orchestrator = orchestrator
        self.vision_enabled = vision_enabled
        self.max_iterations = max_iterations
        self.verbose = verbose
        self.conversation_history: List[Dict[str, str]] = []
        self.tools = create_browser_tools()
        self.iteration_count = 0
        self.task_complete = False
        self.final_result = None
        # Tracking for orchestrator context
        self._stuck_count = 0
        self._url_history: List[str] = []
        self._last_action_type = ""
        # Lazy-loaded vision detector
        self._vision_detector = None
    
    def run(self, query: str) -> Dict[str, Any]:
        """
        Run the agent to complete a task.

        Args:
            query: User's task/query

        Returns:
            Dictionary with task result and (if orchestrator enabled) model selection log
        """
        console.print(Panel(f"[bold cyan]Task:[/bold cyan] {query}", expand=False))
        log.info("Agent starting | query=%r | max_iterations=%d | orchestrator=%s | vision=%s",
                 query, self.max_iterations,
                 "enabled" if self.orchestrator else "disabled",
                 "enabled" if self.vision_enabled else "disabled")

        # Reset state
        self.conversation_history = []
        self.iteration_count = 0
        self.task_complete = False
        self._stuck_count = 0
        self._url_history = []
        self._last_action_type = ""

        # Get initial page state
        state = self.browser.get_state()
        self._url_history.append(state.get("url", ""))
        log.info("Initial page | url=%s | title=%r", state.get("url"), state.get("title"))

        # Add initial user message
        initial_message = self._format_page_state(state, query)
        self.conversation_history.append({
            "role": "user",
            "content": initial_message
        })

        # Main agent loop
        while self.iteration_count < self.max_iterations and not self.task_complete:
            self.iteration_count += 1

            if self.verbose:
                console.print(f"\n[bold yellow]=== Iteration {self.iteration_count} ===[/bold yellow]")

            # Build selection context for orchestrator
            ctx = self._build_selection_context(query, state)
            log.info(
                "Iteration %d/%d | url=%s | dom_elems=%d | stuck=%d | last_action=%r",
                self.iteration_count, self.max_iterations,
                ctx.current_url, ctx.dom_element_count,
                ctx.stuck_count, ctx.last_action_type,
            )

            # Get LLM response (via orchestrator or direct client)
            response = self._get_llm_response(ctx, state)

            # Execute actions
            new_state = self._execute_actions(response)

            # Check if task is complete
            if self.task_complete:
                break

            # Update state tracking
            if new_state:
                current_url = new_state.get("url", "")
                prev_url = self._url_history[-1] if self._url_history else ""
                if current_url == prev_url:
                    self._stuck_count += 1
                    log.warning("Stuck at same URL for %d iteration(s) | url=%s", self._stuck_count, current_url)
                else:
                    if self._stuck_count > 0:
                        log.info("URL changed, resetting stuck counter | new_url=%s", current_url)
                    self._stuck_count = 0
                self._url_history.append(current_url)
                state = new_state

                state_message = self._format_page_state(new_state)
                self.conversation_history.append({
                    "role": "user",
                    "content": state_message
                })

        result: Dict[str, Any] = {
            "success": self.task_complete,
            "result": self.final_result if self.task_complete else "Maximum iterations reached without completing task",
            "iterations": self.iteration_count,
        }
        if self.orchestrator:
            result["model_selection_log"] = self.orchestrator.get_selection_log()

        if self.task_complete:
            log.info("Task COMPLETED in %d iterations | result=%r", self.iteration_count, self.final_result)
        else:
            log.warning("Task FAILED - max iterations (%d) reached", self.max_iterations)
        return result
    
    def _build_selection_context(self, query: str, state: Dict[str, Any]) -> SelectionContext:
        """Build a SelectionContext from the current browser state."""
        dom = state.get("dom", {})
        elements = dom.get("elements", [])
        html = state.get("html", "")
        has_canvas = "<canvas" in html.lower()
        has_iframes = "<iframe" in html.lower()
        has_forms = bool(dom.get("forms"))
        if has_canvas:
            log.debug("Page contains <canvas> element")
        if has_iframes:
            log.debug("Page contains <iframe> element")
        if has_forms:
            log.debug("Page contains forms (%d form(s))", len(dom.get("forms", [])))

        return SelectionContext(
            query=query,
            dom_element_count=len(elements),
            iteration_number=self.iteration_count,
            current_url=state.get("url", ""),
            stuck_count=self._stuck_count,
            page_has_canvas=has_canvas,
            page_has_iframes=has_iframes,
            page_has_forms=has_forms,
            last_action_type=self._last_action_type,
            screenshot_b64=state.get("screenshot"),
        )

    def _get_vision_detector(self):
        """Lazy-load the vision detector."""
        if self._vision_detector is None:
            from ..vision.detector import LightweightVisionDetector
            self._vision_detector = LightweightVisionDetector(use_yolo=False)
        return self._vision_detector

    def _vision_fallback(self, ctx: SelectionContext, state: Dict[str, Any]) -> LLMResponse:
        """
        Build an annotated screenshot (DOM elements marked with numbered boxes)
        and send it to a vision LLM for reasoning.
        """
        from ..vision.detector import image_from_base64, image_to_base64

        screenshot_b64 = ctx.screenshot_b64 or state.get("screenshot", "")
        dom_elements = state.get("dom", {}).get("elements", [])

        if screenshot_b64:
            try:
                detector = self._get_vision_detector()
                image = image_from_base64(screenshot_b64)
                annotated = detector.annotate_screenshot_with_dom(image, dom_elements)
                annotated_b64 = image_to_base64(annotated)
                log.info(
                    "Screenshot annotated | dom_elements_with_bbox=%d | image_shape=%s",
                    sum(1 for e in dom_elements if e.get("bbox")),
                    image.shape,
                )
            except Exception as exc:
                log.warning("Screenshot annotation failed, using raw screenshot: %s", exc)
                annotated_b64 = screenshot_b64
        else:
            log.warning("No screenshot available for vision fallback")
            annotated_b64 = None

        if self.verbose:
            console.print("[bold blue][Vision] Sending annotated screenshot to vision LLM[/bold blue]")
        log.info("Vision fallback | stuck=%d | dom_elems=%d", ctx.stuck_count, ctx.dom_element_count)

        return self.orchestrator.generate(
            ctx=ctx,
            messages=self.conversation_history,
            system_prompt=VISION_SYSTEM_PROMPT,
            tools=self.tools,
            image_b64=annotated_b64,
        )

    def _get_llm_response(
        self,
        ctx: Optional[SelectionContext] = None,
        state: Optional[Dict[str, Any]] = None,
    ) -> LLMResponse:
        """Get response from LLM (via orchestrator or direct client)."""
        # Pre-select tier once (used for both verbose display and routing)
        selected_tier: Optional[ModelTier] = None
        tier_reason = ""
        if self.orchestrator and ctx is not None:
            selected_tier, tier_reason = self.orchestrator.select_tier(ctx)

        def _do_generate() -> LLMResponse:
            if self.orchestrator and ctx is not None and selected_tier is not None:
                # Use vision when tier requires it and vision is enabled
                if self.vision_enabled and selected_tier in (ModelTier.QUICK_VIS, ModelTier.FULL_VIS):
                    return self._vision_fallback(ctx, state or {})
                return self.orchestrator.generate(
                    ctx=ctx,
                    messages=self.conversation_history,
                    system_prompt=SYSTEM_PROMPT,
                    tools=self.tools,
                    _preselected_tier=(selected_tier, tier_reason),
                )
            # Fallback: direct LLM client
            return self.llm.generate_with_retry(
                messages=self.conversation_history,
                system_prompt=SYSTEM_PROMPT,
                tools=self.tools,
            )

        if self.verbose:
            model_info = ""
            if selected_tier is not None:
                cfg = self.orchestrator.get_tier_config(selected_tier)
                model_info = f" [dim]({cfg['model']} - {tier_reason})[/dim]"
            with console.status(f"[bold green]Thinking...{model_info}", spinner="dots"):
                response = _do_generate()
        else:
            response = _do_generate()

        # Add assistant response to history
        self.conversation_history.append({
            "role": "assistant",
            "content": response.content
        })

        if self.verbose and response.content:
            console.print(Panel(
                response.content,
                title="[bold magenta]Reasoning[/bold magenta]",
                expand=False
            ))

        return response
    
    def _execute_actions(self, response: LLMResponse) -> Optional[Dict[str, Any]]:
        """Execute actions from LLM response."""
        actions = response.parse_actions()

        if not actions:
            if self.verbose:
                console.print("[yellow]WARNING: No actions returned by LLM[/yellow]")
            log.warning("LLM returned no actions")
            # Inject a strong recovery message so the next iteration forces a tool call
            current_url = self._url_history[-1] if self._url_history else "unknown"
            self.conversation_history.append({
                "role": "user",
                "content": (
                    'ERROR: You did not output a valid tool call. You MUST respond with JSON only.\n'
                    f'You are at: {current_url}\n'
                    'If you already have the answer, respond with exactly:\n'
                    '{"name": "task_complete", "arguments": {"result": "your full answer", "success": true}}\n'
                    'Otherwise pick an element and respond with:\n'
                    '{"name": "click", "arguments": {"element_id": "elem_X"}}\n'
                    'No other text. JSON only.'
                )
            })
            return None

        new_state = None
        log.debug("Executing %d action(s): %s", len(actions), [a.get("name") for a in actions])

        for action in actions:
            action_name = action.get("name")
            args = action.get("arguments", {})

            if self.verbose:
                console.print(f"\n[bold cyan]Action:[/bold cyan] {action_name}")
                console.print(f"[cyan]Args:[/cyan] {json.dumps(args, indent=2)}")
            log.info("Executing action | name=%s | args=%s", action_name, args)

            try:
                new_state = self._execute_single_action(action_name, args)
                if new_state:
                    log.debug("Action result | url=%s | dom_elems=%d",
                              new_state.get("url"), len(new_state.get("dom", {}).get("elements", [])))
            except Exception as e:
                error_msg = f"Error executing {action_name}: {str(e)}"
                console.print(f"[bold red]{error_msg}[/bold red]")
                log.error("Action failed | name=%s | args=%s | error=%s", action_name, args, e)

                # Add error to conversation
                self.conversation_history.append({
                    "role": "user",
                    "content": f"Error: {error_msg}. Please try a different approach."
                })
                break

        return new_state
    
    def _execute_single_action(self, action_name: str, args: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Execute a single browser action."""
        self._last_action_type = action_name

        if action_name == "navigate":
            target = args["url"].rstrip("/")
            current = (self._url_history[-1] if self._url_history else "").rstrip("/")
            if target == current:
                log.warning("Blocked same-URL navigate | url=%s", args["url"])
                self.conversation_history.append({
                    "role": "user",
                    "content": (
                        f"ERROR: You are already at {args['url']}. "
                        "Navigate was blocked. Choose a different action: "
                        "click an element, scroll, or call task_complete if you have the answer."
                    )
                })
                return None
            return self.browser.navigate(args["url"])
        
        elif action_name == "click":
            return self.browser.click(args["element_id"])
        
        elif action_name == "type_text":
            return self.browser.type_text(
                args["element_id"],
                args["text"],
                press_enter=args.get("press_enter", False)
            )
        
        elif action_name == "scroll":
            return self.browser.scroll(args["direction"])
        
        elif action_name == "go_back":
            return self.browser.go_back()
        
        elif action_name == "wait":
            seconds = args.get("seconds", 2)
            return self.browser.wait(seconds)
        
        elif action_name == "task_complete":
            self.task_complete = True
            self.final_result = args.get("result", "Task completed")
            success = args.get("success", True)

            if self.verbose:
                success_icon = "[OK]" if success else "[FAIL]"
                console.print(Panel(
                    f"{success_icon} {self.final_result}",
                    title="[bold green]Task Complete[/bold green]",
                    expand=False
                ))
            log.info("task_complete called | success=%s | result=%r", success, self.final_result)
            return None
        
        else:
            raise ValueError(f"Unknown action: {action_name}")
    
    def _extract_page_text(self, state: Dict[str, Any], max_chars: int = 500) -> str:
        """Extract readable page text for context (article intro, key facts)."""
        import re
        html = state.get("html", "")
        if not html:
            return ""
        try:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(html, "lxml")
            # Remove noise
            for tag in soup(["script", "style", "nav", "footer", "header", "noscript"]):
                tag.decompose()
            text = soup.get_text(separator=" ", strip=True)
            text = re.sub(r"\s+", " ", text).strip()
            return text[:max_chars]
        except Exception:
            return ""

    def _format_page_state(self, state: Dict[str, Any], initial_query: Optional[str] = None) -> str:
        """Format page state for LLM."""
        dom = state.get("dom", {})

        message_parts = []

        if initial_query:
            message_parts.append(f"Task: {initial_query}\n")

        message_parts.append(f"Current URL: {state['url']}")
        message_parts.append(f"Page Title: {state['title']}\n")

        # Page text snippet (article intro / key facts)
        page_text = self._extract_page_text(state)
        if page_text:
            message_parts.append("Page Content (first 500 chars):")
            message_parts.append(f"  {page_text}")
            message_parts.append("")

        # Add headings for context
        if dom.get("headings"):
            message_parts.append("Page Headings:")
            for h in dom["headings"][:8]:
                indent = "  " * (h["level"] - 1)
                message_parts.append(f"  {indent}H{h['level']}: {h['text']}")
            message_parts.append("")

        # Add interactive elements — relevant ones first
        elements = dom.get("elements", [])
        if elements:
            # Extract keywords from task for relevance sorting
            task_keywords: List[str] = []
            if initial_query:
                task_keywords = [w.lower() for w in initial_query.split() if len(w) > 3]

            def _relevance(e: Dict) -> int:
                text = (e.get("text", "") or "").lower()
                href = (e.get("href", "") or "").lower()
                combined = text + " " + href
                return sum(1 for kw in task_keywords if kw in combined)

            # Sort: relevant elements first, then by original order
            sorted_elements = sorted(elements, key=lambda e: -_relevance(e))

            # Show first 30 (more than before)
            shown = sorted_elements[:30]

            message_parts.append(f"Interactive Elements ({len(elements)} total, showing {len(shown)}):")
            for elem in shown:
                elem_desc = f"  [{elem['id']}] {elem['tag']}"

                if elem.get("type"):
                    elem_desc += f"({elem['type']})"

                label = (
                    elem.get("text")
                    or elem.get("aria_label")
                    or elem.get("placeholder")
                    or elem.get("alt")
                    or ""
                )
                if label:
                    elem_desc += f" \"{label[:60]}\""

                if elem.get("href"):
                    elem_desc += f" -> {elem['href'][:60]}"

                message_parts.append(elem_desc)

            if len(elements) > 30:
                message_parts.append(f"  ... and {len(elements) - 30} more (scroll down to reveal)")
        else:
            message_parts.append("No interactive elements found on this page.")

        # Forms
        forms = dom.get("forms", [])
        if forms:
            message_parts.append(f"\nForms ({len(forms)}):")
            for i, form in enumerate(forms[:3]):
                message_parts.append(f"  Form {i+1}: action={form.get('action','?')} fields={len(form.get('fields',[]))}")

        return "\n".join(message_parts)
    
    def get_conversation_history(self) -> List[Dict[str, str]]:
        """Get the conversation history."""
        return self.conversation_history.copy()
    
    def save_conversation(self, filepath: str):
        """Save conversation history to a file."""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.conversation_history, f, indent=2, ensure_ascii=False)
        console.print(f"Conversation saved to: {filepath}")
