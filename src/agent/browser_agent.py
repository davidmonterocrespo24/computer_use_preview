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


SYSTEM_PROMPT = """You are a helpful web browsing agent. Your task is to navigate web pages and complete user requests.

You have access to a web browser and can see the DOM structure of pages. You will receive:
1. The current URL
2. The page title
3. A list of interactive elements with their IDs, types, and text content
4. Page headings for context
5. Form structures if present
6. (When visual mode is active) An annotated screenshot with numbered element boxes

When you receive page information, analyze the available elements and decide what actions to take.

Available actions:
- navigate(url): Navigate to a URL
- click(element_id): Click on an element using its ID (e.g., "elem_0")
- type_text(element_id, text, press_enter): Type text into an input field
- scroll(direction): Scroll the page (up/down/left/right)
- go_back(): Go back in browser history
- wait(seconds): Wait for content to load
- task_complete(result, success): Mark the task as complete with the final result

Important guidelines:
1. Always examine the page elements carefully before acting
2. Look for elements with relevant text, aria-labels, or placeholders
3. Use element IDs exactly as provided (e.g., "elem_0", "elem_1")
4. If you can't find what you're looking for, try scrolling or navigating
5. Break complex tasks into steps
6. When the task is complete, always call task_complete() with the result

Example element:
{
  "id": "elem_0",
  "tag": "input",
  "type": "text",
  "placeholder": "Search...",
  "aria_label": "Search box"
}

To interact with this element, use: type_text("elem_0", "your search query", True)

Think step by step and explain your reasoning before taking each action.
"""

VISION_SYSTEM_PROMPT = """You are a helpful web browsing agent operating in VISUAL MODE.

The screenshot you receive has colored numbered boxes drawn over each interactive element:
- Green boxes: buttons and links
- Blue boxes: input fields and text areas
- Yellow boxes: other interactive elements

Each box label shows the element ID and tag (e.g., "elem_3: button"). Use these element IDs
exactly as labeled when calling actions.

Available actions:
- navigate(url): Navigate to a URL
- click(element_id): Click on an element using its ID
- type_text(element_id, text, press_enter): Type text into an input field
- scroll(direction): Scroll the page (up/down/left/right)
- go_back(): Go back in browser history
- wait(seconds): Wait for content to load
- task_complete(result, success): Mark the task as complete with the final result

Analyze what you see in the annotated screenshot, identify the relevant elements, and take action.
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
    
    def _format_page_state(self, state: Dict[str, Any], initial_query: Optional[str] = None) -> str:
        """Format page state for LLM."""
        dom = state.get("dom", {})
        
        message_parts = []
        
        if initial_query:
            message_parts.append(f"Task: {initial_query}\n")
        
        message_parts.append(f"Current URL: {state['url']}")
        message_parts.append(f"Page Title: {state['title']}\n")
        
        # Add headings for context
        if dom.get("headings"):
            message_parts.append("Page Structure:")
            for h in dom["headings"][:5]:  # Top 5 headings
                indent = "  " * (h["level"] - 1)
                message_parts.append(f"{indent}H{h['level']}: {h['text']}")
            message_parts.append("")
        
        # Add interactive elements
        elements = dom.get("elements", [])
        if elements:
            message_parts.append(f"Interactive Elements ({len(elements)} total):")
            
            # Show detailed info for first 20 elements, then summary
            shown_elements = elements[:20]
            
            for elem in shown_elements:
                elem_desc = f"  [{elem['id']}] {elem['tag']}"
                
                if elem.get('type'):
                    elem_desc += f" (type={elem['type']})"
                
                if elem.get('text'):
                    elem_desc += f" - \"{elem['text'][:50]}\""
                elif elem.get('aria_label'):
                    elem_desc += f" - aria-label: \"{elem['aria_label']}\""
                elif elem.get('placeholder'):
                    elem_desc += f" - placeholder: \"{elem['placeholder']}\""
                elif elem.get('alt'):
                    elem_desc += f" - alt: \"{elem['alt']}\""
                
                if elem.get('href'):
                    elem_desc += f" -> {elem['href'][:50]}"
                
                message_parts.append(elem_desc)
            
            if len(elements) > 20:
                message_parts.append(f"  ... and {len(elements) - 20} more elements")
        else:
            message_parts.append("No interactive elements found on this page.")
        
        # Add forms if present
        forms = dom.get("forms", [])
        if forms:
            message_parts.append(f"\nForms ({len(forms)}):")
            for i, form in enumerate(forms[:3]):  # Show top 3 forms
                message_parts.append(f"  Form {i+1}:")
                message_parts.append(f"    Action: {form.get('action', 'N/A')}")
                message_parts.append(f"    Method: {form.get('method', 'N/A')}")
                message_parts.append(f"    Fields: {len(form.get('fields', []))}")
        
        return "\n".join(message_parts)
    
    def get_conversation_history(self) -> List[Dict[str, str]]:
        """Get the conversation history."""
        return self.conversation_history.copy()
    
    def save_conversation(self, filepath: str):
        """Save conversation history to a file."""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.conversation_history, f, indent=2, ensure_ascii=False)
        console.print(f"Conversation saved to: {filepath}")
