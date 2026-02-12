"""
Browser Agent - Main agent that navigates web pages using DOM extraction and LLM.
"""
from typing import Dict, List, Any, Optional
import json
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.json import JSON

from ..browser.playwright_browser import PlaywrightBrowser
from ..llm.client import LLMClient, create_browser_tools, LLMResponse


console = Console()


SYSTEM_PROMPT = """You are a helpful web browsing agent. Your task is to navigate web pages and complete user requests.

You have access to a web browser and can see the DOM structure of pages. You will receive:
1. The current URL
2. The page title
3. A list of interactive elements with their IDs, types, and text content
4. Page headings for context
5. Form structures if present

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


class BrowserAgent:
    """Agent that uses LLM to control a web browser."""
    
    def __init__(
        self,
        browser: PlaywrightBrowser,
        llm_client: LLMClient,
        max_iterations: int = 20,
        verbose: bool = True
    ):
        """
        Initialize browser agent.
        
        Args:
            browser: Playwright browser instance
            llm_client: LLM client instance
            max_iterations: Maximum number of action iterations
            verbose: Whether to print detailed information
        """
        self.browser = browser
        self.llm = llm_client
        self.max_iterations = max_iterations
        self.verbose = verbose
        self.conversation_history: List[Dict[str, str]] = []
        self.tools = create_browser_tools()
        self.iteration_count = 0
        self.task_complete = False
        self.final_result = None
    
    def run(self, query: str) -> Dict[str, Any]:
        """
        Run the agent to complete a task.
        
        Args:
            query: User's task/query
            
        Returns:
            Dictionary with task result
        """
        console.print(Panel(f"[bold cyan]Task:[/bold cyan] {query}", expand=False))
        
        # Initialize conversation with task
        self.conversation_history = []
        self.iteration_count = 0
        self.task_complete = False
        
        # Get initial page state
        state = self.browser.get_state()
        
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
                console.print(f"\n[bold yellow]═══ Iteration {self.iteration_count} ═══[/bold yellow]")
            
            # Get LLM response
            response = self._get_llm_response()
            
            # Execute actions
            new_state = self._execute_actions(response)
            
            # Check if task is complete
            if self.task_complete:
                break
            
            # Add new state to conversation
            if new_state:
                state_message = self._format_page_state(new_state)
                self.conversation_history.append({
                    "role": "user",
                    "content": state_message
                })
        
        # Return final result
        if self.task_complete:
            return {
                "success": True,
                "result": self.final_result,
                "iterations": self.iteration_count
            }
        else:
            return {
                "success": False,
                "result": "Maximum iterations reached without completing task",
                "iterations": self.iteration_count
            }
    
    def _get_llm_response(self) -> LLMResponse:
        """Get response from LLM."""
        if self.verbose:
            with console.status("[bold green]Thinking...", spinner="dots"):
                response = self.llm.generate_with_retry(
                    messages=self.conversation_history,
                    system_prompt=SYSTEM_PROMPT,
                    tools=self.tools
                )
        else:
            response = self.llm.generate_with_retry(
                messages=self.conversation_history,
                system_prompt=SYSTEM_PROMPT,
                tools=self.tools
            )
        
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
                console.print("[yellow]⚠️  No actions returned by LLM[/yellow]")
            return None
        
        new_state = None
        
        for action in actions:
            action_name = action.get("name")
            args = action.get("arguments", {})
            
            if self.verbose:
                console.print(f"\n[bold cyan]Action:[/bold cyan] {action_name}")
                console.print(f"[cyan]Args:[/cyan] {json.dumps(args, indent=2)}")
            
            try:
                new_state = self._execute_single_action(action_name, args)
            except Exception as e:
                error_msg = f"Error executing {action_name}: {str(e)}"
                console.print(f"[bold red]❌ {error_msg}[/bold red]")
                
                # Add error to conversation
                self.conversation_history.append({
                    "role": "user",
                    "content": f"Error: {error_msg}. Please try a different approach."
                })
                break
        
        return new_state
    
    def _execute_single_action(self, action_name: str, args: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Execute a single browser action."""
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
            
            if self.verbose:
                success_icon = "✅" if args.get("success", True) else "❌"
                console.print(Panel(
                    f"{success_icon} {self.final_result}",
                    title="[bold green]Task Complete[/bold green]",
                    expand=False
                ))
            
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
                    elem_desc += f" → {elem['href'][:50]}"
                
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
        console.print(f"💾 Conversation saved to: {filepath}")
