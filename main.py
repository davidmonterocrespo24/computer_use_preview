"""
Main entry point for the browser agent.
"""
import argparse
import os
from pathlib import Path

from src.browser.playwright_browser import PlaywrightBrowser
from src.llm.client import LLMClient
from src.agent.browser_agent import BrowserAgent
from src.dom.extractor import DOMExtractor


def main():
    parser = argparse.ArgumentParser(
        description="Browser Agent - Navigate web pages using DOM extraction and LLM"
    )
    
    parser.add_argument(
        "--query",
        type=str,
        required=True,
        help="The task for the browser agent to complete"
    )
    
    parser.add_argument(
        "--provider",
        type=str,
        choices=["openai", "anthropic", "local"],
        default="openai",
        help="LLM provider to use"
    )
    
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o-mini",
        help="Model name (e.g., gpt-4o-mini, claude-3-5-sonnet-20241022, etc.)"
    )
    
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run browser in headless mode"
    )
    
    parser.add_argument(
        "--initial-url",
        type=str,
        default="https://www.google.com",
        help="Initial URL to load"
    )
    
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=20,
        help="Maximum number of iterations"
    )
    
    parser.add_argument(
        "--api-key",
        type=str,
        help="API key for LLM provider (overrides environment variable)"
    )
    
    parser.add_argument(
        "--base-url",
        type=str,
        help="Base URL for local LLM server"
    )
    
    parser.add_argument(
        "--save-conversation",
        type=str,
        help="Path to save conversation history"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=True,
        help="Print detailed information"
    )
    
    args = parser.parse_args()
    
    # Check for API key
    if args.provider == "openai" and not args.api_key and not os.getenv("OPENAI_API_KEY"):
        print("Error: OpenAI API key required. Set OPENAI_API_KEY environment variable or use --api-key")
        return 1
    
    if args.provider == "anthropic" and not args.api_key and not os.getenv("ANTHROPIC_API_KEY"):
        print("Error: Anthropic API key required. Set ANTHROPIC_API_KEY environment variable or use --api-key")
        return 1
    
    print("🤖 Initializing Browser Agent...")
    print(f"   Provider: {args.provider}")
    print(f"   Model: {args.model}")
    print(f"   Headless: {args.headless}")
    print(f"   Max Iterations: {args.max_iterations}\n")
    
    # Initialize components
    dom_extractor = DOMExtractor(max_text_length=200, max_elements=100)
    
    llm_client = LLMClient(
        provider=args.provider,
        model=args.model,
        api_key=args.api_key,
        base_url=args.base_url,
        temperature=0.7,
        max_tokens=4096
    )
    
    # Run agent
    with PlaywrightBrowser(
        headless=args.headless,
        initial_url=args.initial_url,
        dom_extractor=dom_extractor
    ) as browser:
        agent = BrowserAgent(
            browser=browser,
            llm_client=llm_client,
            max_iterations=args.max_iterations,
            verbose=args.verbose
        )
        
        result = agent.run(args.query)
        
        # Save conversation if requested
        if args.save_conversation:
            agent.save_conversation(args.save_conversation)
        
        # Print final result
        print("\n" + "="*60)
        if result["success"]:
            print(f"✅ Task completed successfully in {result['iterations']} iterations")
            print(f"Result: {result['result']}")
        else:
            print(f"❌ Task failed after {result['iterations']} iterations")
            print(f"Reason: {result['result']}")
        print("="*60)
    
    return 0 if result["success"] else 1


if __name__ == "__main__":
    exit(main())
