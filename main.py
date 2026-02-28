"""
Main entry point for the browser agent.
"""
import argparse
import json
import logging
import os
from pathlib import Path

import yaml

from src.browser.playwright_browser import PlaywrightBrowser
from src.llm.client import LLMClient
from src.llm.orchestrator import ModelOrchestrator
from src.agent.browser_agent import BrowserAgent
from src.dom.extractor import DOMExtractor


def setup_logging(level: str = "INFO", log_file: str | None = None) -> None:
    """Configure logging for all src.* modules.

    Format example:
      2026-02-27 12:34:56 INFO  [orchestrator ] [iter=2] Model selected | tier=BALANCED ...
    """
    fmt = "%(asctime)s %(levelname)-5s [%(name)-16s] %(message)s"
    datefmt = "%H:%M:%S"
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if log_file:
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))

    logging.basicConfig(level=getattr(logging, level.upper(), logging.INFO),
                        format=fmt, datefmt=datefmt, handlers=handlers)

    # Keep third-party libs quiet unless explicitly set to DEBUG
    for noisy in ("httpx", "httpcore", "openai", "anthropic", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration from YAML file. Returns empty dict if not found."""
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except FileNotFoundError:
        return {}


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

    parser.add_argument(
        "--no-orchestrator",
        action="store_true",
        default=False,
        help="Disable dynamic model selection (use single --model instead)"
    )

    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to config.yaml"
    )

    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (default: INFO)"
    )

    parser.add_argument(
        "--log-file",
        type=str,
        default=None,
        help="Save logs to this file in addition to stdout"
    )

    args = parser.parse_args()

    # Configure logging as early as possible
    setup_logging(level=args.log_level, log_file=args.log_file)

    # Load config
    config = load_config(args.config)

    # Check for API key (only required for non-local providers)
    if args.provider == "openai" and not args.api_key and not os.getenv("OPENAI_API_KEY"):
        print("Error: OpenAI API key required. Set OPENAI_API_KEY environment variable or use --api-key")
        return 1

    if args.provider == "anthropic" and not args.api_key and not os.getenv("ANTHROPIC_API_KEY"):
        print("Error: Anthropic API key required. Set ANTHROPIC_API_KEY environment variable or use --api-key")
        return 1

    # Determine if orchestrator should be used
    orch_config = config.get("orchestrator", {})
    use_orchestrator = (
        not args.no_orchestrator
        and orch_config.get("enabled", False)
        and args.provider == "local"  # orchestrator only makes sense with local/Ollama
    )

    vision_enabled = config.get("vision", {}).get("enabled", False)

    print("Initializing Browser Agent...")
    print(f"   Provider: {args.provider}")
    if use_orchestrator:
        print("   Orchestrator: enabled (dynamic model selection)")
    else:
        print(f"   Model: {args.model}")
    print(f"   Vision: {'enabled' if vision_enabled else 'disabled'}")
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

    orchestrator = None
    if use_orchestrator:
        orchestrator = ModelOrchestrator(config=orch_config)

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
            verbose=args.verbose,
            orchestrator=orchestrator,
            vision_enabled=vision_enabled,
        )

        result = agent.run(args.query)

        # Save conversation if requested
        if args.save_conversation:
            agent.save_conversation(args.save_conversation)
            # Also save model selection log if orchestrator was used
            if orchestrator and result.get("model_selection_log"):
                log_path = args.save_conversation.replace(".json", "_model_log.json")
                with open(log_path, "w", encoding="utf-8") as f:
                    json.dump(result["model_selection_log"], f, indent=2)
                print(f"   Model selection log saved to: {log_path}")

        # Print final result
        print("\n" + "=" * 60)
        if result["success"]:
            print(f"Task completed successfully in {result['iterations']} iterations")
            print(f"Result: {result['result']}")
        else:
            print(f"Task failed after {result['iterations']} iterations")
            print(f"Reason: {result['result']}")
        print("=" * 60)

    return 0 if result["success"] else 1


if __name__ == "__main__":
    exit(main())
