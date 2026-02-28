"""
Run All Models — Interactive browser demo with full logging.

Executes each model config one after another with the browser VISIBLE,
captures all output per model, and prints a consolidated summary at the end.

Usage:
    python benchmark/run_all_models.py
    python benchmark/run_all_models.py --configs A B C
    python benchmark/run_all_models.py --max-iterations 6

Requires:
    - ollama serve
    - playwright install chromium
"""
import argparse
import io
import json
import os
import sys
import time
from contextlib import redirect_stdout
from dataclasses import dataclass, field
from typing import List, Optional, Dict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

OLLAMA_BASE_URL = "http://localhost:11434/v1"

TASK = (
    "Navigate to https://en.wikipedia.org/wiki/Python_(programming_language). "
    "Find out: (1) who created Python, (2) what year was Python first released. "
    "Then click on the link to 'Guido van Rossum' inside the article to go to his Wikipedia page, "
    "and tell me his birth year from his page."
)
EXPECTED_KEYWORDS = ["guido", "1956"]


# ---------------------------------------------------------------------------
# Model configs — ordered smallest to largest
# ---------------------------------------------------------------------------

@dataclass
class ModelConfig:
    id: str
    label: str
    model: str
    use_orchestrator: bool = False
    use_llm_router: bool = False
    vision_enabled: bool = False
    router_model: Optional[str] = None
    description: str = ""


CONFIGS: Dict[str, ModelConfig] = {
    "A": ModelConfig(
        id="A", label="lfm2.5-thinking (0.7B)", model="lfm2.5-thinking:latest",
        description="0.7B - smallest thinking model",
    ),
    "B": ModelConfig(
        id="B", label="granite4:3b (2.1GB)", model="granite4:3b",
        description="IBM Granite 3B text",
    ),
    "C": ModelConfig(
        id="C", label="granite3.2-vision (2.4GB)", model="granite3.2-vision:latest",
        vision_enabled=True,
        description="IBM Granite Vision 2.4GB",
    ),
    "D": ModelConfig(
        id="D", label="gemma3:4b (3.3GB)", model="gemma3:4b",
        description="Google Gemma3 4B text",
    ),
    "E": ModelConfig(
        id="E", label="qwen3-vl:4b vision (3.3GB)", model="qwen3-vl:4b",
        vision_enabled=True,
        description="Qwen3-VL 4B vision+text",
    ),
    "F": ModelConfig(
        id="F", label="ministral-3:3b (3.8GB)", model="ministral-3:3b",
        description="Mistral 3B text - baseline",
    ),
    "G": ModelConfig(
        id="G", label="ministral-3:8b (6GB)", model="ministral-3:8b",
        description="Mistral 8B text - large reference",
    ),
    "H": ModelConfig(
        id="H", label="orchestrator/rule-based", model="ministral-3:3b",
        use_orchestrator=True, use_llm_router=False, vision_enabled=True,
        description="Rule-based orchestrator with 4 tiers",
    ),
    "I": ModelConfig(
        id="I", label="orchestrator/llm-router", model="ministral-3:3b",
        use_orchestrator=True, use_llm_router=True, vision_enabled=True,
        router_model="gemma3:1b",
        description="LLM router (gemma3:1b) + ministral-3:3b",
    ),
}


# ---------------------------------------------------------------------------
# Per-run result
# ---------------------------------------------------------------------------

@dataclass
class RunResult:
    config_id: str
    label: str
    description: str
    success: bool
    accurate: bool
    iterations: int
    total_sec: float
    result_text: str
    error: str = ""
    log: str = ""          # full captured output for this run


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _sep(char="=", width=72):
    return char * width


def _banner(title: str, width=72):
    pad = max(0, width - len(title) - 4)
    left = pad // 2
    right = pad - left
    return f"{'=' * (left + 2)}  {title}  {'=' * (right + 2)}"


def _install_timing_hook(llm_client, timings: list):
    original = llm_client.generate_with_retry

    def timed(*args, **kwargs):
        t0 = time.perf_counter()
        result = original(*args, **kwargs)
        elapsed = time.perf_counter() - t0
        timings.append({"model": llm_client.model, "elapsed_sec": elapsed})
        return result

    llm_client.generate_with_retry = timed


# ---------------------------------------------------------------------------
# Run one config
# ---------------------------------------------------------------------------

def run_config(cfg: ModelConfig, task: str, max_iterations: int) -> RunResult:
    from src.browser.playwright_browser import PlaywrightBrowser
    from src.agent.browser_agent import BrowserAgent
    from src.dom.extractor import DOMExtractor
    from src.llm.client import LLMClient
    from src.llm.orchestrator import ModelOrchestrator, LLMRouter

    timings: list = []

    dom_extractor = DOMExtractor(max_text_length=200, max_elements=100)
    llm_client = LLMClient(
        provider="local",
        model=cfg.model,
        base_url=OLLAMA_BASE_URL,
        temperature=0.5,
        max_tokens=2048,
    )
    _install_timing_hook(llm_client, timings)

    orchestrator = None
    if cfg.use_orchestrator:
        orch_config = {
            "dom_sparse_threshold": 5,
            "stuck_iteration_threshold": 3,
        }
        router = None
        if cfg.use_llm_router and cfg.router_model:
            router_client = LLMClient(
                provider="local",
                model=cfg.router_model,
                base_url=OLLAMA_BASE_URL,
                temperature=0.1,
                max_tokens=16,
            )
            router = LLMRouter(client=router_client, timeout_sec=8.0)
        orchestrator = ModelOrchestrator(config=orch_config, router=router)

    t_start = time.perf_counter()
    try:
        with PlaywrightBrowser(
            headless=False,          # browser VISIBLE
            initial_url="https://www.google.com",
            dom_extractor=dom_extractor,
        ) as browser:
            agent = BrowserAgent(
                browser=browser,
                llm_client=llm_client,
                max_iterations=max_iterations,
                verbose=True,        # full agent output
                orchestrator=orchestrator,
                vision_enabled=cfg.vision_enabled,
            )
            result = agent.run(task)

    except Exception as exc:
        total_sec = time.perf_counter() - t_start
        return RunResult(
            config_id=cfg.id,
            label=cfg.label,
            description=cfg.description,
            success=False,
            accurate=False,
            iterations=0,
            total_sec=total_sec,
            result_text="",
            error=str(exc),
        )

    total_sec = time.perf_counter() - t_start
    result_text = result.get("result", "")
    accurate = all(kw in result_text.lower() for kw in EXPECTED_KEYWORDS)

    return RunResult(
        config_id=cfg.id,
        label=cfg.label,
        description=cfg.description,
        success=result.get("success", False),
        accurate=accurate,
        iterations=result.get("iterations", 0),
        total_sec=total_sec,
        result_text=result_text,
    )


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def print_summary(results: List[RunResult]):
    w = 92
    print()
    print(_sep("=", w))
    print("  RESULTS SUMMARY — All Models")
    print(f"  Task: {TASK[:75]}...")
    print(f"  Expected: {EXPECTED_KEYWORDS}")
    print(_sep("=", w))

    hdr = f"  {'ID':<3}  {'Label':<28}  {'OK':>4}  {'Acc':>4}  {'Iters':>5}  {'Total':>8}  {'Description'}"
    print(hdr)
    print("  " + "-" * (w - 2))

    for r in results:
        ok  = "YES" if r.success  else "NO "
        acc = "YES" if r.accurate else "NO "
        err = f"  ERROR: {r.error[:50]}" if r.error else ""
        print(
            f"  {r.config_id:<3}  {r.label:<28}  {ok:>4}  {acc:>4}  "
            f"{r.iterations:>5}  {r.total_sec:>7.0f}s  {r.description}{err}"
        )

    print("  " + "-" * (w - 2))
    print()

    # Per-model result text
    print(_sep("-", w))
    print("  RESULT TEXTS")
    print(_sep("-", w))
    for r in results:
        snippet = (r.result_text[:120] + "...") if len(r.result_text) > 120 else r.result_text
        print(f"  [{r.config_id}] {r.label}")
        print(f"      {snippet!r}")
        print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Run all models interactively")
    parser.add_argument(
        "--configs", nargs="+", choices=list(CONFIGS.keys()),
        default=list(CONFIGS.keys()),
        help="Which configs to run (default: all, A-I)"
    )
    parser.add_argument(
        "--max-iterations", type=int, default=8,
        help="Max iterations per model (default: 8)"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="JSON output file (default: run_all_<timestamp>.json)"
    )
    args = parser.parse_args()

    # Check Ollama
    try:
        import requests
        r = requests.get("http://localhost:11434/api/tags", timeout=3)
        r.raise_for_status()
        installed = {m["name"] for m in r.json().get("models", [])}
        print(f"Ollama models installed: {', '.join(sorted(installed))}")
    except Exception as e:
        print(f"ERROR: Ollama not reachable: {e}")
        print("Start Ollama with: ollama serve")
        sys.exit(1)

    selected = [CONFIGS[k] for k in args.configs]
    total = len(selected)

    print()
    print(_sep("="))
    print(f"  Running {total} model(s): {', '.join(c.id for c in selected)}")
    print(f"  Max iterations per model: {args.max_iterations}")
    print(f"  Browser: VISIBLE (headless=False)")
    print(f"  Task: {TASK[:70]}...")
    print(_sep("="))
    print()

    results: List[RunResult] = []

    for idx, cfg in enumerate(selected, start=1):
        print()
        print(_banner(f"CONFIG {cfg.id}/{total}:  {cfg.label}"))
        print(f"  {cfg.description}")
        print(_sep("-"))

        result = run_config(cfg, TASK, args.max_iterations)
        results.append(result)

        # Quick per-run status line
        status = "SUCCESS" if result.success else "FAILED "
        acc    = "ACCURATE" if result.accurate else "WRONG   "
        print()
        print(_sep("-"))
        print(f"  [{cfg.id}] {status} | {acc} | {result.iterations} iters | {result.total_sec:.0f}s")
        print(f"  Result: {result.result_text[:100]!r}")
        if result.error:
            print(f"  Error:  {result.error}")
        print(_sep("-"))

        # Pause between models so Ollama can unload/swap
        if idx < total:
            print(f"\n  [Waiting 5s before next model...]\n")
            time.sleep(5)

    # Final summary
    print_summary(results)

    # Save JSON
    output_file = args.output or f"run_all_{int(time.time())}.json"
    output_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        output_file,
    )
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(
            [
                {
                    "config_id": r.config_id,
                    "label": r.label,
                    "description": r.description,
                    "success": r.success,
                    "accurate": r.accurate,
                    "iterations": r.iterations,
                    "total_sec": round(r.total_sec, 1),
                    "result_text": r.result_text,
                    "error": r.error,
                }
                for r in results
            ],
            f, indent=2,
        )
    print(f"Results saved: {output_path}")


if __name__ == "__main__":
    main()
