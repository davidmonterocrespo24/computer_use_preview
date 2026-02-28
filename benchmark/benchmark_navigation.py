"""
Model Benchmark for Browser Navigation.

Runs the same multi-step Wikipedia task with different model configurations
and reports speed, accuracy, and iteration counts side-by-side.

Usage:
    python benchmark/benchmark_navigation.py
    python benchmark/benchmark_navigation.py --configs A B C
    python benchmark/benchmark_navigation.py --configs F G  # orchestrator configs

Requires:
    - ollama serve  (Ollama running locally)
    - playwright install chromium
"""
import argparse
import json
import sys
import time
import os
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

OLLAMA_BASE_URL = "http://localhost:11434/v1"

# ---------------------------------------------------------------------------
# Benchmark task
# ---------------------------------------------------------------------------

WIKIPEDIA_TASK = (
    "Navigate to https://en.wikipedia.org/wiki/Python_(programming_language). "
    "Find out: (1) who created Python, (2) what year was Python first released. "
    "Then click on the link to 'Guido van Rossum' inside the article to go to his Wikipedia page, "
    "and tell me his birth year from his page."
)

EXPECTED_KEYWORDS = ["guido", "1956"]   # Guido born 1956

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class IterationTiming:
    iteration: int
    model: str
    tier: str
    elapsed_sec: float
    router_elapsed_sec: float
    action: str
    url: str
    dom_elems: int


@dataclass
class BenchmarkResult:
    config_id: str
    config_label: str
    model_description: str
    success: bool
    total_sec: float
    iterations: int
    result_text: str
    accurate: bool           # result contains expected keywords
    iteration_timings: List[IterationTiming] = field(default_factory=list)
    error: str = ""

    def avg_iter_sec(self) -> float:
        timed = [t.elapsed_sec for t in self.iteration_timings if t.elapsed_sec > 0]
        return sum(timed) / len(timed) if timed else 0.0

    def total_router_sec(self) -> float:
        return sum(t.router_elapsed_sec for t in self.iteration_timings)


# ---------------------------------------------------------------------------
# Model configurations
# ---------------------------------------------------------------------------

@dataclass
class ModelConfig:
    id: str
    label: str
    model: str                          # primary model name
    use_orchestrator: bool = False
    use_llm_router: bool = False
    vision_enabled: bool = False
    router_model: Optional[str] = None  # small router model
    description: str = ""


# Ordered from smallest to largest model size
CONFIGS: Dict[str, ModelConfig] = {
    "A": ModelConfig(
        id="A", label="lfm2.5-thinking (0.7B)", model="lfm2.5-thinking:latest",
        description="0.7B thinking model - smallest",
    ),
    "B": ModelConfig(
        id="B", label="granite4:3b (2.1GB)", model="granite4:3b",
        description="3.4B text model (IBM Granite)",
    ),
    "C": ModelConfig(
        id="C", label="granite3.2-vision (2.4GB)", model="granite3.2-vision:latest",
        vision_enabled=True,
        description="2.4GB vision model (IBM Granite Vision)",
    ),
    "D": ModelConfig(
        id="D", label="gemma3:4b (3.3GB)", model="gemma3:4b",
        description="4B text model (Google Gemma3)",
    ),
    "E": ModelConfig(
        id="E", label="qwen3-vl:4b vision (3.3GB)", model="qwen3-vl:4b",
        vision_enabled=True,
        description="4.4B vision+text model",
    ),
    "F": ModelConfig(
        id="F", label="ministral-3:3b (3.8GB)", model="ministral-3:3b",
        description="3.8B text model - baseline",
    ),
    "G": ModelConfig(
        id="G", label="ministral-3:8b (6GB)", model="ministral-3:8b",
        description="8B text model - larger reference",
    ),
    "H": ModelConfig(
        id="H", label="orchestrator/rule-based", model="ministral-3:3b",
        use_orchestrator=True, use_llm_router=False, vision_enabled=True,
        description="Rule-based orchestrator (current system)",
    ),
    "I": ModelConfig(
        id="I", label="orchestrator/llm-router", model="ministral-3:3b",
        use_orchestrator=True, use_llm_router=True, vision_enabled=True,
        router_model="gemma3:1b",
        description="LLM router (gemma3:1b decides tier)",
    ),
}

# ---------------------------------------------------------------------------
# Timing hook — monkey-patch LLMClient to capture per-iteration timings
# ---------------------------------------------------------------------------

_timings_collector: List[dict] = []   # populated during each run_config call


def _install_timing_hook(llm_client):
    """Wrap generate_with_retry to capture elapsed_sec per call."""
    original = llm_client.generate_with_retry

    def timed_generate(*args, **kwargs):
        result = original(*args, **kwargs)
        _timings_collector.append({
            "model": llm_client.model,
            "elapsed_sec": result.elapsed_sec or 0.0,
        })
        return result

    llm_client.generate_with_retry = timed_generate


# ---------------------------------------------------------------------------
# Run a single config
# ---------------------------------------------------------------------------

def run_config(cfg: ModelConfig, task: str, max_iterations: int = 8) -> BenchmarkResult:
    """Run the browser agent with the given model config and return timing + result."""
    from src.browser.playwright_browser import PlaywrightBrowser
    from src.agent.browser_agent import BrowserAgent
    from src.dom.extractor import DOMExtractor
    from src.llm.client import LLMClient
    from src.llm.orchestrator import ModelOrchestrator, LLMRouter

    print(f"\n{'='*60}")
    print(f"  Config {cfg.id}: {cfg.label}")
    print(f"  {cfg.description}")
    print(f"{'='*60}")

    _timings_collector.clear()

    dom_extractor = DOMExtractor(max_text_length=200, max_elements=100)

    llm_client = LLMClient(
        provider="local",
        model=cfg.model,
        base_url=OLLAMA_BASE_URL,
        temperature=0.5,
        max_tokens=2048,
    )
    _install_timing_hook(llm_client)

    orchestrator = None
    if cfg.use_orchestrator:
        orch_config: Dict[str, Any] = {
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
            print(f"  LLM Router: {cfg.router_model}")

        orchestrator = ModelOrchestrator(config=orch_config, router=router)

    t_start = time.perf_counter()
    error_msg = ""
    try:
        with PlaywrightBrowser(
            headless=True,
            initial_url="https://www.google.com",
            dom_extractor=dom_extractor,
        ) as browser:
            agent = BrowserAgent(
                browser=browser,
                llm_client=llm_client,
                max_iterations=max_iterations,
                verbose=False,
                orchestrator=orchestrator,
                vision_enabled=cfg.vision_enabled,
            )
            result = agent.run(task)
    except Exception as exc:
        total_sec = time.perf_counter() - t_start
        print(f"  ERROR: {exc}")
        return BenchmarkResult(
            config_id=cfg.id,
            config_label=cfg.label,
            model_description=cfg.description,
            success=False,
            total_sec=total_sec,
            iterations=0,
            result_text="",
            accurate=False,
            error=str(exc),
        )

    total_sec = time.perf_counter() - t_start
    result_text = result.get("result", "")
    accurate = all(kw in result_text.lower() for kw in EXPECTED_KEYWORDS)

    # Build iteration timings from orchestrator log if available
    iteration_timings: List[IterationTiming] = []
    if orchestrator:
        for entry in orchestrator.get_selection_log():
            iteration_timings.append(IterationTiming(
                iteration=entry["iteration"],
                model=entry["model"],
                tier=entry["tier"],
                elapsed_sec=entry.get("elapsed_sec", 0.0),
                router_elapsed_sec=entry.get("router_elapsed_sec", 0.0),
                action="",
                url=entry["url"],
                dom_elems=entry["dom_element_count"],
            ))
    else:
        # For non-orchestrator configs, use the timing collector
        for i, t in enumerate(_timings_collector, start=1):
            iteration_timings.append(IterationTiming(
                iteration=i,
                model=t["model"],
                tier="SINGLE",
                elapsed_sec=t["elapsed_sec"],
                router_elapsed_sec=0.0,
                action="",
                url="",
                dom_elems=0,
            ))

    benchmark = BenchmarkResult(
        config_id=cfg.id,
        config_label=cfg.label,
        model_description=cfg.description,
        success=result.get("success", False),
        total_sec=total_sec,
        iterations=result.get("iterations", 0),
        result_text=result_text,
        accurate=accurate,
        iteration_timings=iteration_timings,
    )

    # Print per-iteration summary
    print(f"\n  Iterations: {benchmark.iterations}  |  Total: {total_sec:.0f}s  |  Success: {benchmark.success}")
    if iteration_timings:
        print(f"  {'Iter':>4}  {'Model':<30}  {'Tier':<12}  {'Time(s)':>8}  {'Router(s)':>10}")
        print(f"  {'-'*4}  {'-'*30}  {'-'*12}  {'-'*8}  {'-'*10}")
        for t in iteration_timings:
            print(f"  {t.iteration:>4}  {t.model:<30}  {t.tier:<12}  {t.elapsed_sec:>8.1f}  {t.router_elapsed_sec:>10.2f}")
    print(f"  Result: {result_text[:100]!r}")
    print(f"  Accurate (contains {EXPECTED_KEYWORDS}): {accurate}")

    return benchmark


# ---------------------------------------------------------------------------
# Comparison table
# ---------------------------------------------------------------------------

def print_comparison_table(results: List[BenchmarkResult]) -> None:
    print("\n")
    print("=" * 90)
    print("  BENCHMARK RESULTS — Wikipedia multi-step navigation")
    print(f"  Task: {WIKIPEDIA_TASK[:70]}...")
    print("=" * 90)

    header = (
        f"  {'ID':<3}  {'Label':<28}  {'OK':>4}  {'Acc':>4}  "
        f"{'Iters':>5}  {'Total(s)':>9}  {'Avg/iter(s)':>12}  {'Result (50 chars)'}"
    )
    print(header)
    print("  " + "-" * 86)

    for r in results:
        ok = "YES" if r.success else "NO "
        acc = "YES" if r.accurate else "NO "
        total = f"{r.total_sec:.0f}s"
        avg = f"{r.avg_iter_sec():.1f}s" if r.iteration_timings else "  -"
        snippet = (r.result_text[:50] + "...") if len(r.result_text) > 50 else r.result_text
        print(
            f"  {r.config_id:<3}  {r.config_label:<28}  {ok:>4}  {acc:>4}  "
            f"{r.iterations:>5}  {total:>9}  {avg:>12}  {snippet!r}"
        )

    print("  " + "-" * 86)
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Benchmark browser navigation models")
    parser.add_argument(
        "--configs", nargs="+", choices=list(CONFIGS.keys()), default=list(CONFIGS.keys()),
        help="Which configs to run (default: all)"
    )
    parser.add_argument(
        "--max-iterations", type=int, default=8,
        help="Max browser agent iterations per config"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Save results to JSON file (default: benchmark_results_<timestamp>.json)"
    )
    args = parser.parse_args()

    # Check Ollama
    try:
        import requests
        r = requests.get("http://localhost:11434/api/tags", timeout=3)
        r.raise_for_status()
        installed = {m["name"] for m in r.json().get("models", [])}
        print(f"Ollama models installed: {', '.join(sorted(installed))}\n")
    except Exception as e:
        print(f"ERROR: Ollama not reachable: {e}")
        print("Start Ollama with: ollama serve")
        sys.exit(1)

    selected = [CONFIGS[k] for k in args.configs]
    print(f"Running {len(selected)} config(s): {', '.join(c.id for c in selected)}")
    print(f"Task: {WIKIPEDIA_TASK}\n")

    results: List[BenchmarkResult] = []
    for cfg in selected:
        result = run_config(cfg, WIKIPEDIA_TASK, max_iterations=args.max_iterations)
        results.append(result)

    print_comparison_table(results)

    # Save JSON
    output_file = args.output or f"benchmark_results_{int(time.time())}.json"
    output_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        output_file,
    )
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(
            [
                {
                    "config_id": r.config_id,
                    "config_label": r.config_label,
                    "model_description": r.model_description,
                    "success": r.success,
                    "accurate": r.accurate,
                    "total_sec": round(r.total_sec, 1),
                    "iterations": r.iterations,
                    "avg_iter_sec": round(r.avg_iter_sec(), 1),
                    "result_text": r.result_text,
                    "error": r.error,
                    "iteration_timings": [
                        {
                            "iteration": t.iteration,
                            "model": t.model,
                            "tier": t.tier,
                            "elapsed_sec": round(t.elapsed_sec, 2),
                            "router_elapsed_sec": round(t.router_elapsed_sec, 2),
                        }
                        for t in r.iteration_timings
                    ],
                }
                for r in results
            ],
            f, indent=2,
        )
    print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()
