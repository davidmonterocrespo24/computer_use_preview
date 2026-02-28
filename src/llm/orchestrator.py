"""
Model Orchestrator - Selects the appropriate Ollama model based on task context.
Dynamically routes requests to the lightest or most capable model.
Supports LLM-based routing via a tiny model (1-2B) as an alternative to rule-based selection.
"""
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple
import logging
import re
import time

from .client import LLMClient, LLMResponse, create_browser_tools

log = logging.getLogger(__name__)


OLLAMA_BASE_URL = "http://localhost:11434/v1"

# URL patterns that suggest complex pages (login, checkout, etc.)
COMPLEX_URL_PATTERNS = re.compile(
    r"(login|signin|sign-in|checkout|payment|register|signup|sign-up|auth|account)",
    re.IGNORECASE
)

# Simple/repetitive actions that don't need a powerful model
SIMPLE_ACTIONS = {"scroll", "wait", "go_back"}


class ModelTier(Enum):
    ULTRA_FAST = 1   # lfm2.5-thinking:latest  — repetitive/simple actions
    BALANCED   = 2   # ministral-3:3b           — general navigation (default)
    QUICK_VIS  = 3   # granite3.2-vision:latest — fast visual verification
    FULL_VIS   = 4   # qwen3-vl:4b              — complex visual reasoning


# Default tier configurations (can be overridden by config.yaml)
DEFAULT_TIER_CONFIGS: Dict[ModelTier, dict] = {
    ModelTier.ULTRA_FAST: {
        "model": "lfm2.5-thinking:latest",
        "base_url": OLLAMA_BASE_URL,
        "supports_vision": False,
        "max_tokens": 1024,
        "temperature": 0.3,
    },
    ModelTier.BALANCED: {
        "model": "ministral-3:3b",
        "base_url": OLLAMA_BASE_URL,
        "supports_vision": False,
        "max_tokens": 2048,
        "temperature": 0.5,
    },
    ModelTier.QUICK_VIS: {
        "model": "granite3.2-vision:latest",
        "base_url": OLLAMA_BASE_URL,
        "supports_vision": True,
        "max_tokens": 2048,
        "temperature": 0.5,
    },
    ModelTier.FULL_VIS: {
        "model": "qwen3-vl:4b",
        "base_url": OLLAMA_BASE_URL,
        "supports_vision": True,
        "max_tokens": 4096,
        "temperature": 0.7,
    },
}


@dataclass
class SelectionContext:
    """All observable signals available at model-selection time."""
    query: str
    dom_element_count: int        # < dom_sparse_threshold → consider vision
    iteration_number: int
    current_url: str
    stuck_count: int              # consecutive iterations at same URL
    page_has_canvas: bool = False
    page_has_iframes: bool = False
    page_has_forms: bool = False
    last_action_type: str = ""    # "navigate"|"click"|"type_text"|"scroll"|etc.
    screenshot_b64: Optional[str] = None


@dataclass
class SelectionRecord:
    """Log entry for a single model selection decision."""
    iteration: int
    tier: ModelTier
    model: str
    reason: str
    dom_element_count: int
    stuck_count: int
    url: str
    elapsed_sec: float = 0.0       # wall-clock time for the LLM call
    router_elapsed_sec: float = 0.0  # time spent by the LLM router (0 if rule-based)


class LLMRouter:
    """
    Tiny LLM (1-2B parameters) that classifies which model tier to use.
    Replaces the hand-crafted rule-based select_tier() with a learned routing decision.
    Falls back to rule-based selection on timeout or parse error.
    """

    _TIER_NAMES = {t.name: t for t in ModelTier}

    PROMPT_TEMPLATE = (
        "You are a routing agent for a browser automation system.\n"
        "Choose the best model tier for the current browser state.\n\n"
        "State:\n"
        "  URL: {url}\n"
        "  DOM elements visible: {dom_count}\n"
        "  Stuck iterations (same URL): {stuck_count}\n"
        "  Last browser action: {last_action}\n"
        "  Page has forms: {has_forms}\n"
        "  Page has canvas or iframes: {has_canvas_iframe}\n"
        "  Iteration number: {iteration}\n\n"
        "Available tiers (choose exactly one):\n"
        "  ULTRA_FAST  - repetitive/simple actions: scroll, wait, go_back\n"
        "  BALANCED    - general navigation, link clicking, reading text\n"
        "  QUICK_VIS   - pages with canvas elements or iframes\n"
        "  FULL_VIS    - stuck agent, very sparse DOM, complex visual analysis\n\n"
        "Reply with EXACTLY one word — the tier name. No explanation.\n"
        "Tier:"
    )

    def __init__(self, client: LLMClient, timeout_sec: float = 8.0):
        self._client = client
        self._timeout = timeout_sec

    def route(self, ctx: "SelectionContext") -> Tuple[ModelTier, str, float]:
        """
        Ask the tiny router model which tier to use.

        Returns:
            (ModelTier, reason_string, elapsed_sec)
            Falls back to (BALANCED, "router_fallback", elapsed) on error.
        """
        prompt = self.PROMPT_TEMPLATE.format(
            url=ctx.current_url,
            dom_count=ctx.dom_element_count,
            stuck_count=ctx.stuck_count,
            last_action=ctx.last_action_type or "none",
            has_forms=ctx.page_has_forms,
            has_canvas_iframe=ctx.page_has_canvas or ctx.page_has_iframes,
            iteration=ctx.iteration_number,
        )
        t0 = time.perf_counter()
        try:
            response = self._client.generate(
                messages=[{"role": "user", "content": prompt}],
                system_prompt=None,
                tools=None,
            )
            elapsed = time.perf_counter() - t0
            raw = response.content.strip().upper().split()[0] if response.content.strip() else ""
            # Strip punctuation
            raw = raw.strip(".:,;!?")
            tier = self._TIER_NAMES.get(raw)
            if tier is not None:
                reason = f"llm_routed:{self._client.model}→{raw}"
                log.debug("LLM router | raw=%r → tier=%s | elapsed=%.2fs", raw, tier.name, elapsed)
                return tier, reason, elapsed
            # Unrecognised output — fall back
            log.warning("LLM router unrecognised output %r, falling back to BALANCED", raw)
            return ModelTier.BALANCED, f"router_bad_output({raw})", elapsed
        except Exception as exc:
            elapsed = time.perf_counter() - t0
            log.warning("LLM router error (%.2fs): %s — falling back to BALANCED", elapsed, exc)
            return ModelTier.BALANCED, f"router_error:{exc}", elapsed


class ModelOrchestrator:
    """
    Selects and manages the appropriate LLM model for each browser agent iteration.
    Routes between text-only and vision-capable models based on context signals.
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        dom_sparse_threshold: int = 5,
        stuck_iteration_threshold: int = 3,
        router: Optional[LLMRouter] = None,
    ):
        """
        Args:
            config: Optional config dict (from config.yaml orchestrator section)
            dom_sparse_threshold: element count below which vision is considered
            stuck_iteration_threshold: same URL for N iters → escalate to vision
            router: Optional LLMRouter that replaces rule-based tier selection
        """
        self._tier_configs = dict(DEFAULT_TIER_CONFIGS)
        self.dom_sparse_threshold = dom_sparse_threshold
        self.stuck_iteration_threshold = stuck_iteration_threshold
        self._client_cache: Dict[ModelTier, LLMClient] = {}
        self._selection_log: List[SelectionRecord] = []
        self._router: Optional[LLMRouter] = router

        if config:
            self._apply_config(config)

    def _apply_config(self, config: Dict[str, Any]) -> None:
        """Override default tier configs from config.yaml orchestrator section."""
        self.dom_sparse_threshold = config.get("dom_sparse_threshold", self.dom_sparse_threshold)
        self.stuck_iteration_threshold = config.get("stuck_iteration_threshold", self.stuck_iteration_threshold)

        tiers_config = config.get("tiers", {})
        tier_map = {
            "ultra_fast": ModelTier.ULTRA_FAST,
            "balanced": ModelTier.BALANCED,
            "quick_vision": ModelTier.QUICK_VIS,
            "full_vision": ModelTier.FULL_VIS,
        }
        for key, tier in tier_map.items():
            if key in tiers_config:
                self._tier_configs[tier].update(tiers_config[key])
        # Clear cache so new configs take effect
        self._client_cache.clear()

        # Configure LLM router if specified in config
        router_cfg = config.get("router", {})
        if router_cfg.get("enabled", False) and self._router is None:
            router_client = LLMClient(
                provider="local",
                model=router_cfg.get("model", "gemma3:1b"),
                base_url=router_cfg.get("base_url", OLLAMA_BASE_URL),
                temperature=0.1,
                max_tokens=16,
            )
            self._router = LLMRouter(
                client=router_client,
                timeout_sec=router_cfg.get("timeout_sec", 8.0),
            )
            log.info("LLM router enabled | model=%s", router_cfg.get("model", "gemma3:1b"))

        log.info(
            "Orchestrator config loaded | dom_sparse_threshold=%d | stuck_threshold=%d | router=%s",
            self.dom_sparse_threshold,
            self.stuck_iteration_threshold,
            "enabled" if self._router else "disabled (rule-based)",
        )

    def select_tier(self, ctx: SelectionContext) -> Tuple[ModelTier, str]:
        """
        Select the appropriate model tier based on context signals.
        Uses the LLM router if configured, otherwise falls back to rule-based logic.

        Returns:
            Tuple of (ModelTier, reason string)
        """
        if self._router is not None:
            tier, reason, _ = self._router.route(ctx)
        else:
            tier, reason = self._select_tier_rules(ctx)

        model = self._tier_configs[tier]["model"]
        log.info(
            "[iter=%d] Model selected | tier=%-10s | model=%-30s | reason=%s | dom_elems=%d | stuck=%d",
            ctx.iteration_number,
            tier.name,
            model,
            reason,
            ctx.dom_element_count,
            ctx.stuck_count,
        )
        return tier, reason

    def _select_tier_rules(self, ctx: SelectionContext) -> Tuple[ModelTier, str]:
        """Rule-based tier selection (original logic)."""
        # Priority 1: Agent is stuck or DOM is very sparse → vision required
        if ctx.stuck_count >= self.stuck_iteration_threshold:
            return ModelTier.FULL_VIS, f"stuck for {ctx.stuck_count} iterations"
        elif ctx.dom_element_count < self.dom_sparse_threshold and ctx.iteration_number > 1:
            return ModelTier.FULL_VIS, f"sparse DOM ({ctx.dom_element_count} elements)"
        # Priority 2: Canvas/iframe pages are visually complex
        elif ctx.page_has_canvas or ctx.page_has_iframes:
            return ModelTier.QUICK_VIS, "canvas or iframe detected"
        # Priority 3: Simple/repetitive action on non-first iteration
        elif ctx.last_action_type in SIMPLE_ACTIONS and ctx.iteration_number > 1:
            return ModelTier.ULTRA_FAST, f"repetitive action: {ctx.last_action_type}"
        # Priority 4: Complex pages (auth, checkout, forms)
        elif ctx.page_has_forms or COMPLEX_URL_PATTERNS.search(ctx.current_url):
            return ModelTier.BALANCED, "form or auth/checkout page"
        else:
            return ModelTier.BALANCED, "default balanced"

    def get_client(self, tier: ModelTier) -> LLMClient:
        """Get or create a cached LLMClient for the given tier."""
        if tier not in self._client_cache:
            cfg = self._tier_configs[tier]
            log.debug("Creating new LLMClient for tier=%s model=%s", tier.name, cfg["model"])
            self._client_cache[tier] = LLMClient(
                provider="local",
                model=cfg["model"],
                base_url=cfg["base_url"],
                temperature=cfg["temperature"],
                max_tokens=cfg["max_tokens"],
            )
        return self._client_cache[tier]

    def generate(
        self,
        ctx: SelectionContext,
        messages: List[Dict[str, str]],
        system_prompt: str,
        tools: List[Dict],
        image_b64: Optional[str] = None,
        _preselected_tier: Optional[tuple] = None,
    ) -> LLMResponse:
        """
        Select the right model and generate a response.

        If image_b64 is provided and the selected tier supports vision,
        the annotated screenshot is sent alongside the messages.
        """
        router_elapsed = 0.0
        if _preselected_tier is not None:
            tier, reason = _preselected_tier
        elif self._router is not None:
            tier, reason, router_elapsed = self._router.route(ctx)
            model = self._tier_configs[tier]["model"]
            log.info(
                "[iter=%d] Model selected | tier=%-10s | model=%-30s | reason=%s | dom_elems=%d | stuck=%d",
                ctx.iteration_number, tier.name, model, reason,
                ctx.dom_element_count, ctx.stuck_count,
            )
        else:
            tier, reason = self.select_tier(ctx)
        cfg = self._tier_configs[tier]
        client = self.get_client(tier)

        # Generate response and measure elapsed time
        if image_b64 and cfg.get("supports_vision"):
            log.info("[iter=%d] Calling vision model %s with annotated screenshot", ctx.iteration_number, cfg["model"])
            response = client.generate_with_image(messages, system_prompt, image_b64)
        else:
            log.debug("[iter=%d] Calling text model %s", ctx.iteration_number, cfg["model"])
            response = client.generate_with_retry(messages, system_prompt, tools)

        # Record selection + timing decision
        raw_elapsed = response.elapsed_sec
        elapsed_sec = float(raw_elapsed) if isinstance(raw_elapsed, (int, float)) else 0.0
        self._selection_log.append(SelectionRecord(
            iteration=ctx.iteration_number,
            tier=tier,
            model=cfg["model"],
            reason=reason,
            dom_element_count=ctx.dom_element_count,
            stuck_count=ctx.stuck_count,
            url=ctx.current_url,
            elapsed_sec=elapsed_sec,
            router_elapsed_sec=router_elapsed,
        ))

        return response

    def get_selection_log(self) -> List[Dict[str, Any]]:
        """Return the model selection log as a list of dicts (for serialization)."""
        return [
            {
                "iteration": r.iteration,
                "tier": r.tier.name,
                "model": r.model,
                "reason": r.reason,
                "dom_element_count": r.dom_element_count,
                "stuck_count": r.stuck_count,
                "url": r.url,
                "elapsed_sec": round(r.elapsed_sec, 2),
                "router_elapsed_sec": round(r.router_elapsed_sec, 2),
            }
            for r in self._selection_log
        ]

    def get_tier_config(self, tier: ModelTier) -> Dict[str, Any]:
        """Return the configuration for a given tier (for testing/inspection)."""
        return dict(self._tier_configs[tier])
