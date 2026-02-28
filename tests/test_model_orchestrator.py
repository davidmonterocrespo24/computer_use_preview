"""
Unit tests for the ModelOrchestrator — no real LLM calls.
"""
import pytest
from unittest.mock import MagicMock, patch

from src.llm.orchestrator import (
    ModelOrchestrator,
    ModelTier,
    SelectionContext,
)


def make_ctx(**kwargs) -> SelectionContext:
    """Helper to build a SelectionContext with sensible defaults."""
    defaults = dict(
        query="test query",
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


class TestModelOrchestratorTierSelection:

    def test_balanced_is_default(self):
        orch = ModelOrchestrator()
        ctx = make_ctx()
        tier, reason = orch.select_tier(ctx)
        assert tier == ModelTier.BALANCED

    def test_ultra_fast_for_scroll_after_first_iter(self):
        orch = ModelOrchestrator()
        ctx = make_ctx(last_action_type="scroll", iteration_number=2)
        tier, reason = orch.select_tier(ctx)
        assert tier == ModelTier.ULTRA_FAST
        assert "scroll" in reason

    def test_ultra_fast_for_wait_after_first_iter(self):
        orch = ModelOrchestrator()
        ctx = make_ctx(last_action_type="wait", iteration_number=3)
        tier, reason = orch.select_tier(ctx)
        assert tier == ModelTier.ULTRA_FAST

    def test_ultra_fast_not_on_first_iteration(self):
        """Even simple actions use BALANCED on iteration 1 (no history yet)."""
        orch = ModelOrchestrator()
        ctx = make_ctx(last_action_type="scroll", iteration_number=1)
        tier, _ = orch.select_tier(ctx)
        assert tier == ModelTier.BALANCED

    def test_full_vis_when_stuck(self):
        orch = ModelOrchestrator()
        ctx = make_ctx(stuck_count=3)
        tier, reason = orch.select_tier(ctx)
        assert tier == ModelTier.FULL_VIS
        assert "stuck" in reason

    def test_full_vis_when_dom_sparse_after_first_iter(self):
        orch = ModelOrchestrator()
        ctx = make_ctx(dom_element_count=3, iteration_number=2)
        tier, reason = orch.select_tier(ctx)
        assert tier == ModelTier.FULL_VIS
        assert "sparse" in reason

    def test_sparse_dom_ignored_on_first_iteration(self):
        """On first iteration DOM may be legitimately sparse — don't jump to vision."""
        orch = ModelOrchestrator()
        ctx = make_ctx(dom_element_count=0, iteration_number=1)
        tier, _ = orch.select_tier(ctx)
        assert tier != ModelTier.FULL_VIS

    def test_quick_vis_for_canvas_page(self):
        orch = ModelOrchestrator()
        ctx = make_ctx(page_has_canvas=True)
        tier, reason = orch.select_tier(ctx)
        assert tier == ModelTier.QUICK_VIS
        assert "canvas" in reason

    def test_quick_vis_for_iframe_page(self):
        orch = ModelOrchestrator()
        ctx = make_ctx(page_has_iframes=True)
        tier, reason = orch.select_tier(ctx)
        assert tier == ModelTier.QUICK_VIS

    def test_balanced_for_forms(self):
        orch = ModelOrchestrator()
        ctx = make_ctx(page_has_forms=True)
        tier, reason = orch.select_tier(ctx)
        assert tier == ModelTier.BALANCED
        assert "form" in reason

    def test_balanced_for_login_url(self):
        orch = ModelOrchestrator()
        ctx = make_ctx(current_url="https://example.com/login")
        tier, reason = orch.select_tier(ctx)
        assert tier == ModelTier.BALANCED

    def test_balanced_for_checkout_url(self):
        orch = ModelOrchestrator()
        ctx = make_ctx(current_url="https://shop.example.com/checkout/step2")
        tier, reason = orch.select_tier(ctx)
        assert tier == ModelTier.BALANCED

    def test_stuck_overrides_canvas(self):
        """stuck_count threshold takes higher priority than canvas."""
        orch = ModelOrchestrator()
        ctx = make_ctx(stuck_count=5, page_has_canvas=True)
        tier, _ = orch.select_tier(ctx)
        assert tier == ModelTier.FULL_VIS

    def test_custom_threshold_from_config(self):
        config = {"dom_sparse_threshold": 10, "stuck_iteration_threshold": 2}
        orch = ModelOrchestrator(config=config)
        # stuck_count=2 should now trigger FULL_VIS
        ctx = make_ctx(stuck_count=2)
        tier, _ = orch.select_tier(ctx)
        assert tier == ModelTier.FULL_VIS


class TestModelOrchestratorClientCache:

    def test_get_client_returns_llm_client(self):
        orch = ModelOrchestrator()
        from src.llm.client import LLMClient
        client = orch.get_client(ModelTier.BALANCED)
        assert isinstance(client, LLMClient)

    def test_same_client_instance_returned_for_same_tier(self):
        orch = ModelOrchestrator()
        client1 = orch.get_client(ModelTier.BALANCED)
        client2 = orch.get_client(ModelTier.BALANCED)
        assert client1 is client2

    def test_different_clients_for_different_tiers(self):
        orch = ModelOrchestrator()
        client_fast = orch.get_client(ModelTier.ULTRA_FAST)
        client_balanced = orch.get_client(ModelTier.BALANCED)
        assert client_fast is not client_balanced

    def test_client_uses_correct_model_for_tier(self):
        orch = ModelOrchestrator()
        client = orch.get_client(ModelTier.ULTRA_FAST)
        assert client.model == "lfm2.5-thinking:latest"

    def test_client_uses_correct_model_balanced(self):
        orch = ModelOrchestrator()
        client = orch.get_client(ModelTier.BALANCED)
        assert client.model == "ministral-3:3b"

    def test_client_uses_correct_model_full_vis(self):
        orch = ModelOrchestrator()
        client = orch.get_client(ModelTier.FULL_VIS)
        assert client.model == "qwen3-vl:4b"

    def test_config_overrides_model(self):
        config = {
            "tiers": {
                "balanced": {"model": "my-custom-model:latest", "base_url": "http://localhost:11434/v1"}
            }
        }
        orch = ModelOrchestrator(config=config)
        client = orch.get_client(ModelTier.BALANCED)
        assert client.model == "my-custom-model:latest"


class TestModelOrchestratorSelectionLog:

    def test_selection_log_is_empty_initially(self):
        orch = ModelOrchestrator()
        assert orch.get_selection_log() == []

    def test_selection_log_records_after_generate(self):
        orch = ModelOrchestrator()
        ctx = make_ctx(iteration_number=1)

        mock_client = MagicMock()
        mock_client.generate_with_retry.return_value = MagicMock(content="ok", actions=None, finish_reason="stop")

        with patch.object(orch, "get_client", return_value=mock_client):
            orch.generate(ctx, [{"role": "user", "content": "hi"}], "sys", [])

        log = orch.get_selection_log()
        assert len(log) == 1
        assert log[0]["iteration"] == 1
        assert "tier" in log[0]
        assert "model" in log[0]
        assert "reason" in log[0]

    def test_selection_log_accumulates(self):
        orch = ModelOrchestrator()
        mock_client = MagicMock()
        mock_client.generate_with_retry.return_value = MagicMock(content="ok", actions=None, finish_reason="stop")

        with patch.object(orch, "get_client", return_value=mock_client):
            for i in range(3):
                ctx = make_ctx(iteration_number=i + 1)
                orch.generate(ctx, [], "sys", [])

        assert len(orch.get_selection_log()) == 3

    def test_selection_log_serializable(self):
        """Log entries must be JSON-serializable (no enum values, no custom objects)."""
        import json
        orch = ModelOrchestrator()
        ctx = make_ctx()
        mock_client = MagicMock()
        mock_client.generate_with_retry.return_value = MagicMock(content="ok", actions=None, finish_reason="stop")

        with patch.object(orch, "get_client", return_value=mock_client):
            orch.generate(ctx, [], "sys", [])

        log = orch.get_selection_log()
        json_str = json.dumps(log)  # should not raise
        assert len(json_str) > 0


class TestModelOrchestratorGenerate:

    def test_generate_calls_text_client_for_balanced(self):
        orch = ModelOrchestrator()
        ctx = make_ctx()  # default → BALANCED

        mock_client = MagicMock()
        expected = MagicMock(content="response", actions=None, finish_reason="stop")
        mock_client.generate_with_retry.return_value = expected

        with patch.object(orch, "get_client", return_value=mock_client):
            result = orch.generate(ctx, [], "sys", [])

        mock_client.generate_with_retry.assert_called_once()
        assert result is expected

    def test_generate_uses_vision_when_image_and_vision_tier(self):
        """When image_b64 is provided and tier supports vision, use generate_with_image."""
        orch = ModelOrchestrator()
        ctx = make_ctx(stuck_count=3)  # → FULL_VIS (supports vision)

        mock_client = MagicMock()
        expected = MagicMock(content="vision response", actions=None, finish_reason="stop")
        mock_client.generate_with_image.return_value = expected

        with patch.object(orch, "get_client", return_value=mock_client):
            result = orch.generate(ctx, [], "sys", [], image_b64="fakebase64")

        mock_client.generate_with_image.assert_called_once()
        mock_client.generate_with_retry.assert_not_called()
        assert result is expected

    def test_generate_skips_vision_when_no_image(self):
        """Even with a vision tier, if no image provided, use text generation."""
        orch = ModelOrchestrator()
        ctx = make_ctx(stuck_count=3)  # → FULL_VIS

        mock_client = MagicMock()
        expected = MagicMock(content="text response", actions=None, finish_reason="stop")
        mock_client.generate_with_retry.return_value = expected

        with patch.object(orch, "get_client", return_value=mock_client):
            result = orch.generate(ctx, [], "sys", [], image_b64=None)

        mock_client.generate_with_retry.assert_called_once()
        mock_client.generate_with_image.assert_not_called()

    def test_get_tier_config_returns_dict(self):
        orch = ModelOrchestrator()
        cfg = orch.get_tier_config(ModelTier.BALANCED)
        assert isinstance(cfg, dict)
        assert "model" in cfg
        assert "supports_vision" in cfg
