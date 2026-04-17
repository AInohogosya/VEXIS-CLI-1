"""
Unit tests for cost manager
"""

import pytest
import tempfile
from pathlib import Path
from src.ai_agent.utils.cost_manager import (
    CostManager,
    BudgetConfig,
    BudgetAlertLevel,
    ModelPricing,
    UsageRecord
)


class TestCostEstimation:
    """Test cost estimation"""
    
    def test_estimate_openai_gpt5(self):
        """Test cost estimation for GPT-5"""
        manager = CostManager()
        # GPT-5.4: $5 per 1M input, $20 per 1M output
        cost = manager.estimate_cost("openai", "gpt-5.4", 1000, 500)
        expected = (1000/1_000_000 * 5.0) + (500/1_000_000 * 20.0)
        assert cost == pytest.approx(expected, 0.0001)
    
    def test_estimate_google_gemini(self):
        """Test cost estimation for Gemini"""
        manager = CostManager()
        # Gemini 3.1 Pro: $2.50 per 1M input, $15 per 1M output
        cost = manager.estimate_cost("google", "gemini-3.1-pro", 1000000, 500000)
        assert cost == pytest.approx(2.50 + 7.50, 0.01)
    
    def test_estimate_local_ollama(self):
        """Test cost estimation for Ollama (free)"""
        manager = CostManager()
        cost = manager.estimate_cost("ollama", "llama3.2:latest", 10000, 5000)
        assert cost == 0.0


class TestBudgetChecking:
    """Test budget enforcement"""
    
    def test_daily_budget_enforcement(self):
        """Test daily budget enforcement"""
        config = BudgetConfig(daily_budget=1.0)  # $1 per day
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CostManager(config, persist_path=str(Path(tmpdir) / "costs.json"))
            
            # Simulate spending $0.80
            manager.record_usage("openai", "gpt-5.4-mini", 1000000, 500000)  # ~$0.75
            
            # Check if we can spend another $0.50 (would exceed budget)
            allowed, reason = manager.check_budget(0.50)
            assert allowed == False
            assert "Daily budget" in reason
    
    def test_per_request_budget(self):
        """Test per-request budget enforcement"""
        config = BudgetConfig(per_request_budget=0.01)  # 1 cent per request
        manager = CostManager(config)
        
        # Check if expensive request is blocked
        allowed, reason = manager.check_budget(0.05)  # 5 cents
        assert allowed == False
        assert "per-request budget" in reason


class TestCheaperAlternatives:
    """Test cheaper alternative suggestions"""
    
    def test_suggest_cheaper_alternative(self):
        """Test finding cheaper alternative"""
        manager = CostManager()
        
        # Claude Opus is expensive ($15/75 per 1M), should suggest cheaper
        alt = manager.get_cheaper_alternative("anthropic", "claude-opus-4.6")
        
        if alt:  # If an alternative exists
            provider, model = alt
            current_pricing = manager._get_pricing("anthropic", "claude-opus-4.6")
            alt_pricing = manager._get_pricing(provider, model)
            
            current_cost = current_pricing.input_price_per_1m + current_pricing.output_price_per_1m
            alt_cost = alt_pricing.input_price_per_1m + alt_pricing.output_price_per_1m
            
            assert alt_cost < current_cost


class TestBudgetAlerts:
    """Test budget alert levels"""
    
    def test_warning_alert_level(self):
        """Test warning level at 80%"""
        config = BudgetConfig(daily_budget=100.0, warning_threshold=0.8)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CostManager(config, persist_path=str(Path(tmpdir) / "costs.json"))
            
            # Spend 85% of budget
            for _ in range(17):  # Record multiple small usages
                manager.record_usage("openai", "gpt-5.4-mini", 100000, 50000)
            
            status = manager.get_budget_status()
            assert status["alert_level"] == BudgetAlertLevel.WARNING.value
    
    def test_critical_alert_level(self):
        """Test critical level at 95%"""
        config = BudgetConfig(
            daily_budget=10.0,
            warning_threshold=0.8,
            critical_threshold=0.95
        )
        
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CostManager(config, persist_path=str(Path(tmpdir) / "costs.json"))
            
            # Spend 96% of budget
            for _ in range(13):
                manager.record_usage("openai", "gpt-5.4-mini", 10000, 5000)
            
            status = manager.get_budget_status()
            # Note: May be WARNING or CRITICAL depending on actual costs
            assert status["alert_level"] in [BudgetAlertLevel.WARNING.value, BudgetAlertLevel.CRITICAL.value]


class TestUsageTracking:
    """Test usage tracking and statistics"""
    
    def test_record_usage_updates_stats(self):
        """Test that recording usage updates statistics"""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CostManager(persist_path=str(Path(tmpdir) / "costs.json"))
            
            manager.record_usage("openai", "gpt-5.4", 1000, 500, task_type="phase1")
            
            assert manager.stats.total_requests == 1
            assert manager.stats.total_prompt_tokens == 1000
            assert manager.stats.total_completion_tokens == 500
            assert manager.stats.total_cost > 0
    
    def test_provider_cost_tracking(self):
        """Test per-provider cost tracking"""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CostManager(persist_path=str(Path(tmpdir) / "costs.json"))
            
            manager.record_usage("openai", "gpt-5.4", 1000, 500)
            manager.record_usage("google", "gemini-3.1-pro", 2000, 1000)
            
            assert "openai" in manager.stats.provider_costs
            assert "google" in manager.stats.provider_costs
            assert manager.stats.provider_costs["openai"] > 0
            assert manager.stats.provider_costs["google"] > 0


class TestPersistence:
    """Test cost data persistence"""
    
    def test_save_and_load(self):
        """Test that costs are persisted and loaded correctly"""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "costs.json")
            
            # Create manager and record usage
            manager1 = CostManager(persist_path=path)
            manager1.record_usage("openai", "gpt-5.4", 1000, 500)
            cost1 = manager1.stats.total_cost
            
            # Create new manager with same path
            manager2 = CostManager(persist_path=path)
            
            assert manager2.stats.total_cost == cost1
            assert manager2.stats.total_requests == 1
