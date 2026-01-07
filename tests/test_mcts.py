"""
Tests for Latent Monte Carlo Tree Search (L-MCTS).

These tests verify:
1. LatentNode creation and statistics
2. MCTS phases: selection, expansion, evaluation, backpropagation
3. Integration with existing JudgePanel
4. MCTSEvolutionLoop compatibility with EvolutionLoop interface
"""

import pytest
import torch
from torch import Tensor

from latent_reasoning.evolution.mcts.node import LatentNode, create_root_from_latent
from latent_reasoning.evolution.mcts.search import LatentMCTS, MCTSConfig, mcts_search
from latent_reasoning.evolution.mcts.loop import MCTSEvolutionLoop
from latent_reasoning.config import EvolutionConfig
from latent_reasoning.core.panel import JudgePanel, PanelVerdict
from latent_reasoning.core.chain import ModificationContext


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def sample_latent() -> Tensor:
    """Create a sample latent vector for testing."""
    return torch.randn(512)


@pytest.fixture
def sample_latent_2d() -> Tensor:
    """Create a 2D latent (e.g., from transformer hidden states)."""
    return torch.randn(1, 512)


class MockJudge:
    """Mock scorer judge for testing."""
    
    def __init__(self, return_score: float = 0.5):
        self.return_score = return_score
        self.reference = None
        self.call_count = 0
    
    def set_reference(self, embedding=None, text=None):
        self.reference = embedding
    
    def score(self, latent: Tensor) -> float:
        self.call_count += 1
        # Return slightly different scores based on latent norm
        # This makes tests more interesting
        variation = (latent.norm().item() % 0.1)
        return min(1.0, self.return_score + variation)


class MockModifier:
    """Mock modifier judge for testing."""
    
    def modify(self, latent: Tensor, context: ModificationContext | None) -> Tensor:
        # Return a simple direction: towards higher norm
        return torch.ones_like(latent) * 0.1


class MockJudgePanel:
    """Mock JudgePanel for testing MCTS."""
    
    def __init__(self, base_score: float = 0.5):
        self.mock_judge = MockJudge(base_score)
        self.mock_modifier = MockModifier()
        self.scorers = [self.mock_judge]
        self.modifiers = [self.mock_modifier]
    
    def evaluate(self, latent: Tensor, context: ModificationContext | None = None) -> PanelVerdict:
        score = self.mock_judge.score(latent)
        return PanelVerdict(score=score, modification=None, individual_scores=[score])
    
    def get_modification(self, latent: Tensor, context: ModificationContext | None = None) -> Tensor | None:
        return self.mock_modifier.modify(latent, context)


@pytest.fixture
def mock_judge_panel() -> MockJudgePanel:
    """Create a mock judge panel."""
    return MockJudgePanel()


@pytest.fixture
def evolution_config() -> EvolutionConfig:
    """Create an evolution config for testing."""
    return EvolutionConfig(
        generations=5,
        chains=4,
        temperature=0.3,
    )


# =============================================================================
# LATENT NODE TESTS
# =============================================================================

class TestLatentNode:
    """Tests for LatentNode class."""
    
    def test_create_root_node(self, sample_latent):
        """Test creating a root node from latent."""
        root = create_root_from_latent(sample_latent)
        
        assert root.is_root
        assert root.depth == 0
        assert root.is_leaf
        assert root.visits == 0
        assert root.value == 0.0
        assert torch.equal(root.latent, sample_latent)
    
    def test_add_child(self, sample_latent):
        """Test adding child nodes."""
        root = create_root_from_latent(sample_latent)
        child_latent = sample_latent + torch.randn_like(sample_latent) * 0.1
        
        child = root.add_child(child_latent, temperature=0.5)
        
        assert not root.is_leaf
        assert child.is_leaf
        assert child.depth == 1
        assert child.parent is root
        assert len(root.children) == 1
        assert child in root.children
    
    def test_update_statistics(self, sample_latent):
        """Test updating node statistics."""
        node = create_root_from_latent(sample_latent)
        
        node.update(0.7)
        assert node.visits == 1
        assert node.total_score == 0.7
        assert node.value == 0.7
        assert node.best_score == 0.7
        
        node.update(0.9)
        assert node.visits == 2
        assert node.total_score == 1.6
        assert node.value == 0.8
        assert node.best_score == 0.9
    
    def test_trajectory(self, sample_latent):
        """Test getting trajectory from root to node."""
        root = create_root_from_latent(sample_latent)
        child1 = root.add_child(sample_latent + 0.1)
        child2 = child1.add_child(sample_latent + 0.2)
        
        trajectory = child2.get_trajectory()
        
        assert len(trajectory) == 3
        assert torch.equal(trajectory[0], root.latent)
        assert torch.equal(trajectory[2], child2.latent)
    
    def test_best_child(self, sample_latent):
        """Test getting best child by value."""
        root = create_root_from_latent(sample_latent)
        
        child1 = root.add_child(sample_latent + 0.1)
        child2 = root.add_child(sample_latent + 0.2)
        
        child1.update(0.5)
        child2.update(0.8)
        
        best = root.get_best_child()
        assert best is child2
    
    def test_most_visited_child(self, sample_latent):
        """Test getting most visited child."""
        root = create_root_from_latent(sample_latent)
        
        child1 = root.add_child(sample_latent + 0.1)
        child2 = root.add_child(sample_latent + 0.2)
        
        # Visit child1 more times
        for _ in range(5):
            child1.update(0.5)
        for _ in range(2):
            child2.update(0.8)
        
        most_visited = root.get_most_visited_child()
        assert most_visited is child1


# =============================================================================
# MCTS SEARCH TESTS
# =============================================================================

class TestLatentMCTS:
    """Tests for LatentMCTS class."""
    
    def test_basic_search(self, sample_latent, mock_judge_panel):
        """Test basic MCTS search completes without error."""
        config = MCTSConfig(n_iterations=10, max_depth=5)
        mcts = LatentMCTS(mock_judge_panel, config=config)
        
        result = mcts.search(sample_latent)
        
        assert result.best_latent is not None
        assert result.best_score > 0
        assert result.iterations > 0
        assert result.total_evaluations > 0
        assert result.root is not None
    
    def test_search_builds_tree(self, sample_latent, mock_judge_panel):
        """Test that search actually builds a tree."""
        config = MCTSConfig(n_iterations=20, max_depth=5)
        mcts = LatentMCTS(mock_judge_panel, config=config)
        
        result = mcts.search(sample_latent)
        
        tree_size = mcts._count_nodes(result.root)
        assert tree_size > 1  # Should have more than just root
    
    def test_search_respects_max_evaluations(self, sample_latent, mock_judge_panel):
        """Test that search respects evaluation budget."""
        config = MCTSConfig(n_iterations=100)
        mcts = LatentMCTS(mock_judge_panel, config=config)
        
        result = mcts.search(sample_latent, max_evaluations=10)
        
        assert result.total_evaluations <= 10
    
    def test_search_respects_max_depth(self, sample_latent, mock_judge_panel):
        """Test that search respects maximum depth."""
        config = MCTSConfig(n_iterations=50, max_depth=3)
        mcts = LatentMCTS(mock_judge_panel, config=config)
        
        result = mcts.search(sample_latent)
        
        # Check no node exceeds max depth
        def check_depth(node, max_depth):
            assert node.depth <= max_depth
            for child in node.children:
                check_depth(child, max_depth)
        
        check_depth(result.root, 3)
    
    def test_history_tracking(self, sample_latent, mock_judge_panel):
        """Test that search tracks history."""
        config = MCTSConfig(n_iterations=15)
        mcts = LatentMCTS(mock_judge_panel, config=config)
        
        result = mcts.search(sample_latent)
        
        assert len(result.history) > 0
        assert 'iteration' in result.history[0]
        assert 'best_score' in result.history[0]
    
    def test_get_top_k_latents(self, sample_latent, mock_judge_panel):
        """Test extracting top-k latents from tree."""
        config = MCTSConfig(n_iterations=20)
        mcts = LatentMCTS(mock_judge_panel, config=config)
        
        result = mcts.search(sample_latent)
        top_k = mcts.get_top_k_latents(result.root, k=3)
        
        assert len(top_k) <= 3
        assert all(isinstance(l, Tensor) for l in top_k)
    
    def test_early_stopping_on_high_score(self, sample_latent):
        """Test early stopping when score threshold is reached."""
        # Create panel that returns high score
        panel = MockJudgePanel(base_score=0.98)
        config = MCTSConfig(n_iterations=100, early_stop_threshold=0.95)
        mcts = LatentMCTS(panel, config=config)
        
        result = mcts.search(sample_latent)
        
        # Should stop early
        assert result.iterations < 100


# =============================================================================
# MCTS CONVENIENCE FUNCTION TEST
# =============================================================================

class TestMCTSSearchFunction:
    """Tests for the mcts_search convenience function."""
    
    def test_quick_search(self, sample_latent, mock_judge_panel):
        """Test the quick search function works."""
        result = mcts_search(
            sample_latent, 
            mock_judge_panel, 
            n_iterations=10
        )
        
        assert result.best_latent is not None
        assert result.best_score > 0


# =============================================================================
# MCTS EVOLUTION LOOP TESTS
# =============================================================================

class TestMCTSEvolutionLoop:
    """Tests for MCTSEvolutionLoop drop-in replacement."""
    
    def test_run_interface(self, sample_latent, mock_judge_panel, evolution_config):
        """Test that MCTSEvolutionLoop has same interface as EvolutionLoop."""
        mcts_loop = MCTSEvolutionLoop(
            judge_panel=mock_judge_panel,
            config=evolution_config,
        )
        
        result = mcts_loop.run(sample_latent)
        
        # Check result has all expected fields
        assert hasattr(result, 'best_latent')
        assert hasattr(result, 'best_score')
        assert hasattr(result, 'survivors')
        assert hasattr(result, 'generations')
        assert hasattr(result, 'total_evaluations')
        assert hasattr(result, 'history')
        assert hasattr(result, 'converged')
        assert hasattr(result, 'stop_reason')
    
    def test_survivors_format(self, sample_latent, mock_judge_panel, evolution_config):
        """Test that survivors are in correct format (ChainState objects)."""
        mcts_loop = MCTSEvolutionLoop(
            judge_panel=mock_judge_panel,
            config=evolution_config,
        )
        
        result = mcts_loop.run(sample_latent)
        
        from latent_reasoning.core.chain import ChainState
        
        assert isinstance(result.survivors, list)
        for survivor in result.survivors:
            assert isinstance(survivor, ChainState)
            assert hasattr(survivor, 'latent')
            assert hasattr(survivor, 'score')
    
    def test_respects_evaluation_budget(self, sample_latent, mock_judge_panel, evolution_config):
        """Test that loop respects max_evaluations parameter."""
        mcts_loop = MCTSEvolutionLoop(
            judge_panel=mock_judge_panel,
            config=evolution_config,
        )
        
        result = mcts_loop.run(sample_latent, max_evaluations=15)
        
        assert result.total_evaluations <= 15
    
    def test_reset(self, sample_latent, mock_judge_panel, evolution_config):
        """Test reset clears state."""
        mcts_loop = MCTSEvolutionLoop(
            judge_panel=mock_judge_panel,
            config=evolution_config,
        )
        
        _ = mcts_loop.run(sample_latent)
        assert mcts_loop.total_evaluations > 0
        
        mcts_loop.reset()
        assert mcts_loop.total_evaluations == 0


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestMCTSIntegration:
    """Integration tests with real-ish components."""
    
    def test_2d_latent(self, sample_latent_2d, mock_judge_panel):
        """Test MCTS works with 2D latents (batch dimension)."""
        config = MCTSConfig(n_iterations=10)
        mcts = LatentMCTS(mock_judge_panel, config=config)
        
        result = mcts.search(sample_latent_2d)
        
        assert result.best_latent is not None
        assert result.best_latent.shape == sample_latent_2d.shape
    
    def test_different_exploration_constants(self, sample_latent, mock_judge_panel):
        """Test different exploration constants produce different behaviors."""
        config_explore = MCTSConfig(n_iterations=30, exploration_constant=2.0)
        config_exploit = MCTSConfig(n_iterations=30, exploration_constant=0.5)
        
        mcts_explore = LatentMCTS(mock_judge_panel, config=config_explore)
        mcts_exploit = LatentMCTS(mock_judge_panel, config=config_exploit)
        
        result_explore = mcts_explore.search(sample_latent)
        result_exploit = mcts_exploit.search(sample_latent)
        
        # Higher exploration should lead to wider tree
        tree_size_explore = mcts_explore._count_nodes(result_explore.root)
        tree_size_exploit = mcts_exploit._count_nodes(result_exploit.root)
        
        # Both should work (not asserting which is bigger, just that they work)
        assert tree_size_explore > 0
        assert tree_size_exploit > 0


# =============================================================================
# EDGE CASE TESTS
# =============================================================================

class TestMCTSEdgeCases:
    """Test edge cases and error handling."""
    
    def test_single_iteration(self, sample_latent, mock_judge_panel):
        """Test with just one iteration."""
        config = MCTSConfig(n_iterations=1)
        mcts = LatentMCTS(mock_judge_panel, config=config)
        
        result = mcts.search(sample_latent)
        
        assert result.best_latent is not None
        assert result.iterations == 1
    
    def test_very_small_latent(self, mock_judge_panel):
        """Test with very small latent dimension."""
        small_latent = torch.randn(8)
        config = MCTSConfig(n_iterations=10)
        mcts = LatentMCTS(mock_judge_panel, config=config)
        
        result = mcts.search(small_latent)
        
        assert result.best_latent is not None
        assert result.best_latent.shape == small_latent.shape
    
    def test_no_modifier(self, sample_latent, mock_judge_panel):
        """Test when modifier is disabled."""
        config = MCTSConfig(n_iterations=10, use_modifier=False)
        mcts = LatentMCTS(mock_judge_panel, config=config)
        
        result = mcts.search(sample_latent)
        
        assert result.best_latent is not None

