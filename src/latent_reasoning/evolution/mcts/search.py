"""
Latent Monte Carlo Tree Search (L-MCTS) - Core Algorithm.

This module implements MCTS adapted for latent space exploration.
Instead of searching game states, we search the model's internal
representation space to find high-quality reasoning trajectories.

=============================================================================
WHAT IS MCTS?
=============================================================================

Monte Carlo Tree Search is a best-first search algorithm that builds a 
search tree incrementally. Originally famous for AlphaGo, it balances
exploring new options vs exploiting known good ones.

The key insight: instead of evaluating every possibility (impossible in 
large spaces), MCTS samples promising paths and uses statistics to guide
future exploration.

=============================================================================
THE FOUR PHASES OF MCTS
=============================================================================

Each iteration of MCTS has four phases:

1. SELECTION: Starting from root, use UCB to pick which child to visit
   - UCB balances exploitation (high value) vs exploration (low visits)
   - Keep selecting until we reach a leaf or unexpanded node
   
2. EXPANSION: Add a new child node to the selected node
   - In L-MCTS, we create a "mutation" of the parent latent
   - We use the ModifierJudge to suggest good mutation directions
   
3. EVALUATION: Score the new node
   - Use the JudgePanel to evaluate the latent quality
   - This is cheap because we DON'T decode - just score the latent
   
4. BACKPROPAGATION: Update statistics up the tree
   - The score propagates from child to root
   - Updates visit counts and total scores

=============================================================================
WHY MCTS FOR LATENT REASONING?
=============================================================================

From the article:
> "Random perturbation means most of your candidates are useless. The 
> evolutionary process is robust enough to survive this, but it's wasteful."

MCTS fixes this by:
1. Remembering which directions were promising (visit statistics)
2. Focusing search on high-value regions (UCB exploitation)
3. Still exploring new areas (UCB exploration term)
4. Building structured paths, not random jumps

=============================================================================
UCB (Upper Confidence Bound)
=============================================================================

The magic formula that makes MCTS work:

    UCB(node) = value(node) + c * sqrt(ln(parent.visits) / node.visits)
    
- First term: exploitation (pick nodes with high observed value)
- Second term: exploration (pick nodes we haven't tried much)
- c: exploration constant (higher = more exploration)

When c is high, we explore more. When c is low, we exploit more.
The sqrt(ln/n) term ensures even rarely-visited nodes get a chance.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Callable

import torch
from torch import Tensor

from latent_reasoning.evolution.mcts.node import LatentNode, create_root_from_latent
from latent_reasoning.core.panel import JudgePanel, PanelVerdict
from latent_reasoning.core.chain import ModificationContext, compute_cross_chain_summary


@dataclass
class MCTSConfig:
    """
    Configuration for L-MCTS search.
    
    These hyperparameters control the behavior of the search.
    Start with defaults and tune based on your use case.
    
    Attributes:
        n_iterations: Number of MCTS iterations to run
        exploration_constant: UCB exploration weight (c in the formula)
        max_depth: Maximum tree depth
        temperature_decay: How fast to reduce mutation temperature with depth
        min_temperature: Minimum mutation temperature
        use_modifier: Whether to use ModifierJudge for directed mutations
        
    Tuning Guide:
        - More iterations = better results but slower
        - Higher exploration_constant = more diverse search
        - Lower exploration_constant = faster convergence to local optima
        - Typical exploration_constant: 1.0 to 2.0
    """
    n_iterations: int = 100
    exploration_constant: float = 1.41  # sqrt(2) is a common default
    max_depth: int = 10
    temperature_decay: float = 0.85
    min_temperature: float = 0.1
    use_modifier: bool = True
    
    # Early stopping
    early_stop_threshold: float = 0.95
    early_stop_patience: int = 20


@dataclass
class MCTSResult:
    """
    Result from L-MCTS search.
    
    Contains the best latent found, along with statistics and 
    alternative candidates for analysis.
    
    Attributes:
        best_latent: The highest-value latent found
        best_score: Score of the best latent
        best_trajectory: Path from root to best node
        root: The root node (access full tree structure)
        iterations: How many iterations ran
        total_evaluations: Number of judge evaluations performed
    """
    best_latent: Tensor
    best_score: float
    best_trajectory: List[Tensor]
    root: LatentNode
    iterations: int
    total_evaluations: int
    history: List[dict] = field(default_factory=list)


class LatentMCTS:
    """
    Monte Carlo Tree Search in Latent Space.
    
    This is the main class you'll interact with. It takes an encoded
    query latent and searches for better latent representations using
    tree-structured exploration.
    
    How it works:
    1. You provide a seed latent (from encoder.encode())
    2. MCTS builds a tree of variations around it
    3. Each node is scored by your JudgePanel (cheap, no decoding)
    4. UCB selection guides search toward promising regions
    5. You get back the best latent(s) found
    
    Example:
        >>> # Setup
        >>> from latent_reasoning import Engine
        >>> engine = Engine(encoder="Qwen/Qwen3-0.6B")
        >>> 
        >>> # Get components
        >>> encoder = engine._get_orchestrator().encoder
        >>> judge_panel = engine._get_orchestrator().judge_panel
        >>> 
        >>> # Create MCTS
        >>> mcts = LatentMCTS(judge_panel, encoder)
        >>> 
        >>> # Search
        >>> seed = encoder.encode("How to implement caching?")
        >>> result = mcts.search(seed)
        >>> 
        >>> # Use best latent
        >>> response = encoder.decode(result.best_latent, query="How to implement caching?")
        >>> print(response)
    
    Integration Note:
        This class is designed to plug into the existing system.
        It uses the same JudgePanel as the evolutionary approach,
        just with a smarter search strategy.
    """
    
    def __init__(
        self,
        judge_panel: JudgePanel,
        encoder=None,  # Optional, used for modifier context
        config: MCTSConfig | None = None,
    ):
        """
        Initialize L-MCTS.
        
        Args:
            judge_panel: Panel of judges for evaluating latents.
                This should be the same panel used in evolution.
            encoder: Optional encoder reference (used for modifier context)
            config: MCTS configuration. Uses defaults if None.
        """
        self.judge_panel = judge_panel
        self.encoder = encoder
        self.config = config or MCTSConfig()
        
        # Statistics
        self.total_evaluations = 0
    
    def search(self, seed: Tensor, max_evaluations: int | None = None) -> MCTSResult:
        """
        Run L-MCTS search from a seed latent.
        
        This is the main entry point. Provide an encoded query and
        get back an optimized latent representation.
        
        Args:
            seed: The seed latent vector (from encoder.encode())
            max_evaluations: Optional limit on judge evaluations
            
        Returns:
            MCTSResult with best latent and search statistics
            
        The search process:
        1. Create root node from seed
        2. For each iteration:
           a. SELECT: Walk down tree using UCB
           b. EXPAND: Create new child node
           c. EVALUATE: Score the new node
           d. BACKPROPAGATE: Update statistics up to root
        3. Return best latent found
        """
        # Reset statistics
        self.total_evaluations = 0
        history = []
        
        # Create root node
        root = create_root_from_latent(seed)
        
        # Set reference for judges (they need the query context)
        for scorer in self.judge_panel.scorers:
            scorer.set_reference(embedding=seed)
        
        # Initial evaluation of root
        root_verdict = self._evaluate_node(root)
        root.update(root_verdict.score)
        
        best_score = root_verdict.score
        best_node = root
        no_improvement_count = 0
        
        # Main MCTS loop
        for iteration in range(self.config.n_iterations):
            # Check evaluation budget
            if max_evaluations and self.total_evaluations >= max_evaluations:
                break
            
            # ========== PHASE 1: SELECTION ==========
            # Walk down tree using UCB until we find a node to expand
            selected = self._select(root)
            
            # ========== PHASE 2: EXPANSION ==========
            # Add a new child to the selected node
            if selected.depth < self.config.max_depth and not selected.is_fully_expanded:
                child = self._expand(selected)
            else:
                child = selected  # Can't expand, evaluate this node
            
            # ========== PHASE 3: EVALUATION ==========
            # Score the new/selected node
            verdict = self._evaluate_node(child)
            score = verdict.score
            
            # ========== PHASE 4: BACKPROPAGATION ==========
            # Update statistics from child up to root
            self._backpropagate(child, score)
            
            # Track best
            if score > best_score:
                best_score = score
                best_node = child
                no_improvement_count = 0
            else:
                no_improvement_count += 1
            
            # Record history
            history.append({
                'iteration': iteration + 1,
                'best_score': best_score,
                'current_score': score,
                'tree_size': self._count_nodes(root),
                'evaluations': self.total_evaluations,
            })
            
            # Early stopping
            if best_score >= self.config.early_stop_threshold:
                break
            if no_improvement_count >= self.config.early_stop_patience:
                break
        
        # Extract best trajectory
        best_trajectory = best_node.get_trajectory()
        
        return MCTSResult(
            best_latent=best_node.latent,
            best_score=best_score,
            best_trajectory=best_trajectory,
            root=root,
            iterations=iteration + 1,
            total_evaluations=self.total_evaluations,
            history=history,
        )
    
    def _select(self, node: LatentNode) -> LatentNode:
        """
        SELECTION PHASE: Walk down tree using UCB.
        
        Starting from the given node (usually root), repeatedly select
        the child with highest UCB value until we reach a leaf or a
        node that isn't fully expanded.
        
        UCB Formula:
            UCB = value + c * sqrt(ln(parent.visits) / visits)
            
        The first term favors high-value nodes (exploitation).
        The second term favors rarely-visited nodes (exploration).
        
        Args:
            node: Starting node (usually root)
            
        Returns:
            The selected node for expansion
        """
        while not node.is_leaf:
            # If node isn't fully expanded, select it for expansion
            if not node.is_fully_expanded:
                return node
            
            # Otherwise, pick best child by UCB
            node = self._best_ucb_child(node)
        
        return node
    
    def _best_ucb_child(self, node: LatentNode) -> LatentNode:
        """
        Select the child with highest UCB value.
        
        This is where exploration vs exploitation is balanced.
        """
        c = self.config.exploration_constant
        
        def ucb(child: LatentNode) -> float:
            if child.visits == 0:
                return float('inf')  # Always try unvisited nodes
            
            exploitation = child.value
            exploration = c * math.sqrt(math.log(node.visits) / child.visits)
            return exploitation + exploration
        
        return max(node.children, key=ucb)
    
    def _expand(self, node: LatentNode) -> LatentNode:
        """
        EXPANSION PHASE: Add a new child node.
        
        This creates a "mutation" of the parent latent. The key insight
        from the article:
        
        > "Mutations would stay on the surface of useful representations."
        
        We use the ModifierJudge (if available) to suggest a direction,
        rather than random noise. This keeps us on the manifold.
        
        Args:
            node: The node to expand
            
        Returns:
            The newly created child node
        """
        # Compute temperature (decreases with depth)
        temperature = max(
            self.config.min_temperature,
            1.0 * (self.config.temperature_decay ** node.depth)
        )
        
        # Get modification direction from judges
        modification = None
        if self.config.use_modifier and self.judge_panel.modifiers:
            context = self._build_context(node)
            modification = self.judge_panel.get_modification(node.latent, context)
        
        # Create child latent
        if modification is not None:
            # Directed mutation: follow the modifier's suggestion
            direction = modification / (modification.norm() + 1e-8)
            noise = torch.randn_like(node.latent) * 0.2
            child_latent = node.latent + temperature * (0.7 * direction + 0.3 * noise)
        else:
            # Random mutation: small step in random direction
            noise = torch.randn_like(node.latent)
            noise = noise / (noise.norm() + 1e-8)  # Normalize
            child_latent = node.latent + temperature * 0.3 * noise
        
        # Create child node
        child = node.add_child(
            child_latent=child_latent,
            temperature=temperature,
            direction=modification,
        )
        
        return child
    
    def _evaluate_node(self, node: LatentNode) -> PanelVerdict:
        """
        EVALUATION PHASE: Score a node using the JudgePanel.
        
        This is where we leverage the existing judge infrastructure.
        Importantly, this is CHEAP - we score the latent directly,
        without needing to decode it to text first.
        
        Args:
            node: The node to evaluate
            
        Returns:
            PanelVerdict with score and optional modification
        """
        context = self._build_context(node)
        verdict = self.judge_panel.evaluate(node.latent, context)
        self.total_evaluations += 1
        return verdict
    
    def _backpropagate(self, node: LatentNode, score: float) -> None:
        """
        BACKPROPAGATION PHASE: Update statistics up the tree.
        
        After evaluating a node, we update the statistics for every
        node on the path from that node back to root. This is how
        MCTS learns which regions of the tree are promising.
        
        Args:
            node: The node that was evaluated
            score: The score from evaluation
        """
        current = node
        while current is not None:
            current.update(score)
            current = current.parent
    
    def _build_context(self, node: LatentNode) -> ModificationContext:
        """
        Build modification context for a node.
        
        This provides the ModifierJudge with information about
        the search trajectory - helpful for suggesting good directions.
        """
        trajectory = node.get_trajectory()
        
        # Compute momentum (direction of recent movement)
        momentum = None
        if len(trajectory) >= 2:
            momentum = trajectory[-1] - trajectory[-2]
        
        # Recent modifications (for the modifier to learn from)
        recent_mods = None
        if len(trajectory) >= 3:
            recent_mods = torch.stack(trajectory[-3:])
        
        # Stuck signal (are we oscillating?)
        stuck = 0.0
        if len(trajectory) >= 5:
            recent_norms = [t.norm().item() for t in trajectory[-5:]]
            if max(recent_norms) - min(recent_norms) < 0.1:
                stuck = 1.0
        
        return ModificationContext(
            momentum=momentum,
            recent_modifications=recent_mods,
            stuck_signal=stuck,
            cross_chain=None,  # Single-chain in MCTS
        )
    
    def _count_nodes(self, node: LatentNode) -> int:
        """Count total nodes in tree."""
        count = 1
        for child in node.children:
            count += self._count_nodes(child)
        return count
    
    def get_top_k_latents(self, root: LatentNode, k: int = 5) -> List[Tensor]:
        """
        Extract top-k latents from the search tree.
        
        Useful when you want multiple candidates for aggregation,
        similar to how the current evolution keeps survivors.
        
        Args:
            root: Root of the search tree
            k: Number of latents to return
            
        Returns:
            List of top-k latents by value
        """
        # Collect all nodes
        all_nodes = []
        
        def collect(node):
            all_nodes.append(node)
            for child in node.children:
                collect(child)
        
        collect(root)
        
        # Sort by value (average score)
        all_nodes.sort(key=lambda n: n.value, reverse=True)
        
        return [n.latent for n in all_nodes[:k]]


# Convenience function for quick usage
def mcts_search(
    seed: Tensor,
    judge_panel: JudgePanel,
    n_iterations: int = 100,
    exploration_constant: float = 1.41,
) -> MCTSResult:
    """
    Quick L-MCTS search.
    
    Convenience function for when you don't need full configuration.
    
    Args:
        seed: Seed latent from encoder.encode()
        judge_panel: Panel of judges for evaluation
        n_iterations: Number of MCTS iterations
        exploration_constant: UCB exploration weight
        
    Returns:
        MCTSResult with best latent found
        
    Example:
        >>> result = mcts_search(seed, judge_panel, n_iterations=50)
        >>> print(f"Best score: {result.best_score:.3f}")
    """
    config = MCTSConfig(
        n_iterations=n_iterations,
        exploration_constant=exploration_constant,
    )
    mcts = LatentMCTS(judge_panel, config=config)
    return mcts.search(seed)

