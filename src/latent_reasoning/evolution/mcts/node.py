"""
Latent Tree Node for Monte Carlo Tree Search in Latent Space.

This module defines the LatentNode class - the fundamental building block of L-MCTS.
Each node represents a point in latent space that we're exploring.

=============================================================================
WHAT IS A NODE IN L-MCTS?
=============================================================================

In traditional MCTS (like AlphaGo), nodes represent game states.
In L-MCTS, nodes represent points in the model's latent space - internal 
representations that will eventually be decoded into text.

Key insight from the original article:
> "The model is a simulator. It can simulate many possible continuations."

Each node is one of those possible continuations, existing in vector form
before we commit to decoding it into tokens.

=============================================================================
WHY TREE STRUCTURE?
=============================================================================

Current evolution in this repo uses a "population" approach:
- Generate N random mutations
- Score all of them
- Keep the best, repeat

This is inefficient because:
1. Random mutations often leave the "useful" latent manifold
2. No memory of which directions were promising
3. Can't backtrack or refine

Tree structure fixes this:
1. Children are directed mutations from parents (staying near manifold)
2. We remember which branches were promising (via visit counts + scores)
3. UCB selection naturally balances exploration vs exploitation

=============================================================================
NODE STATISTICS
=============================================================================

Each node tracks:
- visits: How many times we've explored through this node
- total_score: Sum of all scores from simulations through this node
- best_score: Best score seen in subtree (for pruning decisions)

The ratio (total_score / visits) gives the average value - this is key to UCB.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional
import torch
from torch import Tensor


@dataclass
class LatentNode:
    """
    A node in the latent space search tree.
    
    Think of this as a "candidate thought" the model could have. The tree
    structure lets us explore many thoughts while remembering which ones
    were promising.
    
    Attributes:
        latent: The actual latent vector (internal model representation)
        parent: Reference to parent node (None for root)
        children: List of child nodes (mutations/variations of this node)
        visits: Number of times this node was visited during search
        total_score: Cumulative score from all simulations through this node
        best_score: Best score observed in this subtree
        depth: How deep in the tree (root = 0)
        creation_temperature: The temperature used when creating this node
        modification_direction: The mutation direction used to create this
    
    Example:
        >>> # Create root node from encoded query
        >>> root = LatentNode(latent=encoder.encode("How to implement caching?"))
        >>> print(f"Root at depth {root.depth}")  # depth 0
        >>> 
        >>> # After MCTS runs, check statistics
        >>> print(f"Visited {root.visits} times")
        >>> print(f"Average value: {root.value:.3f}")
    """
    
    # Core data
    latent: Tensor
    parent: Optional[LatentNode] = None
    children: List[LatentNode] = field(default_factory=list)
    
    # MCTS statistics - these are crucial for UCB selection
    visits: int = 0
    total_score: float = 0.0
    best_score: float = float('-inf')
    
    # Metadata
    depth: int = 0
    creation_temperature: float = 1.0
    modification_direction: Optional[Tensor] = None
    
    # Optional: store the raw score from last evaluation
    last_score: Optional[float] = None
    
    def __post_init__(self):
        """Automatically compute depth from parent."""
        if self.parent is not None:
            self.depth = self.parent.depth + 1
    
    @property
    def value(self) -> float:
        """
        Average value of this node (total_score / visits).
        
        This is the "exploitation" term in UCB - how good is this node
        based on what we've seen so far?
        
        Returns:
            Average score, or 0 if never visited
        """
        if self.visits == 0:
            return 0.0
        return self.total_score / self.visits
    
    @property
    def is_leaf(self) -> bool:
        """
        Is this a leaf node (no children yet)?
        
        Leaf nodes are candidates for expansion - we might want to
        explore what mutations of this latent look like.
        """
        return len(self.children) == 0
    
    @property
    def is_root(self) -> bool:
        """Is this the root node?"""
        return self.parent is None
    
    @property
    def is_fully_expanded(self) -> bool:
        """
        Have we expanded this node enough?
        
        In game MCTS, this means all legal moves have been tried.
        In L-MCTS, we use a heuristic: expand up to max_children.
        """
        return len(self.children) >= self.max_children
    
    @property
    def max_children(self) -> int:
        """
        Maximum children for this node.
        
        Deeper nodes get fewer children (focus the search).
        This is a design choice - you could make this configurable.
        """
        # Fewer children at deeper levels (focus the search)
        base_children = 5
        return max(2, base_children - self.depth)
    
    def add_child(self, child_latent: Tensor, temperature: float = 1.0,
                  direction: Optional[Tensor] = None) -> LatentNode:
        """
        Create and add a child node.
        
        Args:
            child_latent: The latent vector for the child
            temperature: Temperature used to create this child
            direction: Optional mutation direction used
            
        Returns:
            The newly created child node
            
        Example:
            >>> # Expand node with a mutation
            >>> mutation = parent.latent + 0.1 * torch.randn_like(parent.latent)
            >>> child = parent.add_child(mutation, temperature=0.5)
        """
        child = LatentNode(
            latent=child_latent,
            parent=self,
            creation_temperature=temperature,
            modification_direction=direction,
        )
        self.children.append(child)
        return child
    
    def update(self, score: float) -> None:
        """
        Update this node's statistics after a simulation.
        
        This is called during backpropagation - we update the score
        for this node and all its ancestors.
        
        Args:
            score: The score from the simulation
        """
        self.visits += 1
        self.total_score += score
        self.best_score = max(self.best_score, score)
        self.last_score = score
    
    def get_trajectory(self) -> List[Tensor]:
        """
        Get the path from root to this node.
        
        Useful for understanding how we got here and for computing
        momentum/direction information.
        
        Returns:
            List of latent vectors from root to this node
        """
        trajectory = []
        node = self
        while node is not None:
            trajectory.append(node.latent)
            node = node.parent
        return list(reversed(trajectory))
    
    def get_best_child(self) -> Optional[LatentNode]:
        """
        Get the child with highest average value.
        
        Used at the end of search to extract the best path.
        """
        if not self.children:
            return None
        return max(self.children, key=lambda c: c.value)
    
    def get_most_visited_child(self) -> Optional[LatentNode]:
        """
        Get the most visited child.
        
        In MCTS, most visited often beats highest value because
        visit count reflects confidence, not just score.
        """
        if not self.children:
            return None
        return max(self.children, key=lambda c: c.visits)
    
    def __repr__(self) -> str:
        return (
            f"LatentNode(depth={self.depth}, visits={self.visits}, "
            f"value={self.value:.3f}, children={len(self.children)})"
        )


def create_root_from_latent(latent: Tensor) -> LatentNode:
    """
    Create a root node from an encoded query latent.
    
    This is the starting point of L-MCTS - we encode the query,
    then build a tree of variations around it.
    
    Args:
        latent: The encoded query latent (from encoder.encode())
        
    Returns:
        Root node for the search tree
        
    Example:
        >>> encoder = LLMEncoder("Qwen/Qwen3-0.6B")
        >>> query_latent = encoder.encode("Design a REST API")
        >>> root = create_root_from_latent(query_latent)
        >>> # Now run MCTS on root
    """
    return LatentNode(
        latent=latent.clone(),  # Clone to avoid modifying original
        parent=None,
        depth=0,
    )

