"""Self-refinement module for improving generated plans.

Uses the LLM itself to critique and improve its output without training.
This is a zero-shot approach that leverages the model's own capabilities.
"""

from __future__ import annotations

import torch
from torch import Tensor
from transformers import AutoModelForCausalLM, AutoTokenizer

from latent_reasoning.utils.device import get_device


class PlanRefiner:
    """
    Refines generated plans through self-critique and improvement.

    This module uses the LLM to:
    1. Critique a plan for weaknesses
    2. Suggest improvements
    3. Generate an improved version
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-0.6B",
        device_preference: str = "auto",
        max_refinement_rounds: int = 2,
    ):
        """
        Initialize the plan refiner.

        Args:
            model_name: HuggingFace model ID
            device_preference: Device to use
            max_refinement_rounds: Maximum refinement iterations
        """
        self.model_name = model_name
        self.device = get_device(device_preference)
        self.max_rounds = max_refinement_rounds

        # Load model and tokenizer
        self._load_model()

    def _load_model(self) -> None:
        """Load the model and tokenizer."""
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            torch_dtype=torch.float16,
        )
        self.model.to(self.device)
        self.model.eval()

    def refine(
        self,
        plan: str,
        query: str,
        num_rounds: int | None = None,
    ) -> str:
        """
        Refine a plan through iterative self-critique.

        Args:
            plan: Initial plan to refine
            query: Original query the plan addresses
            num_rounds: Number of refinement rounds (default: max_rounds)

        Returns:
            Refined plan
        """
        rounds = num_rounds or self.max_rounds
        current_plan = plan

        for i in range(rounds):
            # Get critique
            critique = self._critique(current_plan, query)

            # Generate improved version
            improved = self._improve(current_plan, critique, query)

            # If improvement is empty or too short, keep current
            if len(improved.strip()) < 100:
                break

            current_plan = improved

        return current_plan

    def _critique(self, plan: str, query: str) -> str:
        """Generate a critique of the plan."""
        system_msg = """You are a critical reviewer of technical plans.
Identify weaknesses, missing steps, unclear explanations, or logical gaps.
Be specific and constructive. Keep your critique concise (3-5 key points)."""

        user_msg = f"""Review this plan for: {query}

PLAN:
{plan[:2000]}

What are the main weaknesses or areas for improvement?"""

        return self._generate(system_msg, user_msg, max_tokens=300)

    def _improve(self, plan: str, critique: str, query: str) -> str:
        """Generate an improved version based on critique."""
        system_msg = """You are an expert at improving technical plans.
Given a plan and its critique, create an improved version that addresses the issues.
Keep the good parts, fix the weaknesses, and make the plan more actionable."""

        user_msg = f"""Original query: {query}

CURRENT PLAN:
{plan[:1500]}

CRITIQUE:
{critique[:500]}

Create an improved version of this plan that addresses the critique.
Make it clear, actionable, and comprehensive."""

        return self._generate(system_msg, user_msg, max_tokens=2048)

    def _generate(self, system_msg: str, user_msg: str, max_tokens: int) -> str:
        """Generate text using the model."""
        # Build prompt
        if hasattr(self.tokenizer, 'apply_chat_template'):
            messages = [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ]
            try:
                prompt = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except Exception:
                prompt = f"<|im_start|>system\n{system_msg}<|im_end|>\n<|im_start|>user\n{user_msg}<|im_end|>\n<|im_start|>assistant\n"
        else:
            prompt = f"System: {system_msg}\n\nUser: {user_msg}\n\nAssistant: "

        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.2,
            )

        generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract response
        if user_msg in generated:
            idx = generated.find(user_msg) + len(user_msg)
            response = generated[idx:].strip()
        else:
            response = generated.strip()

        # Clean up markers
        for marker in ["Assistant:", "assistant:", "<|im_start|>", "<|im_end|>"]:
            if response.startswith(marker):
                response = response[len(marker):].strip()

        # Remove thinking block
        if "<think>" in response and "</think>" in response:
            think_end = response.find("</think>") + len("</think>")
            response = response[think_end:].strip()

        return response


class EnsembleDecoder:
    """
    Generates multiple outputs and selects the best one.

    This is useful when evolution produces multiple viable latents -
    we can decode each and pick the best using heuristics or voting.
    """

    def __init__(
        self,
        encoder,  # LLMEncoder instance
        heuristic_scorer=None,  # Optional HeuristicScorer
    ):
        """
        Initialize the ensemble decoder.

        Args:
            encoder: LLMEncoder instance for decoding
            heuristic_scorer: Optional HeuristicScorer for ranking
        """
        self.encoder = encoder
        self.heuristic_scorer = heuristic_scorer

    def decode_ensemble(
        self,
        latents: list[Tensor],
        query: str,
        temperatures: list[float] | None = None,
    ) -> tuple[str, list[str], list[float]]:
        """
        Decode multiple latents and return the best output.

        Args:
            latents: List of latent vectors to decode
            query: Original query
            temperatures: Optional list of temperatures for each decode

        Returns:
            Tuple of (best_output, all_outputs, scores)
        """
        if temperatures is None:
            temperatures = [0.0] * len(latents)

        outputs = []
        for latent, temp in zip(latents, temperatures):
            output = self.encoder.decode(latent, query=query, temperature=temp)
            outputs.append(output)

        # Score outputs
        if self.heuristic_scorer:
            scores = [
                self.heuristic_scorer.score(output).overall_score
                for output in outputs
            ]
        else:
            # Use length as proxy for quality
            scores = [len(output) / 2000 for output in outputs]

        # Find best
        best_idx = max(range(len(scores)), key=lambda i: scores[i])
        best_output = outputs[best_idx]

        return best_output, outputs, scores

    def decode_with_temperature_sweep(
        self,
        latent: Tensor,
        query: str,
        temperatures: list[float] = [0.0, 0.3, 0.5, 0.7],
    ) -> tuple[str, list[str], list[float]]:
        """
        Decode the same latent at different temperatures.

        This explores how temperature affects output quality
        for the same latent representation.

        Args:
            latent: Single latent vector
            query: Original query
            temperatures: Temperatures to try

        Returns:
            Tuple of (best_output, all_outputs, scores)
        """
        outputs = []
        for temp in temperatures:
            output = self.encoder.decode(latent, query=query, temperature=temp)
            outputs.append(output)

        # Score outputs
        if self.heuristic_scorer:
            scores = [
                self.heuristic_scorer.score(output).overall_score
                for output in outputs
            ]
        else:
            scores = [len(output) / 2000 for output in outputs]

        best_idx = max(range(len(scores)), key=lambda i: scores[i])
        best_output = outputs[best_idx]

        return best_output, outputs, scores
