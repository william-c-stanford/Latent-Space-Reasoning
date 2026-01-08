"""
SciEvalKit model wrappers for Latent Space Reasoning evaluation.

This module provides model wrappers that are compatible with SciEvalKit's
evaluation infrastructure without modifying the benchmark repo.

Two wrappers are provided:
1. BaselineQwenModel - Direct Qwen3 generation (no latent space reasoning)
2. LatentReasoningModel - Qwen3 with latent space reasoning

Both wrappers implement the interface expected by SciEvalKit's inference loop:
- generate(message, dataset=None) -> str
- set_dump_image(func) -> None
"""

import torch
from typing import List, Dict, Optional

from transformers import AutoTokenizer, AutoModelForCausalLM


class BaselineQwenModel:
    """
    Baseline Qwen model without latent space reasoning.
    Uses direct text generation from the model.
    """
    
    INSTALL_REQ = False
    INTERLEAVE = False
    is_api = False
    
    def __init__(
        self,
        model_path: str = "Qwen/Qwen3-0.6B",
        device: str = "auto",
        max_new_tokens: int = 1024,
        temperature: float = 0.7,
        quantization: Optional[str] = "4bit",
        **kwargs
    ):
        """
        Initialize baseline Qwen model.
        
        Args:
            model_path: HuggingFace model path
            device: Device to use ('auto', 'cuda', 'cpu')
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            quantization: Quantization mode ('4bit', 'none')
        """
        self.model_path = model_path
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        
        print(f"[Baseline] Loading model: {model_path}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, 
            trust_remote_code=True
        )
        
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        load_kwargs = {
            "trust_remote_code": True,
            "torch_dtype": torch.float16 if self.device == "cuda" else torch.float32,
        }
        
        if quantization == "4bit" and self.device == "cuda":
            try:
                from transformers import BitsAndBytesConfig
                load_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                )
                load_kwargs["device_map"] = "auto"
            except ImportError:
                print("[Baseline] bitsandbytes not available, loading without quantization")
                load_kwargs["device_map"] = "auto"
        else:
            load_kwargs["device_map"] = "auto" if self.device == "cuda" else None
            
        self.model = AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs)
        
        if load_kwargs.get("device_map") is None:
            self.model = self.model.to(self.device)
            
        self.model.eval()
        print(f"[Baseline] Model loaded on device: {self.device}")
        
    def set_dump_image(self, func):
        """Compatibility method for SciEvalKit. Text-only model ignores images."""
        self.dump_image_func = func
        
    def _extract_text_from_message(self, message: List[Dict]) -> str:
        """Extract text content from SciEvalKit message format."""
        texts = []
        for item in message:
            if isinstance(item, dict):
                if item.get('type') == 'text':
                    texts.append(item.get('value', ''))
            elif isinstance(item, str):
                texts.append(item)
        return '\n'.join(texts)
    
    def generate(self, message, dataset: str = None, **kwargs) -> str:
        """
        Generate response using baseline Qwen model.
        
        Args:
            message: List of message dicts with 'type' and 'value' keys
            dataset: Dataset name (unused, for API compatibility)
            
        Returns:
            Generated text response
        """
        if isinstance(message, str):
            prompt = message
        elif isinstance(message, list):
            prompt = self._extract_text_from_message(message)
        else:
            prompt = str(message)
            
        messages = [{"role": "user", "content": prompt}]
        
        if hasattr(self.tokenizer, 'apply_chat_template'):
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        else:
            text = f"User: {prompt}\nAssistant:"
            
        inputs = self.tokenizer(text, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                do_sample=self.temperature > 0,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            )
            
        response = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )
        
        return response.strip()


class LatentReasoningModel:
    """
    Latent Space Reasoning wrapper for SciEvalKit.
    Uses evolutionary optimization in latent space for improved responses.
    """
    
    INSTALL_REQ = False
    INTERLEAVE = False
    is_api = False
    
    def __init__(
        self,
        model_path: str = "Qwen/Qwen3-0.6B",
        device: str = "auto",
        max_new_tokens: int = 1024,
        temperature: float = 0.7,
        quantization: str = "4bit",
        chains: int = 5,
        generations: int = 10,
        verbosity: str = "minimal",
        **kwargs
    ):
        """
        Initialize Latent Space Reasoning model.
        
        Args:
            model_path: HuggingFace model path for encoder
            device: Device to use
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            quantization: Quantization mode
            chains: Number of evolution chains
            generations: Maximum evolution generations
            verbosity: Logging verbosity
        """
        self.model_path = model_path
        self.max_new_tokens = max_new_tokens
        
        print(f"[LatentReasoning] Loading engine with model: {model_path}")
        
        from latent_reasoning import Engine, Config
        
        config = Config()
        config.encoder.model = model_path
        config.encoder.quantization = quantization
        config.encoder.device = device
        config.evolution.chains = chains
        config.evolution.generations = generations
        config.synthesis.max_tokens = max_new_tokens
        config.synthesis.temperature = temperature
        config.output.verbosity = verbosity
        
        self.engine = Engine(config=config)
        print(f"[LatentReasoning] Engine initialized")
        
    def set_dump_image(self, func):
        """Compatibility method for SciEvalKit. Text-only model ignores images."""
        self.dump_image_func = func
        
    def _extract_text_from_message(self, message: List[Dict]) -> str:
        """Extract text content from SciEvalKit message format."""
        texts = []
        for item in message:
            if isinstance(item, dict):
                if item.get('type') == 'text':
                    texts.append(item.get('value', ''))
            elif isinstance(item, str):
                texts.append(item)
        return '\n'.join(texts)
    
    def generate(self, message, dataset: str = None, **kwargs) -> str:
        """
        Generate response using latent space reasoning.
        
        Args:
            message: List of message dicts with 'type' and 'value' keys
            dataset: Dataset name (unused, for API compatibility)
            
        Returns:
            Generated text response after latent space evolution
        """
        if isinstance(message, str):
            prompt = message
        elif isinstance(message, list):
            prompt = self._extract_text_from_message(message)
        else:
            prompt = str(message)
            
        result = self.engine.run(prompt)
        
        return result.plan

