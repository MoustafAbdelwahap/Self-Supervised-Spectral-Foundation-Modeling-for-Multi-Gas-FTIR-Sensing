"""Local LLM inference and rule-based safety advisory engine for FTIR gas interpretation.

Uses lightweight Qwen/Qwen2.5-0.5B to generate human-readable safety assessments and
actionable mitigation guidance from multi-gas detection/quantification predictions,
with automatic fallback to deterministic rule-based heuristics when running offline or memory-constrained.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List

import torch

from sfm.interpretation.scenarios import (
    format_prediction_context,
    get_rule_based_advisory,
)

logger = logging.getLogger(__name__)


class GasAdvisor:
    """Natural language interpretation engine for gas sensing predictions.

    Integrates local small language models (SLMs) via HuggingFace transformers
    with a graceful deterministic fallback for low-resource CPU or offline environments.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-0.5B",
        device: str = "cpu",
    ) -> None:
        """Initialize GasAdvisor.

        Args:
            model_name: HuggingFace model identifier (default: Qwen/Qwen2.5-0.5B).
            device: Execution device ('cpu' or 'cuda').
        """
        self.model_name = model_name
        self.device = torch.device(device if (device == "cpu" or torch.cuda.is_available()) else "cpu")
        self.use_llm = False
        self.tokenizer = None
        self.model = None

        logger.info("Initializing GasAdvisor (target model: %s on %s)...", model_name, self.device)
        self._try_load_model()

    def _try_load_model(self) -> None:
        """Attempt to load the causal language model and tokenizer."""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            start_t = time.time()
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            dtype = torch.float32 if self.device.type == "cpu" else torch.float16
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=dtype,
            )
            self.model.eval()
            self.model.to(self.device)
            self.use_llm = True
            logger.info(
                "Successfully loaded LLM %s in %.2fs. Local generation active.",
                self.model_name,
                time.time() - start_t,
            )
        except Exception as e:
            logger.warning(
                "Local LLM loading failed (%s). Falling back to rule-based gas advisory engine.",
                e,
            )
            self.use_llm = False
            self.tokenizer = None
            self.model = None

    def generate_advisory(
        self,
        gas_names: List[str],
        presence: List[bool],
        concentrations: List[float],
    ) -> str:
        """Generate a concise, human-readable safety assessment and mitigation advisory.

        Args:
            gas_names: List of monitored gas names.
            presence: Boolean presence flag for each gas.
            concentrations: Measured concentration values.

        Returns:
            2-3 sentence safety recommendation.
        """
        if self.use_llm and self.model is not None and self.tokenizer is not None:
            try:
                return self._generate_llm(gas_names, presence, concentrations)
            except Exception as e:
                logger.warning("LLM inference error (%s); falling back to rule-based engine.", e)

        return self._generate_rule_based(gas_names, presence, concentrations)

    def _generate_llm(
        self,
        gas_names: List[str],
        presence: List[bool],
        concentrations: List[float],
    ) -> str:
        """Generate advisory via local autoregressive LLM."""
        context = format_prediction_context(gas_names, presence, concentrations)
        system_prompt = (
            "You are an industrial safety expert. Given the following FTIR multi-gas sensor readings, "
            "provide a brief safety assessment and actionable mitigation recommendations. "
            "Be clear and concise (2-3 sentences max)."
        )

        # Check if chat template is supported
        if hasattr(self.tokenizer, "apply_chat_template") and self.tokenizer.chat_template:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": context},
            ]
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            prompt = f"System: {system_prompt}\n\n{context}\n\nSafety Advisory:"

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            max_length=512,
            truncation=True,
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=150,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        gen_tokens = outputs[0][inputs["input_ids"].shape[1] :]
        advisory = self.tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
        return advisory if advisory else self._generate_rule_based(gas_names, presence, concentrations)

    def _generate_rule_based(
        self,
        gas_names: List[str],
        presence: List[bool],
        concentrations: List[float],
    ) -> str:
        """Generate deterministic rule-based safety advisory based on exposure limits."""
        hazards: List[str] = []
        monitored_detected = 0

        for name, is_present, conc in zip(gas_names, presence, concentrations):
            if is_present or conc > 0.01:
                monitored_detected += 1
                adv = get_rule_based_advisory(name, conc)
                # Only append if an elevated hazard or warning is present
                if "within normal baseline" not in adv:
                    hazards.append(adv)

        if not hazards:
            if monitored_detected > 0:
                return (
                    "Safety Assessment: Detected gas levels remain within standard baseline thresholds. "
                    "No immediate hazard detected. Maintain normal ventilation and continuous monitoring."
                )
            return (
                "Safety Assessment: No target gases detected above sensor noise floor. "
                "Atmospheric conditions are within standard operational limits. No mitigation actions required."
            )

        combined = " ".join(hazards)
        return (
            f"Safety Advisory: {combined} Immediate inspection of source and ventilation protocols recommended."
        )

    def batch_interpret(self, predictions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Interpret a list of multi-gas prediction dictionaries.

        Args:
            predictions: List of dicts, each with keys 'gas_names', 'presence', 'concentrations'.

        Returns:
            List of prediction dicts augmented with an 'advisory' text field.
        """
        results: List[Dict[str, Any]] = []
        for pred in predictions:
            gas_names = pred.get("gas_names", [])
            presence = [bool(p) for p in pred.get("presence", [])]
            concentrations = [float(c) for c in pred.get("concentrations", [])]

            advisory = self.generate_advisory(
                gas_names=gas_names,
                presence=presence,
                concentrations=concentrations,
            )

            result_item = dict(pred)
            result_item["advisory"] = advisory
            results.append(result_item)
        return results

