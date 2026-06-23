#!/usr/bin/env python3
"""
Validation example: Replicating The Stone Guest on Pythia-410m
"""

from transformers import AutoTokenizer, AutoModelForCausalLM
from sd_monitor import SDMonitor

MODEL_NAME = "EleutherAI/pythia-410m"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto",
)

monitor = SDMonitor(model, tokenizer)

coherent = "La inteligencia artificial transforma nuestra comprensión del lenguaje."
broken = "Subatómicas partículas de comportamiento el describe cuántica teoría la."

results = monitor.evaluate(coherent, broken)

print("=" * 60)
print("THE STONE GUEST - VALIDATION SAMPLE")
print("=" * 60)
print(f"Coherent S:  {results['S_coherent']:.4f}")
print(f"Broken S:    {results['S_broken']:.4f}")
print(f"Delta S:     {results['Delta_S']:+.4f}")
print(f"Status:      {results['Status']}")
print("=" * 60)