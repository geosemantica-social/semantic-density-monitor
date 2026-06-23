# Semantic Density Monitor (SD-Monitor)

A lightweight, non-invasive observability tool to detect geometric phase transitions in Large Language Models (LLMs).

## Background

This tool implements the empirical metric **Semantic Density (S)** and the **Phase Inversion (ΔS)** observable defined in the research paper *"The Stone Guest: Harmonic Quantization of Semantic Phase Transitions in Large Language Models"* (Cerda Seguel, 2025).

Unlike behavioral benchmarks, SD-Monitor looks directly at the geometry of the `last_hidden_state` to determine if the model is processing language through a structured, low-entropy manifold, or if it is experiencing entropic drift.

## Installation

```bash
pip install git+https://github.com/geosemantica/semantic-density-monitor.git