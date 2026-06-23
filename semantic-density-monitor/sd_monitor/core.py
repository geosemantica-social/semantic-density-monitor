"""Semantic Density Monitor - Core geometric phase detection"""

import torch
import numpy as np
from scipy.linalg import svd
from typing import Dict


class SDMonitor:
    """
    Semantic Density Monitor: Computes S and ΔS for transformer embeddings.

    Based on: Cerda Seguel, D. (2025). The Stone Guest.
    """

    def __init__(self, model, tokenizer, max_length: int = 128):
        self.model = model
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.model.eval()

    def _compute_S(self, text: str) -> float:
        # Tokenize and forward pass
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
            padding=True,
        )
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)

        embeddings = outputs.hidden_states[-1][0].cpu().numpy()

        if embeddings.shape[0] < 2:
            return 0.0

        # Normalize
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        emb_norm = embeddings / (norms + 1e-10)

        # 1. Local coherence
        sims = np.einsum("ij,ij->i", emb_norm[:-1], emb_norm[1:])
        local_coh = np.mean(sims)
        local_coh = np.clip(local_coh, -1.0, 1.0)

        # 2. Spectral focus
        try:
            _, s, _ = svd(emb_norm, full_matrices=False)
            s_squared = s**2
            spectral_focus = np.sum(s_squared[:5]) / (np.sum(s_squared) + 0.1)
        except np.linalg.LinAlgError:
            spectral_focus = 0.0

        # 3. True isotropy
        gram = emb_norm @ emb_norm.T
        eig_vals = np.linalg.eigvalsh(gram)
        eig_vals = eig_vals[eig_vals > 1e-6]

        if len(eig_vals) > 0:
            condition_num = np.max(eig_vals) / (np.min(eig_vals) + 0.1)
            true_isotropy = 1.0 / (1.0 + np.log(condition_num + 1))
        else:
            true_isotropy = 0.0

        # 4. Semantic Density (S)
        S = ((1 + local_coh) / 2) * spectral_focus * (1 - true_isotropy)

        return float(S)

    def evaluate(self, coherent_text: str, broken_text: str) -> Dict:
        S_coh = self._compute_S(coherent_text)
        S_bro = self._compute_S(broken_text)
        delta_S = S_coh - S_bro

        return {
            "S_coherent": S_coh,
            "S_broken": S_bro,
            "Delta_S": delta_S,
            "Phase_Inversion": delta_S < 0,
            "Status": "STABLE_MANIFOLD" if delta_S < 0 else "ENTROPIC_DRIFT",
        }