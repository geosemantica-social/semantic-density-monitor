import gradio as gr
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
from scipy.linalg import svd
import warnings
import random
warnings.filterwarnings('ignore')

# ---------- CLASE SDMonitor ----------
class SDMonitor:
    def __init__(self, model, tokenizer, max_length=128):
        self.model = model
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.model.eval()

    def _compute_S(self, text):
        try:
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

            if not hasattr(outputs, 'hidden_states') or outputs.hidden_states is None:
                return None
                
            embeddings = outputs.hidden_states[-1][0].cpu().numpy()

            if embeddings.shape[0] < 2:
                return None

            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            emb_norm = embeddings / (norms + 1e-10)

            sims = np.einsum("ij,ij->i", emb_norm[:-1], emb_norm[1:])
            local_coh = np.mean(sims)
            local_coh = np.clip(local_coh, -1.0, 1.0)

            try:
                _, s, _ = svd(emb_norm, full_matrices=False)
                s_squared = s**2
                spectral_focus = np.sum(s_squared[:5]) / (np.sum(s_squared) + 0.1)
            except np.linalg.LinAlgError:
                spectral_focus = 0.0

            gram = emb_norm @ emb_norm.T
            eig_vals = np.linalg.eigvalsh(gram)
            eig_vals = eig_vals[eig_vals > 1e-6]

            if len(eig_vals) > 0:
                condition_num = np.max(eig_vals) / (np.min(eig_vals) + 0.1)
                true_isotropy = 1.0 / (1.0 + np.log(condition_num + 1))
            else:
                true_isotropy = 0.0

            S = ((1 + local_coh) / 2) * spectral_focus * (1 - true_isotropy)
            return float(S)
        except Exception as e:
            print(f"Error: {e}")
            return None

    def evaluate_corpus(self, coherent_list, broken_list):
        S_coh = []
        S_bro = []
        errors = 0

        for i, (coh, bro) in enumerate(zip(coherent_list, broken_list)):
            s_coh = self._compute_S(coh)
            s_bro = self._compute_S(bro)
            if s_coh is not None:
                S_coh.append(s_coh)
            else:
                errors += 1
            if s_bro is not None:
                S_bro.append(s_bro)
            else:
                errors += 1

        if len(S_coh) == 0 or len(S_bro) == 0:
            return {"Error": f"No sentences could be processed. Errors: {errors}"}

        mean_coh = np.mean(S_coh)
        mean_bro = np.mean(S_bro)
        delta_S = mean_coh - mean_bro

        return {
            "S_coh_mean": float(mean_coh),
            "S_bro_mean": float(mean_bro),
            "Delta_S": float(delta_S),
            "Phase_Inversion": delta_S < 0,
            "Status": "STABLE_MANIFOLD" if delta_S < 0 else "ENTROPIC_DRIFT",
            "N_coherent": len(S_coh),
            "N_broken": len(S_bro),
            "Errors": errors,
        }

# ---------- FUNCIÓN PARA GENERAR CORPUS ROTO ----------
def generate_broken(corpus, seed=42):
    random.seed(seed)
    broken = []
    for sent in corpus:
        words = sent.rstrip('.').split()
        shuffled = random.sample(words, len(words))
        broken.append(' '.join(shuffled) + '.')
    return broken

# ---------- CORPUS ESPAÑOL (50 frases) ----------
CORPUS_ES_COHERENT = [
    "La inteligencia artificial transforma nuestra comprensión del lenguaje.",
    "Los neurocientíficos estudian las conexiones sinápticas del cerebro humano.",
    "La fotosíntesis convierte la luz solar en energía química.",
    "Los átomos se organizan en estructuras cristalinas complejas.",
    "El ADN contiene las instrucciones genéticas de los organismos vivos.",
    "Las ondas electromagnéticas viajan a la velocidad de la luz.",
    "Los ecosistemas marinos mantienen el equilibrio del planeta.",
    "La gravedad mantiene a los planetas en órbita alrededor del sol.",
    "Las reacciones químicas liberan o absorben energía térmica.",
    "Los algoritmos de aprendizaje profundo procesan millones de datos.",
    "La mecánica cuántica desafía nuestra intuición sobre la realidad.",
    "El cambio climático es el desafío más urgente de nuestra generación.",
    "Las células madre ofrecen esperanza para enfermedades degenerativas.",
    "La teoría de la relatividad transformó nuestra comprensión del espacio.",
    "Los exoplanetas podrían albergar formas de vida desconocidas.",
    "El gato se sentó en la alfombra y comenzó a ronronear.",
    "Los niños juegan al fútbol en el parque cada tarde soleada.",
    "Mi abuela prepara el desayuno todas las mañanas temprano.",
    "El perro ladra cuando escucha ruidos extraños en la noche.",
    "Las flores del jardín necesitan agua todos los días.",
    "Mi hermano estudia medicina en la universidad de Santiago.",
    "El autobús llega puntualmente a las ocho de la mañana.",
    "La profesora explica la lección con mucha paciencia.",
    "Los pájaros cantan al amanecer desde los árboles cercanos.",
    "El cartero entrega las cartas en la tarde.",
    "El café de la mañana es el mejor momento del día.",
    "Mi hermana mayor vive en Barcelona desde hace años.",
    "El cine de barrio es un tesoro cultural que debemos preservar.",
    "Los viernes por la noche siempre salimos a cenar.",
    "La biblioteca municipal organiza actividades para niños.",
    "El tiempo fluye irreversiblemente hacia el futuro.",
    "La conciencia emerge de procesos neurales complejos.",
    "La belleza reside en la percepción subjetiva del observador.",
    "El conocimiento se construye a través de la experiencia acumulada.",
    "La libertad implica responsabilidad sobre nuestras decisiones.",
    "El significado surge del contexto y las relaciones semánticas.",
    "La verdad se revela mediante el método científico riguroso.",
    "La memoria transforma nuestras experiencias en narrativas coherentes.",
    "El lenguaje estructura nuestra forma de pensar el mundo.",
    "La razón nos permite distinguir lo real de lo aparente.",
    "La inteligencia artificial está redefiniendo los límites de la creatividad.",
    "El internet de las cosas conectará cada dispositivo del hogar.",
    "La computación cuántica resolverá problemas imposibles para las máquinas actuales.",
    "Los vehículos autónomos transformarán el transporte urbano.",
    "La impresión 3D permitirá construir casas en el espacio.",
    "El conocimiento científico siempre debe estar al servicio de la humanidad.",
    "La educación es la herramienta más poderosa para transformar sociedades.",
    "El aprendizaje continuo es clave para adaptarse al cambio tecnológico.",
    "La colaboración entre disciplinas genera las soluciones más innovadoras.",
    "La curiosidad es el motor más potente del conocimiento humano."
]

# ---------- CORPUS INGLÉS (50 frases) ----------
CORPUS_EN_COHERENT = [
    "Artificial intelligence is transforming our understanding of language.",
    "Neuroscientists study the synaptic connections of the human brain.",
    "Photosynthesis converts sunlight into chemical energy.",
    "Atoms organize into complex crystalline structures.",
    "DNA contains the genetic instructions of all living organisms.",
    "Electromagnetic waves travel at the speed of light.",
    "Marine ecosystems maintain the planet's delicate balance.",
    "Gravity keeps planets in orbit around the sun.",
    "Chemical reactions release or absorb thermal energy.",
    "Deep learning algorithms process millions of data points.",
    "Quantum mechanics challenges our intuition about reality.",
    "Climate change is the most urgent challenge of our generation.",
    "Stem cells offer hope for degenerative diseases.",
    "The theory of relativity transformed our understanding of space.",
    "Exoplanets may harbor unknown forms of life.",
    "The cat sat on the carpet and began to purr.",
    "Children play soccer in the park every sunny afternoon.",
    "My grandmother prepares breakfast early every morning.",
    "The dog barks when it hears strange noises at night.",
    "The flowers in the garden need water every day.",
    "My brother studies medicine at the University of Santiago.",
    "The bus arrives punctually at eight in the morning.",
    "The teacher explains the lesson with great patience.",
    "The birds sing at dawn from the nearby trees.",
    "The mailman delivers the letters in the afternoon.",
    "Morning coffee is the best moment of the day.",
    "My older sister has lived in Barcelona for years.",
    "The neighborhood cinema is a cultural treasure we must preserve.",
    "On Friday nights we always go out for dinner.",
    "The public library organizes activities for children.",
    "Time flows irreversibly toward the future.",
    "Consciousness emerges from complex neural processes.",
    "Beauty resides in the subjective perception of the observer.",
    "Knowledge is constructed through accumulated experience.",
    "Freedom implies responsibility for our decisions.",
    "Meaning arises from context and semantic relationships.",
    "Truth is revealed through rigorous scientific method.",
    "Memory transforms our experiences into coherent narratives.",
    "Language structures our way of thinking about the world.",
    "Reason allows us to distinguish the real from the apparent.",
    "Artificial intelligence is redefining the boundaries of creativity.",
    "The Internet of Things will connect every device in the home.",
    "Quantum computing will solve problems impossible for current machines.",
    "Autonomous vehicles will transform urban transportation.",
    "3D printing will enable the construction of houses in space.",
    "Scientific knowledge must always serve humanity.",
    "Education is the most powerful tool for transforming societies.",
    "Continuous learning is key to adapting to technological change.",
    "Collaboration across disciplines generates the most innovative solutions.",
    "Curiosity is the most powerful engine of human knowledge."
]

# Generar corpus rotos
CORPUS_ES_BROKEN = generate_broken(CORPUS_ES_COHERENT)
CORPUS_EN_BROKEN = generate_broken(CORPUS_EN_COHERENT)

# ---------- MODELO Y MONITOR ----------
print("Loading DistilBERT model...")
MODEL_NAME = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)
monitor = SDMonitor(model, tokenizer)

# ---------- FUNCIONES DE LA DEMO ----------
def analyze_corpus_es():
    return monitor.evaluate_corpus(CORPUS_ES_COHERENT, CORPUS_ES_BROKEN)

def analyze_corpus_en():
    return monitor.evaluate_corpus(CORPUS_EN_COHERENT, CORPUS_EN_BROKEN)

# ---------- INTERFAZ BILINGÜE ----------
with gr.Blocks(title="Semantic Density Monitor", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🌐 Semantic Density Phase Inversion Monitor

    **ΔS = S(coherent) - S(broken)** — ΔS < 0 indicates a stable representational regime.

    This tool evaluates the **structural integrity** of a language model's embedding manifold by comparing how it processes coherent versus syntactically degraded text.

    **Model:** DistilBERT (lightweight, fast)
    **Corpus:** 50 sentences per language
    **Estimated time:** ~5-10 seconds
    """)

    with gr.Tab("🇪🇸 Español"):
        with gr.Row():
            with gr.Column():
                btn_es = gr.Button("📊 Ejecutar análisis (ES)", variant="primary")
                gr.Markdown("""
                **Corpus:** 50 frases en español
                **Modelo:** DistilBERT
                **Tiempo estimado:** ~5-10 segundos
                """)
            with gr.Column():
                output_es = gr.JSON(label="Resultados (ES)")
        btn_es.click(analyze_corpus_es, inputs=[], outputs=output_es)

        with gr.Accordion("📖 Ver corpus completo (50 frases)", open=False):
            gr.Markdown("### Frases coherentes")
            for i, sent in enumerate(CORPUS_ES_COHERENT, 1):
                gr.Markdown(f"{i}. {sent}")
            gr.Markdown("### Frases rotas")
            for i, sent in enumerate(CORPUS_ES_BROKEN, 1):
                gr.Markdown(f"{i}. {sent}")

    with gr.Tab("🇬🇧 English"):
        with gr.Row():
            with gr.Column():
                btn_en = gr.Button("📊 Run analysis (EN)", variant="primary")
                gr.Markdown("""
                **Corpus:** 50 English sentences
                **Model:** DistilBERT
                **Estimated time:** ~5-10 seconds
                """)
            with gr.Column():
                output_en = gr.JSON(label="Results (EN)")
        btn_en.click(analyze_corpus_en, inputs=[], outputs=output_en)

        with gr.Accordion("📖 View full corpus (50 sentences)", open=False):
            gr.Markdown("### Coherent sentences")
            for i, sent in enumerate(CORPUS_EN_COHERENT, 1):
                gr.Markdown(f"{i}. {sent}")
            gr.Markdown("### Broken sentences")
            for i, sent in enumerate(CORPUS_EN_BROKEN, 1):
                gr.Markdown(f"{i}. {sent}")

    gr.Markdown("""
    ---
    **Reference:** Cerda Seguel, D. (2025). *The Stone Guest: Harmonic Quantization of Semantic Phase Transitions in Large Language Models*. arXiv.

    *For dynamic, real-time monitoring (Prompt → Response), see the CTAG framework.*
    """)

demo.launch()
