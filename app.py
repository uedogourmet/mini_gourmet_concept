
# app.py
import os
import json
from dataclasses import dataclass
from typing import List, Optional

import streamlit as st
from openai import OpenAI, APIError, APIConnectionError, RateLimitError

# =========================
# Config & Helpers
# =========================
st.set_page_config(page_title="Mini-Gourmet – Générateur de recettes", page_icon="🥘", layout="centered")
st.title("🥘 Mini-Gourmet – Générateur de recettes anti-gaspi")

# Récupération clé (env ou secrets Streamlit Cloud)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY", "")

# Nombre de variantes (1..3)
colA, colB = st.columns([3, 1])
with colA:
    st.markdown("Entrez les ingrédients disponibles et vos contraintes alimentaires.\nL’IA génère des recettes adaptées.")
with colB:
    num_recipes = st.slider("Variantes", min_value=1, max_value=3, value=1, help="Nombre de recettes générées")

# --------- UI Entrées ---------
ingredients_raw = st.text_area("Ingrédients (séparés par des virgules)", placeholder="Ex : tomates, oignons, riz, ail, lait, œufs", height=100)
contraintes = st.text_input("Contraintes (ex : végétarien, rapide, sans gluten)", "")
portions = st.number_input("Portions", min_value=1, max_value=8, value=2)
temps_max = st.number_input("Temps max (minutes)", min_value=5, max_value=180, value=45, step=5)
style = st.selectbox("Style de cuisine", ["libre", "française", "italienne", "méditerranéenne", "asiatique", "mexicaine"])

# Parse ingrédients utilisateur
def parse_ingredients(text: str) -> List[str]:
    return [i.strip() for i in text.split(",") if i.strip()]

user_ingredients = parse_ingredients(ingredients_raw)

# =========================
# Fallback hors-ligne
# =========================
def fallback_recipe(ings: List[str], portions: int, temps: int, style: str):
    main = ings[0] if ings else "ingrédients du placard"
    title = f"Poêlée rustique de {main} ({style})"
    qties = [f"{min(150, max(50, 30 + 10*idx))} g de {i}" for idx, i in enumerate(ings)]
    if "sel" not in " ".join(ings).lower():
        qties.append("1 pincée de sel")
    if "poivre" not in " ".join(ings).lower():
        qties.append("1 pincée de poivre")
    qties.append("1 c. à s. d'huile d'olive")

    return {
        "titre": title,
        "temps_total": f"{temps} min",
        "portions": portions,
        "ingredients": qties,
        "etapes": [
            "Préparer les ingrédients : laver, émincer si nécessaire.",
            "Chauffer l'huile dans une poêle.",
            f"Ajouter {', '.join(ings[:2]) if ings else 'les ingrédients'} et saisir 3–4 min.",
            "Cuire à feu moyen jusqu'à texture fondante, mélanger régulièrement.",
            "Assaisonner, goûter, ajuster.",
        ],
        "substitutions": [
            "Huile d’olive ↔ beurre ou huile de tournesol.",
            "Herbes fraîches ↔ herbes sèches."
        ],
    }

# =========================
# Client OpenAI (cache)
# =========================
@st.cache_resource(show_spinner=False)
def get_client(api_key: str) -> Optional[OpenAI]:
    if not api_key:
        return None
    return OpenAI(api_key=api_key)

client = get_client(OPENAI_API_KEY)

# =========================
# Schéma JSON strict (Structured Output)
# =========================
RECIPE_SCHEMA = {
    "name": "mini_gourmet_recipe_schema",
    "schema": {
        "type": "object",
        "properties": {
            "titre": {"type": "string"},
            "temps_total": {"type": "string", "description": "Ex: '35 min'"},
            "portions": {"type": "integer", "minimum": 1, "maximum": 12},
            "ingredients": {"type": "array", "items": {"type": "string"}},
            "etapes": {"type": "array", "items": {"type": "string"}},
            "substitutions": {"type": "array", "items": {"type": "string"}}
        },
        "required": ["titre", "temps_total", "portions", "ingredients", "etapes", "substitutions"],
        "additionalProperties": False
    },
    "strict": True
}

SYSTEM_PROMPT = (
    "Tu es un chef pragmatique et créatif. Réponds STRICTEMENT au format JSON demandé par le schéma. "
    "Règles importantes : "
    "- Utiliser UNIQUEMENT les ingrédients fournis par l'utilisateur (tolérance: sel, poivre, huile, eau). "
    "- Quantités réalistes en unités usuelles (g, c. à s., c. à c., pièce). "
    "- Étapes numérotées et concises. "
    "- Proposer 2 substitutions utiles. "
    "- Adapter au temps max, au style de cuisine, aux contraintes. "
    "- Pas de texte hors JSON."
)

def build_user_prompt(ings: List[str], contraintes: str, portions: int, temps: int, style: str) -> str:
    return f"""
Ingrédients disponibles: {', '.join(ings) if ings else 'aucun'}
Contraintes: {contraintes}
Style: {style}
Portions: {portions}
Temps maximum: {temps} minutes
"""

# =========================
# Appel LLM structuré (n variantes)
# =========================
def generate_recipes(ings: List[str], contraintes: str, portions: int, temps: int, style: str, n: int):
    if not client:
        # Fallback: génère n recettes simples hors-ligne
        return [fallback_recipe(ings, portions, temps, style) for _ in range(n)], None

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_prompt(ings, contraintes, portions, temps, style)}
    ]

    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",             # Excellent rapport qualité/coût/latence
            messages=messages,
            temperature=0.7,
            n=n,                              # Génère n variantes
            max_tokens=900,
            response_format={"type": "json_schema", "json_schema": RECIPE_SCHEMA},
        )
        recipes = []
        for choice in resp.choices:
            raw = choice.message.content
            try:
                data = json.loads(raw)
                recipes.append(data)
            except json.JSONDecodeError:
                # En pratique, response_format garantit un JSON valide.
                # Si cela arrive, on met la sortie brute pour debug.
                recipes.append({"titre": "Recette (JSON invalide)", "temps_total": "?", "portions": portions,
                                "ingredients": [raw], "etapes": [], "substitutions": []})

        usage = getattr(resp, "usage", None)
        return recipes, usage

    except RateLimitError:
        st.error("⏳ Trop de requêtes. Réessaie d’ici quelques secondes.")
        return [], None
    except APIConnectionError as e:
        st.error(f"Connexion API OpenAI impossible : {e}")
        return [], None
    except APIError as e:
        st.error(f"Erreur API OpenAI : {e}")
        return [], None
    except Exception as e:
        st.error(f"Erreur inattendue : {e}")
        return [], None

# =========================
# Validation : ingrédients couvrants
# =========================
def validate_ingredient_coverage(user_ings: List[str], recipe_ings: List[str]) -> List[str]:
    """Retourne la liste des ingrédients utilisateur qui ne sont pas retrouvés (approximativement) dans la recette."""
    missing = []
    concat = " ".join(recipe_ings).lower()
    for u in user_ings:
        u_low = u.lower()
        if u_low and u_low not in concat:
            # Tolérance très simple; pour mieux faire: stemming / fuzzy-match
            missing.append(u)
    return missing

# =========================
# Action : Générer
# =========================
if st.button("Générer la/les recette(s)"):
    if not user_ingredients:
        st.warning("Veuillez entrer au moins un ingrédient.")
    else:
        with st.spinner("Génération en cours..."):
            data, usage = generate_recipes(user_ingredients, contraintes, portions, temps_max, style, num_recipes)

        if not data:
            st.stop()

        for idx, rec in enumerate(data, 1):
            with st.container(border=True):
                st.subheader(f"Recette {idx} — {rec.get('titre', 'Sans titre')}")
                st.caption(f"⏱ {rec.get('temps_total','?')} • 🍽 {rec.get('portions', portions)} portions")

                st.markdown("**Ingrédients**")
                for ing in rec.get("ingredients", []):
                    st.write(f"• {ing}")

                st.markdown("**Étapes**")
                for i, step in enumerate(rec.get("etapes", []), 1):
                    st.write(f"{i}. {step}")

                subs = rec.get("substitutions", [])
                if subs:
                    st.markdown("**Substitutions**")
                    for s in subs:
                        st.write(f"• {s}")

                # Validation de couverture des ingrédients
                missing = validate_ingredient_coverage(user_ingredients, rec.get("ingredients", []))
                if missing:
                    st.warning(
                        "Certains ingrédients saisis ne figurent pas explicitement dans la recette : "
                        + ", ".join(missing)
                    )

                # Export Markdown
                md = [
                    f"# {rec.get('titre','Recette')}",
                    f"_Temps : {rec.get('temps_total','?')} • Portions : {rec.get('portions', portions)}_",
                    "\n## Ingrédients",
                ] + [f"- {x}" for x in rec.get("ingredients", [])] + [
                    "\n## Étapes",
                ] + [f"{i+1}. {s}" for i, s in enumerate(rec.get("etapes", []))] + [
                    "\n## Substitutions",
                ] + [f"- {s}" for s in subs]

                st.download_button(
                    label="💾 Télécharger (.md)",
                    data="\n".join(md),
                    file_name=f"mini_gourmet_recette_{idx}.md",
                    mime="text/markdown",
                    use_container_width=True
                )

        # Coûts (approximation via tokens)
        if usage:
            prompt_t = getattr(usage, "prompt_tokens", None)
            comp_t = getattr(usage, "completion_tokens", None)
            total_t = getattr(usage, "total_tokens", None)
            st.info(f"Usage tokens — prompt: {prompt_t}, completion: {comp_t}, total: {total_t}")

# Bandeau bas
st.divider()
if not OPENAI_API_KEY:
    st.caption("Mode démo : aucune clé OpenAI détectée → génération hors‑ligne simplifiée.")
else:
    st.caption("IA activée via OpenAI (sortie structurée JSON). Vérifie les
