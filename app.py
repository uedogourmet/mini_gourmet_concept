
# app.py
import os
import json
from typing import List, Optional

import streamlit as st
from openai import OpenAI, APIError, APIConnectionError, RateLimitError

# =========================
# Config & Helpers
# =========================
st.set_page_config(page_title="Mini-Gourmet – Generateur de recettes", page_icon="🥘", layout="centered")
st.title("🥘 Mini-Gourmet – Generateur de recettes anti-gaspi")

# Cle API (env ou secrets)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY", "")

# Variantes
colA, colB = st.columns([3, 1])
with colA:
    st.markdown("Entrez les ingredients disponibles et vos contraintes alimentaires.\nL'IA genere des recettes adaptees.")
with colB:
    num_recipes = st.slider("Variantes", min_value=1, max_value=3, value=1, help="Nombre de recettes generees")

# --------- UI Entrees ---------
ingredients_raw = st.text_area("Ingredients (separes par des virgules)", placeholder="Ex : tomates, oignons, riz, ail, lait, oeufs", height=100)
contraintes = st.text_input("Contraintes (ex : vegetarien, rapide, sans gluten)", "")
portions = st.number_input("Portions", min_value=1, max_value=8, value=2)
temps_max = st.number_input("Temps max (minutes)", min_value=5, max_value=180, value=45, step=5)
style = st.selectbox("Style de cuisine", ["libre", "francaise", "italienne", "mediterraneenne", "asiatique", "mexicaine"])

# Parse ingredients utilisateur
def parse_ingredients(text: str) -> List[str]:
    return [i.strip() for i in text.split(",") if i.strip()]

user_ingredients = parse_ingredients(ingredients_raw)

# =========================
# Fallback hors-ligne
# =========================
def fallback_recipe(ings: List[str], portions: int, temps: int, style: str):
    main = ings[0] if ings else "ingredients du placard"
    title = f"Poelee rustique de {main} ({style})"
    qties = [f"{min(150, max(50, 30 + 10*idx))} g de {i}" for idx, i in enumerate(ings)]
    if "sel" not in " ".join(ings).lower():
        qties.append("1 pincee de sel")
    if "poivre" not in " ".join(ings).lower():
        qties.append("1 pincee de poivre")
    qties.append("1 c. a s. d'huile d'olive")

    return {
        "titre": title,
        "temps_total": f"{temps} min",
        "portions": portions,
        "ingredients": qties,
        "etapes": [
            "Preparer les ingredients : laver, emincer si necessaire.",
            "Chauffer l'huile dans une poele.",
            f"Ajouter {', '.join(ings[:2]) if ings else 'les ingredients'} et saisir 3–4 min.",
            "Cuire a feu moyen jusqu'a texture fondante, melanger regulierement.",
            "Assaisonner, gouter, ajuster.",
        ],
        "substitutions": [
            "Huile d'olive <-> beurre ou huile de tournesol.",
            "Herbes fraiches <-> herbes seches."
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
# Schema JSON strict (Structured Output)
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
    "Tu es un chef pragmatique et creatif. Reponds STRICTEMENT au format JSON demande par le schema. "
    "Regles importantes : "
    "- Utiliser UNIQUEMENT les ingredients fournis par l'utilisateur (tolerance: sel, poivre, huile, eau). "
    "- Quantites realistes en unites usuelles (g, c. a s., c. a c., piece). "
    "- Etapes numerotees et concises. "
    "- Proposer 2 substitutions utiles. "
    "- Adapter au temps max, au style de cuisine, aux contraintes. "
    "- Pas de texte hors JSON."
)

def build_user_prompt(ings: List[str], contraintes: str, portions: int, temps: int, style: str) -> str:
    return (
        "Ingredients disponibles: " + (", ".join(ings) if ings else "aucun") + "\n" +
        "Contraintes: " + contraintes + "\n" +
        "Style: " + style + "\n" +
        "Portions: " + str(portions) + "\n" +
        "Temps maximum: " + str(temps) + " minutes\n"
    )

# =========================
# Appel LLM structure (n variantes)
# =========================
def generate_recipes(ings: List[str], contraintes: str, portions: int, temps: int, style: str, n: int):
    if not client:
        return [fallback_recipe(ings, portions, temps, style) for _ in range(n)], None

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_prompt(ings, contraintes, portions, temps, style)}
    ]

    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.7,
            n=n,
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
                recipes.append({"titre": "Recette (JSON invalide)", "temps_total": "?", "portions": portions,
                                "ingredients": [raw], "etapes": [], "substitutions": []})

        usage = getattr(resp, "usage", None)
        return recipes, usage

    except RateLimitError:
        st.error("Trop de requetes. Reessaie dans un instant.")
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
# Validation : ingredients couvrants
# =========================
def validate_ingredient_coverage(user_ings: List[str], recipe_ings: List[str]) -> List[str]:
    missing = []
    concat = " ".join(recipe_ings).lower()
    for u in user_ings:
        u_low = u.lower()
        if u_low and u_low not in concat:
            missing.append(u)
    return missing

# =========================
# Action : Generer
# =========================
if st.button("Generer la/les recette(s)"):
    if not user_ingredients:
        st.warning("Veuillez entrer au moins un ingredient.")
    else:
        with st.spinner("Generation en cours..."):
            data, usage = generate_recipes(user_ingredients, contraintes, portions, temps_max, style, num_recipes)

        if not data:
            st.stop()

        for idx, rec in enumerate(data, 1):
            with st.container(border=True):
                st.subheader(f"Recette {idx} — {rec.get('titre', 'Sans titre')}")
                st.caption(f"Temps {rec.get('temps_total','?')} • Portions {rec.get('portions', portions)}")

                st.markdown("**Ingredients**")
                for ing in rec.get("ingredients", []):
                    st.write(f"• {ing}")

                st.markdown("**Etapes**")
                for i, step in enumerate(rec.get("etapes", []), 1):
                    st.write(f"{i}. {step}")

                subs = rec.get("substitutions", [])
                if subs:
                    st.markdown("**Substitutions**")
                    for s in subs:
                        st.write(f"• {s}")

                missing = validate_ingredient_coverage(user_ingredients, rec.get("ingredients", []))
                if missing:
                    st.warning("Ingredients saisis non retrouves explicitement : " + ", ".join(missing))

                # Export Markdown - parenthèses et virgules vérifiées
                md = [
                    f"# {rec.get('titre','Recette')}",
                    f"_Temps : {rec.get('temps_total','?')} • Portions : {rec.get('portions', portions)}_",
                    "\n## Ingredients",
                ] + [f"- {x}" for x in rec.get("ingredients", [])] + [
                    "\n## Etapes",
                ] + [f"{i+1}. {s}" for i, s in enumerate(rec.get("etapes", []))] + [
                    "\n## Substitutions",
                ] + [f"- {s}" for s in subs]

                st.download_button(
                    label="Telecharger (.md)",
                    data="\n".join(md),
                    file_name=f"mini_gourmet_recette_{idx}.md",
                    mime="text/markdown",
                    use_container_width=True
                )

        if usage:
            prompt_t = getattr(usage, "prompt_tokens", None)
            comp_t = getattr(usage, "completion_tokens", None)
            total_t = getattr(usage, "total_tokens", None)
            st.info(f"Usage tokens — prompt: {prompt_t}, completion: {comp_t}, total: {total_t}")

# Bandeau bas
st.divider()
if not OPENAI_API_KEY:
    st.caption("Mode demo : aucune    st.caption("Mode demo : aucune cle OpenAI detectee -> generation hors-ligne simplifiee.")
else:
