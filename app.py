# app.py
import streamlit as st
import openai

# 🔑 Remplace 'YOUR_OPENAI_API_KEY' par ta clé OpenAI
openai.api_key = "YOUR_OPENAI_API_KEY"

st.title("Mini-Gourmet – Générateur de recettes")

st.markdown("""
Entrez les ingrédients disponibles et vos contraintes alimentaires.
L'IA générera une recette adaptée.
""")

# Formulaire d'entrée
ingredients = st.text_area("Ingrédients (séparés par des virgules)", "")
contraintes = st.text_input("Contraintes ou préférences (ex: végétarien, rapide, sans gluten)", "")

if st.button("Générer la recette"):
    if ingredients.strip() == "":
        st.warning("Veuillez entrer au moins un ingrédient.")
    else:
        prompt = f"""
        Tu es un chef créatif. Propose une recette claire et réalisable en utilisant les ingrédients suivants : 
        {ingredients}. 
        Respecte ces contraintes : {contraintes}.
        Fournis : titre de la recette, liste d'ingrédients et étapes de préparation.
        """

        try:
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500,
                temperature=0.8
            )

            recette = response['choices'][0]['message']['content']
            st.subheader("Recette générée")
            st.write(recette)

        except Exception as e:
            st.error(f"Erreur lors de la génération de la recette : {e}")
