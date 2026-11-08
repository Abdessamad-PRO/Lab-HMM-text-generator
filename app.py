"""Lab 03 - Chaotic Text Generator (Streamlit UI). Run: streamlit run app.py"""
import random
import pandas as pd
import streamlit as st

from src.pipeline import (HMM, NGramGenerator, load_tagged, split_by_document, 
                         generation_metrics, sentence_features, oov_rate)

st.set_page_config(page_title="Chaotic Text Generator", layout="wide")
st.title("Chaotic Text Generator")

# Notice visible
st.warning("The corpus is synthetic and the grammar is tiny. Metrics such as valid tag pattern are defined relative to this corpus and do not measure the quality of real English.")

@st.cache_resource
def get_models():
    docs = load_tagged()
    train, _ = split_by_document(docs)
    return HMM(train), HMM(train, split_states=True), NGramGenerator([w for w, _ in train])

hmm_plain, hmm_split, ng = get_models()

# Tabs pour différentes fonctionnalités
tab1, tab2 = st.tabs(["Generator", "Tagger"])

with tab1:
    # Panneau de contrôle
    st.header("Text Generator")
    c1, c2, c3, c4 = st.columns(4)
    model_type = c1.selectbox("Model", ["Plain HMM", "Role-split HMM", "Trigram"])
    temperature = c2.slider("Temperature", 0.3, 3.0, 1.0, 0.1)
    seed = c3.number_input("Random seed", 0, 10000, 42)
    n_sentences = c4.number_input("Number of sentences", 10, 500, 50)
    
    if st.button("Generate", type="primary"):
        rng = random.Random(int(seed))
        model = hmm_split if model_type == "Role-split HMM" else (ng if model_type == "Trigram" else hmm_plain)
        
        samples = []
        for _ in range(n_sentences):
            if model_type == "Trigram":
                words = model.sample_sentence(rng, temperature)
                tags = hmm_plain.viterbi(words)
                samples.append((words, tags))
            else:
                words, tags = model.sample_sentence(rng, temperature)
                samples.append((words, tags))
        
        # Calculer les métriques
        # train = [x for x, _ in load_tagged().values()]
        train, _ = split_by_document(load_tagged())
        metrics = generation_metrics(samples, train, hmm_plain)
        
        # Afficher les métriques
        st.subheader("Metrics")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Valid tag pattern", f"{metrics['valid_tag_pattern']:.1%}")
        m2.metric("Plausible subject-verb", f"{metrics['plausible_subject_verb']:.1%}")
        m3.metric("Novel sentences", f"{metrics['novel_sentences']:.1%}")
        m4.metric("Distinct-2", f"{metrics['distinct_2']:.3f}")
        
        # Tableau des phrases
        st.subheader("Generated Sentences")
        train_patterns = {" ".join(t) for _, t in train}
        table_data = []
        for words, tags in samples:
            pattern = " ".join(tags)
            valid = "✓" if pattern in train_patterns else "✗"
            table_data.append({
                "Words": " ".join(words),
                "Tags": " ".join(tags),
                "Valid": valid
            })
        st.dataframe(pd.DataFrame(table_data, index=range(1, len(table_data) + 1)))
        
        # Download CSV
        csv_data = pd.DataFrame(table_data).to_csv(index=False)
        st.download_button("Download CSV", csv_data, "generated_sentences.csv", "text/csv")

with tab2:
    # Tagger
    st.header("POS Tagger (Viterbi)")
    sentence = st.text_input("Enter a sentence:", "the cats sleep in the garden")
    
    if sentence:
        words = sentence.lower().split()
        # Vérifier OOV
        unknown = [w for w in words if w not in hmm_plain.vocab]
        if unknown:
            st.error(f"Unknown word(s): {', '.join(unknown)}")
        
        try:
            tags = hmm_plain.viterbi(words)
            # Afficher mot/tag alignés
            cols = st.columns(len(words))
            for col, w, t in zip(cols, words, tags):
                col.write(f"**{w}**")
                col.caption(t)
        except Exception as e:
            st.error(f"Tagging error: {e}")