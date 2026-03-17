import streamlit as st
import pandas as pd
import random
import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

# --- CONFIGURAÇÃO E PERSISTÊNCIA ---
st.set_page_config(page_title="Human vs Machine Challenge", page_icon="🤖", layout="wide")

def carregar_stats():
    """Lê as vitórias acumuladas de um ficheiro CSV local."""
    if not os.path.exists("stats.csv"):
        return 0, 0
    try:
        stats_df = pd.read_csv("stats.csv")
        return int(stats_df["human_wins"].iloc[0]), int(stats_df["machine_wins"].iloc[0])
    except:
        return 0, 0

def salvar_stats(h_wins, m_wins):
    """Guarda as vitórias acumuladas num ficheiro CSV local."""
    stats_df = pd.DataFrame({"human_wins": [h_wins], "machine_wins": [m_wins]})
    stats_df.to_csv("stats.csv", index=False)

def explicacao_heuristica(msg, label):
    msg_lower = msg.lower()
    explicacoes = []
    
    if label == "spam":
        if any(w in msg_lower for w in ["ganha", "grátis", "oferta", "dinheiro", "prémio", "urgente"]):
            explicacoes.append("🟡 Linguagem promocional detetada")
        if "http" in msg_lower or "www" in msg_lower:
            explicacoes.append("🔗 Link suspeito encontrado")
        if any(char.isdigit() for char in msg_lower) and "sms" in msg_lower:
            explicacoes.append("📲 Instrução de SMS automático")
    else:
        if any(w in msg_lower for w in ["aula", "almoço", "viagem", "combinamos", "amanhã"]):
            explicacoes.append("💬 Linguagem pessoal/quotidiana")

    return " | ".join(explicacoes) if explicacoes else "ℹ️ Padrão estatístico identificado pelo modelo"

# --- INICIALIZAÇÃO DE DADOS ---
try:
    df = pd.read_csv("messages.csv")
except FileNotFoundError:
    st.error("❌ Erro: O ficheiro 'messages.csv' não foi encontrado!")
    st.stop()

# Inicializar vetores e modelos
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df["message"])
y = df["label"]

# Modelos com suporte a probabilidade para a barra de confiança
models = {
    "Naive Bayes": MultinomialNB(),
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "SVM (High Precision)": SVC(probability=True)
}

for mdl in models.values():
    mdl.fit(X, y)

# Carregar estatísticas persistentes
h_inicial, m_inicial = carregar_stats()
if "global_human_wins" not in st.session_state:
    st.session_state.global_human_wins = h_inicial
if "global_machine_wins" not in st.session_state:
    st.session_state.global_machine_wins = m_inicial

# --- INTERFACE ---
st.title("🤖 Human vs Machine: O Quiz do Spam")
st.markdown("Testa os teus instintos contra a Inteligência Artificial!")
st.divider()

tab1, tab2 = st.tabs(["🎮 O Grande Duelo", "🔬 Laboratório de Testes"])

with tab1:
    # Métricas de topo (Gamificação)
    m1, m2, m3 = st.columns(3)
    m1.metric("Vitórias Humanos", st.session_state.global_human_wins)
    m2.metric("Vitórias Máquina", st.session_state.global_machine_wins, delta_color="inverse")
    m3.metric("Modelo de IA", "Ativo", "Estável")

    if "sample_df" not in st.session_state:
        st.session_state.sample_df = df.sample(5).reset_index(drop=True)

    sample_df = st.session_state.sample_df
    user_guesses = []

    st.subheader("Classifica as seguintes mensagens:")
    for i, row in sample_df.iterrows():
        with st.expander(f"Mensagem #{i+1}", expanded=True):
            st.write(f"**{row['message']}**")
            user_guesses.append(st.radio("É Spam?", ["ham", "spam"], key=f"user_q_{i}", horizontal=True))

    if st.button("🚀 SUBMETER RESPOSTAS", use_container_width=True):
        # Usamos o Naive Bayes como o modelo padrão para o quiz
        current_model = models["Naive Bayes"]
        X_sample = vectorizer.transform(sample_df["message"])
        model_preds = current_model.predict(X_sample)
        model_probs = current_model.predict_proba(X_sample)

        u_correct = 0
        m_correct = 0

        st.divider()
        for i in range(len(sample_df)):
            true_l = sample_df.loc[i, "label"]
            machine_l = model_preds[i]
            conf = max(model_probs[i]) * 100
            
            c1, c2 = st.columns(2)
            with c1:
                if user_guesses[i] == true_l:
                    st.success(f"Tu: {user_guesses[i].upper()} ✅")
                    u_correct += 1
                else:
                    st.error(f"Tu: {user_guesses[i].upper()} ❌")
            with c2:
                icon = "✅" if machine_l == true_l else "❌"
                st.info(f"Máquina: {machine_l.upper()} ({conf:.1f}%) {icon}")
                if machine_l == true_l: m_correct += 1
            
            st.caption(f"_{explicacao_heuristica(sample_df.loc[i, 'message'], true_l)}_")
            st.divider()

        # Atualizar e salvar recordes
        if u_correct > m_correct:
            st.balloons()
            st.session_state.global_human_wins += 1
            st.success("🏆 VITÓRIA HUMANA! Superaste o algoritmo.")
        elif m_correct > u_correct:
            st.error("💻 A MÁQUINA VENCEU! O algoritmo foi mais preciso.")
            st.session_state.global_machine_wins += 1
        else:
            st.warning("🤝 EMPATE! Estás ao nível da máquina.")
        
        salvar_stats(st.session_state.global_human_wins, st.session_state.global_machine_wins)
        
        if st.button("Jogar Nova Ronda"):
            del st.session_state.sample_df
            st.rerun()

with tab2:
    st.header("🔬 Laboratório de Testes")
    st.markdown("Escreve a tua própria mensagem para ver a confiança de cada modelo.")
    
    test_input = st.text_input("Mensagem personalizada:")
    if test_input:
        input_vec = vectorizer.transform([test_input])
        cols = st.columns(len(models))
        
        for idx, (name, mdl) in enumerate(models.items()):
            with cols[idx]:
                pred = mdl.predict(input_vec)[0]
                prob = max(mdl.predict_proba(input_vec)[0]) * 100
                st.subheader(name)
                if pred == "spam":
                    st.error("🚨 SPAM")
                else:
                    st.success("✅ LEGÍTIMA")
                st.write(f"Confiança: {prob:.1f}%")
                st.progress(prob / 100)
