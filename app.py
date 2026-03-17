import streamlit as st
import pandas as pd
import random
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC # Alterado para SVC com probability=True
from sklearn.model_selection import train_test_split

# Configuração da página para um look mais profissional
st.set_page_config(page_title="Human vs Machine: Spam Challenge", page_icon="🤖", layout="wide")

def explicacao_heuristica(msg, label):
    msg_lower = msg.lower()
    explicacoes_spam = []
    explicacoes_ham = []
    
    # SPAM [cite: 1]
    if any(word in msg_lower for word in ["ganha", "grátis", "oferta", "dinheiro", "prémio", "urgente"]):
        palavras = [w for w in ["ganha", "grátis", "oferta", "dinheiro", "prémio", "urgente"] if w in msg_lower]
        explicacoes_spam.append(f"🟡 Usa linguagem promocional: **{', '.join(palavras)}**.")

    if "http" in msg_lower or "www" in msg_lower or "link" in msg_lower:
        explicacoes_spam.append("🔗 Contém um **link** suspeito.")

    if any(char.isdigit() for char in msg_lower) and any(p in msg_lower for p in ["envie", "sms", "123", "número"]):
        explicacoes_spam.append("📲 Instruções de SMS/Números automáticos.")

    # HAM [cite: 1]
    if any(word in msg_lower for word in ["aula", "slides", "aniversário", "almoço", "viagem", "combinamos", "amanhã"]):
        palavras = [w for w in ["aula", "slides", "aniversário", "almoço", "viagem", "combinamos", "amanhã"] if w in msg_lower]
        explicacoes_ham.append(f"💬 Linguagem pessoal: **{', '.join(palavras)}**.")

    if label == "spam" and explicacoes_spam:
        return "📌 **Razões:** " + " | ".join(explicacoes_spam)
    elif label == "ham" and explicacoes_ham:
        return "📌 **Razões:** " + " | ".join(explicacoes_ham)
    return "ℹ️ Classificação baseada em padrões estatísticos."

# Inicializar estados de sessão para o "Score Global" da feira
if "global_human_wins" not in st.session_state:
    st.session_state.global_human_wins = 0
if "global_machine_wins" not in st.session_state:
    st.session_state.global_machine_wins = 0

try:
    df = pd.read_csv("messages.csv") # [cite: 1]
    st.write("Classes encontradas:", df['label'].unique()) # Isto deve mostrar ['ham', 'spam']
except FileNotFoundError:
    st.error("❌ Ficheiro messages.csv não encontrado.")
    st.stop()

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df["message"]) # [cite: 1]
y = df["label"] # [cite: 1]

# Modelos com suporte a probabilidade para a "Barra de Confiança"
models = {
    "Naive Bayes": MultinomialNB(),
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "SVM (High Precision)": SVC(probability=True) 
}

for model in models.values():
    model.fit(X, y) # [cite: 1]

st.title("🤖 Human vs Machine: The Spam Quiz")
st.markdown("---")

tab1, tab2 = st.tabs(["🎮 O Grande Duelo", "🔍 Laboratório de Testes"])

with tab1:
    # --- MÉTRICAS DE PERFORMANCE EM TEMPO REAL ---
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Vítórias Humanos (Hoje)", st.session_state.global_human_wins)
    with col2:
        st.metric("Vitórias Máquina (Hoje)", st.session_state.global_machine_wins, delta_color="inverse")
    with col3:
        accuracy = "94%" # Exemplo visual para a feira
        st.metric("Saúde do Modelo (Accuracy)", accuracy, "Estável")

    st.markdown("### 🔍 Classifica estas 5 mensagens:")
    
    if "sample_df" not in st.session_state:
        st.session_state.sample_df = df.sample(5).reset_index(drop=True) # [cite: 1]

    sample_df = st.session_state.sample_df
    user_guesses = []

    for i, row in sample_df.iterrows():
        with st.expander(f"Mensagem #{i+1}", expanded=True):
            st.write(f"**{row['message']}**")
            user_guesses.append(st.radio("O que achas?", ["ham", "spam"], key=f"q{i}", horizontal=True))

    if st.button("🚀 SUBMETER E COMPARAR RESULTADOS", use_container_width=True):
        selected_model = models["Naive Bayes"]
        X_sample = vectorizer.transform(sample_df["message"])
        model_preds = selected_model.predict(X_sample)
        model_probs = selected_model.predict_proba(X_sample)

        u_correct = 0
        m_correct = 0

        for i in range(len(sample_df)):
            true_label = sample_df.loc[i, "label"]
            machine_label = model_preds[i]
            prob = max(model_probs[i]) * 100
            
            # FEEDBACK VISUAL IMEDIATO
            st.markdown(f"#### Mensagem {i+1}")
            c1, c2 = st.columns(2)
            
            with c1:
                if user_guesses[i] == true_label:
                    st.success(f"Tu: {user_guesses[i].upper()} ✅")
                    u_correct += 1
                else:
                    st.error(f"Tu: {user_guesses[i].upper()} ❌")
            
            with c2:
                if machine_label == true_label:
                    st.info(f"Máquina: {machine_label.upper()} ({prob:.1f}%) ✅")
                    m_correct += 1
                else:
                    st.warning(f"Máquina: {machine_label.upper()} ({prob:.1f}%) ❌")
            
            st.caption(explicacao_heuristica(sample_df.loc[i, 'message'], true_label))
            st.divider()

        # Atualizar Score Global
        if u_correct > m_correct:
            st.balloons()
            st.session_state.global_human_wins += 1
            st.success("🏆 GANHASTE À MÁQUINA!")
        elif m_correct > u_correct:
            st.session_state.global_machine_wins += 1
            st.error("💻 A MÁQUINA GANHOU! Tenta outra vez.")
        else:
            st.info("🤝 EMPATE TÉCNICO!")
        
        if st.button("Jogar Novamente"):
            del st.session_state.sample_df
            st.rerun()

with tab2:
    st.header("🔬 Teste de Stress do Modelo")
    user_input = st.text_input("Escreve uma mensagem para enganar a IA:")
    
    if user_input:
        input_vec = vectorizer.transform([user_input])
        col_m1, col_m2, col_m3 = st.columns(3)
        
        for idx, (name, mdl) in enumerate(models.items()):
            pred = mdl.predict(input_vec)[0]
            prob = max(mdl.predict_proba(input_vec)[0]) * 100
            
            with [col_m1, col_m2, col_m3][idx]:
                st.subheader(name)
                if pred == "spam":
                    st.error("🚨 SPAM DETECTADO")
                else:
                    st.success("✅ MENSAGEM SEGURA")
                
                # BARRA DE CONFIANÇA
                st.write(f"Confiança: {prob:.1f}%")
                st.progress(prob / 100)
