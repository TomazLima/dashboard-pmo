import streamlit as st
import hashlib

def make_hashes(password):
    return hashlib.sha256(str.encode(password)).hexdigest()

def check_hashes(password, hashed_text):
    if make_hashes(password) == hashed_text:
        return hashed_text
    return False

def login_user():
    """
    Função para autenticação simples por senha
    """
    
    # Hash da senha (substitua 'sua_senha_aqui' pela senha desejada)
    # Para gerar o hash: print(make_hashes("sua_senha_aqui"))
    SENHA_HASH = "56ad85dd8e1b8e90af1d2dece16548897635a3f1f288c8d2042f6a68f9404196"  # Você vai gerar isso
    
    if 'authentication_status' not in st.session_state:
        st.session_state['authentication_status'] = False
    
    if not st.session_state['authentication_status']:
        st.title("🔐 Login Dashboard PMO")
        
        with st.form("login_form"):
            senha = st.text_input("Senha:", type="password")
            submit_button = st.form_submit_button("Entrar")
            
            if submit_button:
                if check_hashes(senha, SENHA_HASH):
                    st.session_state['authentication_status'] = True
                    st.success("Login realizado com sucesso!")
                    st.rerun()
                else:
                    st.error("Senha incorreta!")
        
        return False
    
    return True

def logout():
    """Função para logout"""
    if st.button("🚪 Logout"):
        st.session_state['authentication_status'] = False
        st.rerun()
