import streamlit as st
import pandas as pd
import plotly.graph_objects as go

# CONFIGURAÇÃO OBRIGATÓRIA COMO PRIMEIRA LINHA
st.set_page_config(page_title="PMO Dashboard", page_icon="📊", layout="wide")

# CSS PARA IFRAME
st.markdown("""
<style>
.main > div { padding-top: 0rem; }
.stApp { margin: 0; padding: 0; }
header[data-testid="stHeader"] { display: none; }
.stDecoration { display: none; }
</style>
""", unsafe_allow_html=True)

# DADOS DIRETOS (SEM CACHE PARA EVITAR ERROS)
data = {
    'id': ['1.1', '1.2', '2.1', '2.2', '2.3', '3.1'],
    'atividade': [
        'Kick-Off do Programa', 
        'Workshop EPMO',
        'Diagnóstico de Maturidade', 
        'Análise de Gaps',
        'Quick Wins',
        'Definir Sponsor'
    ],
    'status': [
        'Concluído', 
        'Concluído', 
        'Concluído', 
        'Aguardando Validação',
        'Aguardando Validação', 
        'Concluído'
    ],
    'responsavel': ['OS', 'OS', 'OS', 'DT', 'DT', 'OS'],
    'progresso': [100, 100, 100, 80, 80, 100]
}

df = pd.DataFrame(data)

# TÍTULO
st.title("📊 PMO - Digital Transformation Program")
st.markdown("**Dashboard Integrado - Iframe Ready**")

# MÉTRICAS
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Activities", len(df))

with col2:
    completed = len(df[df['status'] == 'Concluído'])
    st.metric("Completed", completed)

with col3:
    completion_rate = round((completed / len(df)) * 100)
    st.metric("Completion Rate", f"{completion_rate}%")

with col4:
    in_progress = len(df[df['status'] != 'Concluído'])
    st.metric("In Progress", in_progress)

# GRÁFICO
st.subheader("🍩 Activity Status Distribution")

status_counts = df['status'].value_counts()

colors = {
    'Concluído': '#E20074',
    'Aguardando Validação': '#F899C9',
    'Em Andamento': '#F066A7',
    'Identificado': '#8E8E93'
}

fig = go.Figure(data=[go.Pie(
    labels=status_counts.index,
    values=status_counts.values,
    hole=0.6,
    marker_colors=[colors.get(status, '#CCCCCC') for status in status_counts.index],
    textinfo='label+percent',
    textfont_size=12
)])

fig.add_annotation(
    text=f"<b>{len(df)}</b><br>ACTIVITIES",
    x=0.5, y=0.5,
    font_size=16,
    showarrow=False
)

fig.update_layout(title='Status Overview', height=400, showlegend=True)
st.plotly_chart(fig, use_container_width=True)

# TABELA
st.subheader("📋 Activity Details")

def format_status(status):
    if status == 'Concluído':
        return f"✅ {status}"
    elif status == 'Aguardando Validação':
        return f"🟡 {status}"
    else:
        return f"⚪ {status}"

df_display = df.copy()
df_display['Status Visual'] = df_display['status'].apply(format_status)

st.dataframe(
    df_display[['id', 'atividade', 'Status Visual', 'responsavel', 'progresso']], 
    use_container_width=True,
    column_config={
        'id': 'ID',
        'atividade': 'Activity',
        'Status Visual': 'Status',
        'responsavel': 'Responsible',
        'progresso': st.column_config.ProgressColumn('Progress', format='%d%%', min_value=0, max_value=100)
    },
    hide_index=True
)

# STATUS FINAL
st.success("✅ Dashboard operacional - Iframe ready")
st.info("🔗 Configurado para embedding sem autenticação")
