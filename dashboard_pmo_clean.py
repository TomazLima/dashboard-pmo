import streamlit as st
import pandas as pd
import plotly.graph_objects as go

# 🚨 PRIMEIRA LINHA OBRIGATÓRIA - CONFIGURAÇÃO PARA IFRAME
st.set_page_config(
    page_title="PMO Dashboard", 
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ✅ CONFIGURAÇÕES PARA IFRAME EMBEDDING
st.markdown("""
<style>
    /* Otimizar para iframe */
    .main > div {
        padding-top: 0rem;
        padding-bottom: 0rem;
    }
    
    .stApp {
        margin: 0;
        padding: 0;
    }
    
    /* Esconder elementos desnecessários no iframe */
    header[data-testid="stHeader"] {
        display: none;
    }
    
    .stDecoration {
        display: none;
    }
    
    /* Melhorar responsividade no iframe */
    .block-container {
        padding-top: 1rem;
        padding-bottom: 0rem;
        max-width: none;
    }
</style>
""", unsafe_allow_html=True)

# DADOS EMBUTIDOS NO CÓDIGO - SEM DEPENDÊNCIA DE ARQUIVOS
@st.cache_data
def create_data():
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
    return pd.DataFrame(data)

# CARREGAR DADOS
df = create_data()

# TÍTULO PRINCIPAL
st.title("📊 PMO - Digital Transformation Program")
st.markdown("**Dashboard Integrado - Versão Iframe-Ready**")

# MÉTRICAS PRINCIPAIS
col1, col2, col3, col4 = st.columns(4)

with col1:
    total_activities = len(df)
    st.metric("Total Activities", total_activities)

with col2:
    completed = len(df[df['status'] == 'Concluído'])
    st.metric("Completed", completed)

with col3:
    completion_rate = round((completed / total_activities) * 100)
    st.metric("Completion Rate", f"{completion_rate}%")

with col4:
    in_progress = len(df[df['status'] != 'Concluído'])
    st.metric("In Progress", in_progress)

# SEPARADOR
st.markdown("---")

# GRÁFICO DE STATUS
st.subheader("🍩 Activity Status Distribution")

status_counts = df['status'].value_counts()

# CORES PARA CADA STATUS (MESMAS DO DASHBOARD ORIGINAL)
colors = {
    'Concluído': '#E20074',
    'Aguardando Validação': '#F899C9',
    'Em Andamento': '#F066A7',
    'Identificado': '#8E8E93'
}

# CRIAR GRÁFICO DE PIZZA
fig = go.Figure(data=[go.Pie(
    labels=status_counts.index,
    values=status_counts.values,
    hole=0.6,
    marker_colors=[colors.get(status, '#CCCCCC') for status in status_counts.index],
    textinfo='label+percent',
    textfont_size=12,
    pull=[0.05 if status == 'Concluído' else 0.02 for status in status_counts.index]
)])

# TEXTO CENTRAL
fig.add_annotation(
    text=f"<b style='font-size:20px'>{len(df)}</b><br><span style='font-size:12px'>ACTIVITIES</span>",
    x=0.5, y=0.5,
    font_size=16,
    showarrow=False
)

fig.update_layout(
    title='Status Overview',
    height=400,
    showlegend=True,
    margin=dict(t=60, b=60, l=60, r=60)
)

st.plotly_chart(fig, use_container_width=True)

# TABELA DE ATIVIDADES
st.subheader("📋 Activity Details")

# ADICIONAR COLUNA DE STATUS COM CORES
def format_status(status):
    if status == 'Concluído':
        return f"✅ {status}"
    elif status == 'Aguardando Validação':
        return f"🟡 {status}"
    elif status == 'Em Andamento':
        return f"🔄 {status}"
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
        'progresso': st.column_config.ProgressColumn(
            'Progress',
            help='Completion percentage',
            format='%d%%',
            min_value=0,
            max_value=100
        )
    },
    hide_index=True
)

# RESUMO FINAL
st.markdown("---")

# INFORMAÇÕES DE STATUS PARA IFRAME
col1, col2 = st.columns(2)

with col1:
    st.success("✅ **Dashboard operacional** - Configurado para iframe embedding")

with col2:
    # DETECTAR SE ESTÁ EM IFRAME
    st.markdown("""
    <script>
    if (window.parent !== window) {
        console.log('✅ Dashboard rodando em iframe');
    } else {
        console.log('📱 Dashboard rodando standalone');
    }
    </script>
    """, unsafe_allow_html=True)
    
    st.info("🔗 **Iframe-ready** - Sem autenticação")

# INFORMAÇÕES TÉCNICAS (EXPANSÍVEL)
with st.expander("ℹ️ Technical Info"):
    st.write("📊 **Data Source:** Embedded in code")
    st.write("🔐 **Authentication:** None (iframe-ready)")
    st.write("🚀 **Status:** Fully operational")
    st.write("🖼️ **Iframe Support:** Enabled")
    st.write(f"📅 **Last Updated:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")