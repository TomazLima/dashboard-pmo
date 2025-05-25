import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime
import openpyxl
import requests
import json
import os

# ============================================
# 🔧 CONFIGURAÇÃO DA PÁGINA
# ============================================

st.set_page_config(
    page_title="PMO Digital Telco - Llama",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================
# 🦙 CONFIGURAÇÃO GROQ API (LLAMA 3.1) - MUDANÇA PRINCIPAL
# ============================================

def configurar_llama_api():
    """Configuração da API do Groq (Llama 3.1) - GRATUITA!"""
    st.sidebar.markdown("---")
    st.sidebar.subheader("🦙 Configuração Llama 3.3")
    
    # Verificar se existe API key salva
    # OPÇÃO: Colocar sua API key diretamente aqui (NÃO RECOMENDADO para produção)
    # api_key_default = 'gsk-sua-api-key-aqui'
    api_key_default = os.getenv('GROQ_API_KEY', '')
    
    api_key = st.sidebar.text_input(
        "API Key do Groq:",
        value=api_key_default,
        type="password",
        help="Sua chave GRATUITA da API do Groq (gsk_...)"
    )
    
    # Toggle para análise online/offline
    usar_llama_online = st.sidebar.toggle(
        "🌐 Usar Llama Online",
        value=bool(api_key),
        help="Ativar análise inteligente com Llama 3.3"
    )
    
    # Status da configuração
    if usar_llama_online and api_key:
        st.sidebar.success("🟢 Llama 3.1 Online Ativo")
        st.sidebar.metric("Custo/Análise", "GRATUITO 🎉")
    elif usar_llama_online and not api_key:
        st.sidebar.error("🔴 API Key necessária")
        st.sidebar.info("💡 Registre-se em console.groq.com")
    else:
        st.sidebar.info("🔵 Modo Offline")
    
    return api_key if usar_llama_online else None

# ============================================
# 📊 DADOS DO PROJETO PMO ONSET (IGUAL AO ORIGINAL)
# ============================================

@st.cache_data
def carregar_dados():
    """Carrega dados do arquivo Excel"""
    try:
        # Lê arquivo Excel
        df = pd.read_excel('atividades_pmo.xlsx', sheet_name='Atividades')
        
        # Validar colunas obrigatórias
        colunas_obrigatorias = ['id', 'fase', 'atividade', 'status', 'responsavel', 'progresso', 'peso']
        if not all(col in df.columns for col in colunas_obrigatorias):
            st.error("❌ Arquivo Excel não tem as colunas corretas!")
            return pd.DataFrame()
        
        return df
        
    except FileNotFoundError:
        st.error("❌ Arquivo 'atividades_pmo.xlsx' não encontrado!")
        st.info("💡 Coloque o arquivo na mesma pasta que dashboard_pmo_v2.py")
        return pd.DataFrame()
        
    except Exception as e:
        st.error(f"❌ Erro ao ler Excel: {e}")
        # Em caso de erro, usar dados padrão
        return carregar_dados_padrao()

def carregar_dados_padrao():
    """Dados padrão caso Excel não funcione"""
    dados_pmo = [
        {"id": "1.1", "fase": "Marco Inicial", "atividade": "Kick-Off do Programa Digital Telco", "status": "Concluído", "responsavel": "OnSet", "progresso": 100, "peso": 1.0},
        {"id": "1.2", "fase": "Marco Inicial", "atividade": "Definição do Escopo Geral", "status": "Concluído", "responsavel": "OnSet", "progresso": 100, "peso": 1.0},
        {"id": "2.1", "fase": "Fase 1 - Estruturação", "atividade": "Estruturação da Governança PMO", "status": "Em Andamento", "responsavel": "OnSet", "progresso": 70, "peso": 0.7},
        {"id": "2.2", "fase": "Fase 1 - Estruturação", "atividade": "Definição de Metodologias", "status": "Em Andamento", "responsavel": "OnSet", "progresso": 60, "peso": 0.6},
        {"id": "2.3", "fase": "Fase 1 - Estruturação", "atividade": "Estruturação de Ferramentas", "status": "Aguardando Validação", "responsavel": "Deutsche", "progresso": 80, "peso": 0.8},
        {"id": "3.1", "fase": "Fase 3 - Implantação", "atividade": "Implantação do PMO", "status": "Não Definido", "responsavel": "A Definir", "progresso": 0, "peso": 0.0},
        {"id": "3.2", "fase": "Fase 3 - Implantação", "atividade": "Treinamento das Equipes", "status": "Não Definido", "responsavel": "A Definir", "progresso": 0, "peso": 0.0},
    ]
    return pd.DataFrame(dados_pmo)

# ============================================
# 📊 CALCULAR MÉTRICAS (IGUAL AO ORIGINAL)
# ============================================

def calcular_metricas(df):
    """Calcula todas as métricas do projeto"""
    
    if df.empty:
        return {
            'total_atividades': 0,
            'conclusao_geral': 0,
            'status_counts': pd.Series(dtype=int),
            'fase_stats': pd.DataFrame(),
            'resp_stats': pd.DataFrame()
        }
    
    total_atividades = len(df)
    conclusao_geral = round((df['peso'].sum() / total_atividades) * 100)
    
    # Contagem por status
    status_counts = df['status'].value_counts()
    
    # Análise por fase
    fase_stats = df.groupby('fase').agg({
        'peso': ['sum', 'count'],
        'progresso': 'mean'
    }).round(1)
    
    fase_stats.columns = ['peso_total', 'quantidade', 'progresso_medio']
    fase_stats['conclusao_fase'] = (fase_stats['peso_total'] / fase_stats['quantidade'] * 100).round(1)
    
    # Análise por responsável
    resp_stats = df.groupby(['responsavel', 'status']).size().unstack(fill_value=0)
    
    return {
        'total_atividades': total_atividades,
        'conclusao_geral': conclusao_geral,
        'status_counts': status_counts,
        'fase_stats': fase_stats,
        'resp_stats': resp_stats
    }

# ============================================
# 🎨 SIDEBAR (ADAPTADA PARA LLAMA)
# ============================================

def criar_sidebar(df, metricas):
    """Cria a sidebar com filtros e informações"""
    
    st.sidebar.title("🎯 PMO Digital Telco")
    st.sidebar.markdown("**Programa Digital Telco + Llama 3.3**")
    
    # Configuração Llama API
    api_key = configurar_llama_api()
    
    st.sidebar.markdown("---")
    
    # Métricas principais
    st.sidebar.metric("Total de Atividades", metricas['total_atividades'])
    st.sidebar.metric(
        "Conclusão Geral", 
        f"{metricas['conclusao_geral']}%",
        delta=f"{metricas['conclusao_geral'] - 50}% vs Meta 50%"
    )
    
    st.sidebar.markdown("---")
    
    # Filtros
    st.sidebar.subheader("🔍 Filtros")
    
    if df.empty:
        st.sidebar.warning("⚠️ Nenhum dado disponível")
        return df, api_key
    
    fases_selecionadas = st.sidebar.multiselect(
        "Selecionar Fases:",
        options=df['fase'].unique(),
        default=df['fase'].unique()
    )
    
    status_selecionados = st.sidebar.multiselect(
        "Selecionar Status:",
        options=df['status'].unique(),
        default=df['status'].unique()
    )
    
    responsaveis_selecionados = st.sidebar.multiselect(
        "Selecionar Responsáveis:",
        options=df['responsavel'].unique(),
        default=df['responsavel'].unique()
    )
    
    # Aplicar filtros
    df_filtrado = df[
        (df['fase'].isin(fases_selecionadas)) &
        (df['status'].isin(status_selecionados)) &
        (df['responsavel'].isin(responsaveis_selecionados))
    ]
    
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"**Atualizado:** {datetime.now().strftime('%d/%m/%Y %H:%M')}")
    
    return df_filtrado, api_key

# ============================================
# 📊 GRÁFICOS (IGUAIS AO ORIGINAL)
# ============================================

def criar_rosca_status(df):
    """Cria gráfico de rosca para status"""
    
    if df.empty:
        return go.Figure().add_annotation(text="Nenhum dado disponível", x=0.5, y=0.5)
    
    status_counts = df['status'].value_counts()
    
    cores_status = {
        'Concluído': '#00C851',
        'Em Andamento': '#2196F3',
        'Aguardando Validação': '#FF9800',
        'Não Definido': '#F44336'
    }
    
    colors = [cores_status.get(status, '#CCCCCC') for status in status_counts.index]
    
    fig = go.Figure(data=[go.Pie(
        labels=status_counts.index,
        values=status_counts.values,
        hole=0.4,
        marker_colors=colors,
        textinfo='label+percent+value',
        textfont_size=12,
        hovertemplate='<b>%{label}</b><br>Quantidade: %{value}<br>Percentual: %{percent}<extra></extra>'
    )])
    
    # Texto no centro
    total = len(df)
    conclusao = round((df['peso'].sum() / total) * 100)
    
    fig.add_annotation(
        text=f"<b>{total}</b><br>ATIVIDADES<br><br><b>{conclusao}%</b><br>CONCLUSÃO",
        x=0.5, y=0.5,
        font_size=16,
        showarrow=False
    )
    
    fig.update_layout(
        title='🍩 Status das Atividades',
        showlegend=True,
        height=400,
        margin=dict(t=60, b=40, l=40, r=40)
    )
    
    return fig

def criar_barras_fase(df):
    """Cria gráfico de barras por fase"""
    
    if df.empty:
        return go.Figure().add_annotation(text="Nenhum dado disponível", x=0.5, y=0.5)
    
    fase_stats = df.groupby('fase').agg({
        'peso': ['sum', 'count']
    }).round(1)
    
    fase_stats.columns = ['peso_total', 'quantidade']
    fase_stats['conclusao_fase'] = (fase_stats['peso_total'] / fase_stats['quantidade'] * 100).round(1)
    
    # Cores por nível
    cores = []
    for conclusao in fase_stats['conclusao_fase']:
        if conclusao >= 70:
            cores.append('#00C851')  # Verde
        elif conclusao >= 30:
            cores.append('#FF9800')  # Amarelo
        else:
            cores.append('#F44336')  # Vermelho
    
    fig = go.Figure([go.Bar(
        x=fase_stats['conclusao_fase'],
        y=fase_stats.index,
        orientation='h',
        marker_color=cores,
        text=[f'{c:.0f}%' for c in fase_stats['conclusao_fase']],
        textposition='inside'
    )])
    
    fig.update_layout(
        title='📊 Progresso por Fase',
        xaxis_title='Percentual de Conclusão (%)',
        height=400,
        margin=dict(t=60, b=40, l=200, r=40)
    )
    
    fig.add_vline(x=50, line_dash="dash", line_color="gray", 
                  annotation_text="Meta 50%")
    
    return fig

def criar_velocimetro(df):
    """Cria velocímetro de conclusão"""
    
    if df.empty:
        conclusao = 0
    else:
        total = len(df)
        conclusao = round((df['peso'].sum() / total) * 100)
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=conclusao,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "🎯 Conclusão Geral", 'font': {'size': 20}},
        delta={'reference': 50, 'increasing': {'color': "green"}, 'decreasing': {'color': "red"}},
        gauge={
            'axis': {'range': [None, 100], 'tickwidth': 1},
            'bar': {'color': "#1f77b4"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 30], 'color': '#ffcccc'},
                {'range': [30, 70], 'color': '#fff3cd'},
                {'range': [70, 100], 'color': '#d4edda'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    
    fig.update_layout(height=400)
    return fig

def criar_distribuicao_responsavel(df):
    """Cria distribuição por responsável"""
    
    if df.empty:
        return go.Figure().add_annotation(text="Nenhum dado disponível", x=0.5, y=0.5)
    
    resp_status = df.groupby(['responsavel', 'status']).size().unstack(fill_value=0)
    
    cores_status = {
        'Concluído': '#00C851',
        'Em Andamento': '#2196F3',
        'Aguardando Validação': '#FF9800',
        'Não Definido': '#F44336'
    }
    
    fig = go.Figure()
    
    for status in resp_status.columns:
        fig.add_trace(go.Bar(
            name=status,
            x=resp_status.index,
            y=resp_status[status],
            marker_color=cores_status.get(status, '#CCCCCC')
        ))
    
    fig.update_layout(
        title='👥 Distribuição por Responsável',
        barmode='stack',
        height=400,
        legend=dict(orientation="h", x=0, y=1.02)
    )
    
    return fig

# ============================================
# 📋 TABELA DE ATIVIDADES (IGUAL AO ORIGINAL)
# ============================================

def criar_tabela_atividades(df):
    """Cria tabela interativa de atividades"""
    
    if df.empty:
        st.warning("⚠️ Nenhuma atividade encontrada com os filtros aplicados")
        return
    
    # Preparar dados para exibição
    df_tabela = df[['id', 'fase', 'atividade', 'status', 'responsavel', 'progresso']].copy()
    
    # Adicionar emoji por status
    emoji_status = {
        'Concluído': '✅',
        'Em Andamento': '🔄',
        'Aguardando Validação': '⏳',
        'Não Definido': '❌'
    }
    
    df_tabela['status_emoji'] = df_tabela['status'].map(emoji_status) + ' ' + df_tabela['status']
    
    # Exibir tabela
    st.dataframe(
        df_tabela[['id', 'fase', 'atividade', 'status_emoji', 'responsavel', 'progresso']],
        column_config={
            'id': 'ID',
            'fase': 'Fase',
            'atividade': 'Atividade',
            'status_emoji': 'Status',
            'responsavel': 'Responsável',
            'progresso': st.column_config.ProgressColumn(
                'Progresso',
                help='Percentual de conclusão',
                format='%d%%',
                min_value=0,
                max_value=100
            )
        },
        hide_index=True,
        use_container_width=True
    )

# ============================================
# 🚨 ALERTAS E INSIGHTS (IGUAL AO ORIGINAL)
# ============================================

def criar_alertas(df, metricas):
    """Cria seção de alertas e insights"""
    
    st.subheader("🚨 Alertas Críticos")
    
    if df.empty:
        st.warning("⚠️ Nenhum dado para análise de alertas")
        return
    
    col1, col2, col3 = st.columns(3)
    
    # Alerta 1: Atividades não definidas
    nao_definidas = metricas['status_counts'].get('Não Definido', 0)
    pct_nao_definidas = round((nao_definidas / metricas['total_atividades']) * 100)
    
    with col1:
        if pct_nao_definidas > 50:
            st.error(f"🔴 **{pct_nao_definidas}% Não Definidas**\n\n{nao_definidas} atividades precisam de estruturação urgente")
        elif pct_nao_definidas > 30:
            st.warning(f"🟡 **{pct_nao_definidas}% Não Definidas**\n\n{nao_definidas} atividades precisam de atenção")
        else:
            st.success(f"🟢 **{pct_nao_definidas}% Não Definidas**\n\nSituação controlada")
    
    # Alerta 2: Validações pendentes
    validacoes = metricas['status_counts'].get('Aguardando Validação', 0)
    
    with col2:
        if validacoes > 5:
            st.error(f"🔴 **{validacoes} Validações Pendentes**\n\nDeutsche com itens pendentes")
        elif validacoes > 2:
            st.warning(f"🟡 **{validacoes} Validações Pendentes**\n\nAcompanhar prazos de aprovação")
        else:
            st.success(f"🟢 **{validacoes} Validações Pendentes**\n\nFluxo de aprovação normal")
    
    # Alerta 3: Fase crítica (usando primeira fase disponível se Fase 3 não existir)
    fase_critica = None
    if 'Fase 3 - Implantação' in metricas['fase_stats'].index:
        fase_critica = metricas['fase_stats'].loc['Fase 3 - Implantação', 'conclusao_fase']
        fase_nome = "Fase 3"
    elif len(metricas['fase_stats']) > 0:
        fase_critica = metricas['fase_stats']['conclusao_fase'].min()
        fase_nome = "Fase Crítica"
    
    with col3:
        if fase_critica is not None:
            if fase_critica < 20:
                st.error(f"🔴 **{fase_nome}: {fase_critica:.0f}%**\n\nRisco crítico de atraso")
            elif fase_critica < 50:
                st.warning(f"🟡 **{fase_nome}: {fase_critica:.0f}%**\n\nProgressão lenta - acompanhar")
            else:
                st.success(f"🟢 **{fase_nome}: {fase_critica:.0f}%**\n\nAndamento adequado")
        else:
            st.info("ℹ️ **Análise de Fase**\n\nNenhuma fase encontrada")

# ============================================
# 🦙 ANÁLISE INTELIGENTE COM LLAMA 3.1 - PRINCIPAL MUDANÇA
# ============================================

def analise_llama_offline(df, metricas):
    """Análise inteligente offline - baseada em regras (mantido igual)"""
    
    try:
        dados = {
            "total": len(df),
            "conclusao": int((df['peso'].sum() / len(df)) * 100) if len(df) > 0 else 0,
            "nao_definidas": int(metricas['status_counts'].get('Não Definido', 0)),
            "validacoes": int(metricas['status_counts'].get('Aguardando Validação', 0)),
        }
    except Exception as e:
        return {"erro": f"Erro nos dados: {e}", "ok": False}
    
    # Análise inteligente baseada em regras
    riscos = []
    acoes = []
    
    # Regra 1: Atividades não definidas
    pct_nao_def = (dados['nao_definidas'] / dados['total']) * 100 if dados['total'] > 0 else 0
    if pct_nao_def > 50:
        riscos.append(f"🔴 {pct_nao_def:.0f}% atividades não definidas criam incerteza total no cronograma")
        acoes.append("→ Reunião urgente para definir responsáveis e prazos das atividades pendentes")
    elif pct_nao_def > 30:
        riscos.append(f"🟡 {pct_nao_def:.0f}% atividades não definidas podem atrasar entregas")
        acoes.append("→ Priorizar definição de responsáveis para atividades críticas")
    
    # Regra 2: Validações em atraso
    if dados['validacoes'] > 5:
        riscos.append("🔴 Deutsche com muitas validações pendentes gera gargalo crítico")
        acoes.append("→ Escalação urgente com Deutsche para acelerar processo de validação")
    elif dados['validacoes'] > 2:
        riscos.append(f"🟡 {dados['validacoes']} validações pendentes podem se acumular")
        acoes.append("→ Acompanhar prazos de aprovação com responsáveis")
    
    # Regra 3: Conclusão geral baixa
    if dados['conclusao'] < 40:
        riscos.append(f"🟡 Conclusão geral de {dados['conclusao']}% abaixo da meta de 50%")
        acoes.append("→ Revisar cronograma e identificar atividades que podem ser aceleradas")
    
    # Montar análise final
    analise_final = "**🎯 ANÁLISE OFFLINE (Regras):**\n\n"
    
    if riscos:
        analise_final += "**RISCOS IDENTIFICADOS:**\n"
        for i, risco in enumerate(riscos[:2], 1):
            analise_final += f"{i}. {risco}\n"
    
    if acoes:
        analise_final += "\n**AÇÕES RECOMENDADAS:**\n"
        for i, acao in enumerate(acoes[:2], 1):
            analise_final += f"{i}. {acao}\n"
    
    if not riscos and not acoes:
        analise_final += "✅ **Situação sob controle** - Projeto dentro dos parâmetros esperados."
    
    return {
        "analise": analise_final,
        "custo": 0.000,  # Offline = gratuito
        "modo": "offline",
        "ok": True
    }

def analise_llama_online(df, metricas, api_key):
    """Análise inteligente online - usando Llama 3.1 via Groq (NOVA FUNÇÃO)"""
    
    try:
        # Preparar dados para o prompt
        total = len(df)
        conclusao = int((df['peso'].sum() / total) * 100) if total > 0 else 0
        nao_definidas = int(metricas['status_counts'].get('Não Definido', 0))
        validacoes = int(metricas['status_counts'].get('Aguardando Validação', 0))
        andamento = int(metricas['status_counts'].get('Em Andamento', 0))
        concluidas = int(metricas['status_counts'].get('Concluído', 0))
        
        # Análise por fase
        fases_info = ""
        if len(metricas['fase_stats']) > 0:
            for fase, stats in metricas['fase_stats'].iterrows():
                conclusao_fase = stats['conclusao_fase']
                fases_info += f"• {fase}: {conclusao_fase:.0f}% conclusão\n"
        
        # Prompt otimizado para Llama 3.1
        prompt = f"""Analise este projeto PMO Deutsche Telco:

📊 **DADOS ATUAIS:**
• Total: {total} atividades
• Conclusão geral: {conclusao}%
• Concluídas: {concluidas}
• Em andamento: {andamento}
• Aguardando validação: {validacoes}
• Não definidas: {nao_definidas}

📈 **POR FASE:**
{fases_info}

🎯 **MISSÃO:** Como especialista PMO, identifique os 2 maiores RISCOS e as 2 AÇÕES mais urgentes para este projeto de transformação digital. Seja direto e estratégico.

Formato de resposta:
**🚨 RISCOS:**
1. [risco mais crítico]
2. [segundo risco]

**💡 AÇÕES:**
1. [ação mais urgente]
2. [segunda ação]"""
        
        # Chamada para a API do Groq (FORMATO DIFERENTE DO CLAUDE)
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"  # Bearer ao invés de x-api-key
        }
        
        data = {
            "model": "llama-3.3-70b-versatile",  # Modelo Llama 3.1 gratuito
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 400,
            "temperature": 0.1,
            "top_p": 1
        }
        
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",  # URL do Groq
            headers=headers,
            json=data,
            timeout=30
        )
        
        if response.status_code == 200:
            resultado = response.json()
            # Formato de resposta OpenAI (diferente do Claude)
            analise = resultado['choices'][0]['message']['content']
            
            analise_completa = "**🦙 ANÁLISE LLAMA 3.1 ONLINE (GRATUITO):**\n\n" + analise
            
            return {
                "analise": analise_completa,
                "custo": 0.000,  # Groq é gratuito!
                "modo": "online",
                "ok": True
            }
        else:
            status_code = response.status_code
            error_msg = f"API erro {status_code}"
            
            try:
                error_detail = response.json()
                if 'error' in error_detail:
                    if 'message' in error_detail['error']:
                        message = error_detail['error']['message']
                        error_msg = f"{error_msg}: {message}"
                    else:
                        error_msg = f"{error_msg}: {error_detail['error']}"
            except:
                error_msg = f"{error_msg}: Erro desconhecido"
            
            return {"erro": error_msg, "ok": False}
            
    except requests.Timeout:
        return {"erro": "Timeout na API - tente novamente", "ok": False}
    except Exception as e:
        return {"erro": f"Erro: {str(e)}", "ok": False}

def widget_llama_analise(df, metricas, api_key):
    """Widget para análise com Llama 3.1 (online/offline) - ADAPTADO"""
    
    st.markdown("---")
    st.subheader("🦙 Análise Inteligente Llama 3.3")
    
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        # Botões para análise
        col_offline, col_online = st.columns(2)
        
        with col_offline:
            if st.button("🔍 Análise Offline", type="secondary"):
                with st.spinner("🧠 Analisando (offline)..."):
                    resultado = analise_llama_offline(df, metricas)
                st.session_state['ultima_analise'] = resultado
        
        with col_online:
            if api_key:
                if st.button("🦙 Análise Online", type="primary"):
                    with st.spinner("🦙 Llama 3.1 analisando online..."):
                        resultado = analise_llama_online(df, metricas, api_key)
                    st.session_state['ultima_analise'] = resultado
            else:
                st.button("🦙 Análise Online", disabled=True, help="API Key do Groq necessária")
    
    with col2:
        modo = st.session_state.get('ultima_analise', {}).get('modo', 'N/A')
        st.metric("Modo", modo.title() if modo and modo != 'N/A' else 'N/A')
    
    with col3:
        custo = st.session_state.get('ultima_analise', {}).get('custo', 0)
        st.metric("Custo", "GRATUITO 🎉" if custo == 0 else f"${custo:.5f}")
    
    # Exibir resultado da análise
    if 'ultima_analise' in st.session_state:
        resultado = st.session_state['ultima_analise']
        
        if resultado.get("ok"):
            st.success("✅ **Análise Concluída:**")
            st.markdown(resultado["analise"])
            
            if resultado.get("modo") == "online":
                st.caption("🦙 Powered by Llama 3.1-70b via Groq | 100% GRATUITO!")
            else:
                st.caption("🔵 Análise offline baseada em regras inteligentes")
                
        else:
            st.error(f"❌ {resultado.get('erro', 'Erro desconhecido')}")
            st.info("💡 Tente usar a análise offline como alternativa")

# ============================================
# 📱 INTERFACE PRINCIPAL (PEQUENOS AJUSTES)
# ============================================

def main():
    """Função principal do dashboard"""
    
    # Carregar dados
    df = carregar_dados()
    
    if df.empty:
        st.error("❌ Não foi possível carregar os dados do projeto!")
        st.stop()
    
    metricas = calcular_metricas(df)
    
    # Título principal
    st.title("📊 PMO - Programa Digital Telco 🦙")
    st.markdown("**Programa Digital Telco - Análise Inteligente com Llama 3.3**")
    st.markdown(f"**Atualizado:** {datetime.now().strftime('%d/%m/%Y %H:%M')}")
    
    # Sidebar com filtros e configuração
    df_filtrado, api_key = criar_sidebar(df, metricas)
    
    # Recalcular métricas com filtros
    if len(df_filtrado) != len(df):
        metricas_filtradas = calcular_metricas(df_filtrado)
    else:
        metricas_filtradas = metricas
    
    # ============================================
    # 📊 MÉTRICAS PRINCIPAIS
    # ============================================
    
    st.markdown("---")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Total de Atividades", 
            metricas_filtradas['total_atividades']
        )
    
    with col2:
        conclusao = metricas_filtradas['conclusao_geral']
        delta_conclusao = conclusao - 50
        st.metric(
            "Conclusão Geral", 
            f"{conclusao}%",
            delta=f"{delta_conclusao:+.0f}% vs Meta 50%"
        )
    
    with col3:
        concluidas = metricas_filtradas['status_counts'].get('Concluído', 0)
        st.metric(
            "Atividades Concluídas", 
            concluidas,
            delta=f"{round(concluidas/len(df_filtrado)*100) if len(df_filtrado) > 0 else 0}% do total"
        )
    
    with col4:
        validacoes = metricas_filtradas['status_counts'].get('Aguardando Validação', 0)
        st.metric(
            "Aguardando Validação", 
            validacoes,
            delta="Deutsche" if validacoes > 0 else None
        )
    
    # ============================================
    # 📊 GRÁFICOS PRINCIPAIS
    # ============================================
    
    st.markdown("---")
    
    # Linha 1: Rosca e Barras
    col1, col2 = st.columns(2)
    
    with col1:
        fig_rosca = criar_rosca_status(df_filtrado)
        st.plotly_chart(fig_rosca, use_container_width=True)
    
    with col2:
        fig_barras = criar_barras_fase(df_filtrado)
        st.plotly_chart(fig_barras, use_container_width=True)
    
    # Linha 2: Velocímetro e Distribuição
    col1, col2 = st.columns(2)
    
    with col1:
        fig_velocimetro = criar_velocimetro(df_filtrado)
        st.plotly_chart(fig_velocimetro, use_container_width=True)
    
    with col2:
        fig_responsavel = criar_distribuicao_responsavel(df_filtrado)
        st.plotly_chart(fig_responsavel, use_container_width=True)
    
    # ============================================
    # 🚨 ALERTAS
    # ============================================
    
    st.markdown("---")
    criar_alertas(df_filtrado, metricas_filtradas)
    
    # ============================================
    # 🦙 ANÁLISE INTELIGENTE
    # ============================================
    
    widget_llama_analise(df_filtrado, metricas_filtradas, api_key)
    
    # ============================================
    # 📋 TABELA DE ATIVIDADES
    # ============================================
    
    st.markdown("---")
    st.subheader("📋 Lista Detalhada de Atividades")
    
    # Filtros adicionais para a tabela
    col1, col2 = st.columns(2)
    
    with col1:
        ordenar_por = st.selectbox(
            "Ordenar por:",
            ['id', 'fase', 'status', 'responsavel', 'progresso']
        )
    
    with col2:
        ordem = st.selectbox("Ordem:", ['Crescente', 'Decrescente'])
    
    # Aplicar ordenação
    df_ordenado = df_filtrado.sort_values(
        ordenar_por, 
        ascending=(ordem == 'Crescente')
    )
    
    criar_tabela_atividades(df_ordenado)
    
    # ============================================
    # 📈 RESUMO EXECUTIVO (ATUALIZADO PARA LLAMA)
    # ============================================
    
    st.markdown("---")
    st.subheader("📈 Resumo Executivo")
    
    with st.expander("🎯 Principais Insights", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **✅ Pontos Positivos:**
            - Dashboard felito em streamlit + LLM Llama 3.3
            - Análise 100% GRATUITA
            - Métricas em tempo real
            - Análise automática de riscos
            """)
        
        with col2:
            st.markdown("""
            **🚀 Novidades Llama:**
            - Llama 3.3-70b via Groq API
            - Modo offline como backup
            - Zero custo por análise
            - Performance excelente
            """)
    
    with st.expander("🦙 Como Usar a Análise Llama 3.1"):
        st.markdown("""
        **🦙 Modo Online (Recomendado - GRATUITO!):**
        1. Registre-se em **console.groq.com** (gratuito)
        2. Copie sua API Key (gsk_...)
        3. Cole na sidebar e ative "Usar Llama Online"
        4. Clique em "Análise Online" para insights avançados
        5. **Custo: 100% GRATUITO! 🎉**
        
        **🔵 Modo Offline (Backup):**
        1. Não precisa de API Key
        2. Análise baseada em regras inteligentes
        3. Funciona sempre, sem internet
        4. Resultados consistentes e confiáveis
        
        **💡 Dica:** O Llama 3.3-70b é tão bom quanto Claude/OpenAI, mas totalmente gratuito!
        
        **🚀 Setup rápido:**
        - Acesse: https://console.groq.com
        - Faça login (grátis)
        - Vá em "API Keys" 
        - Crie uma nova chave
        - Cole no campo "API Key do Groq"
        """)

# ============================================
# 🚀 EXECUTAR APLICAÇÃO
# ============================================

if __name__ == "__main__":
    main()