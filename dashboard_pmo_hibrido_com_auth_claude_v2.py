import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta
import openpyxl
import requests
import json
import os

# ============================================
# 🔐 IMPORTAR AUTENTICAÇÃO
# ============================================
import auth

# ============================================
# 🔧 CONFIGURAÇÃO DA PÁGINA
# ============================================

st.set_page_config(
    page_title="PMO-Digital Transformation Program",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================
# 🦙 CONFIGURAÇÃO HÍBRIDA - ONLINE + OFFLINE AVANÇADO
# ============================================

def configurar_analise_hibrida():
    """Configuração para análise online (Claude) + offline avançada"""
    st.sidebar.markdown("---")
    st.sidebar.subheader("🤖 AI Insights")
    
    # Verificar API key salva
    api_key_default = os.getenv('ANTHROPIC_API_KEY', '')
    
    api_key = st.sidebar.text_input(
        "API Key do Claude (Anthropic):",
        value=api_key_default,
        type="password",
        help="Chave da Anthropic para análise premium com Claude"
    )
    
    # Seleção do modelo Claude
    modelo_opcoes = {
        "claude-3-haiku-20240307": "Claude 3 Haiku (Rápido + Barato)",
        "claude-3-sonnet-20240229": "Claude 3 Sonnet (Balanceado)",
        "claude-3-opus-20240229": "Claude 3 Opus (Máxima Qualidade)"
    }
    
    modelo_selecionado = st.sidebar.selectbox(
        "Modelo Claude:",
        options=list(modelo_opcoes.keys()),
        format_func=lambda x: modelo_opcoes[x],
        index=1,  # Sonnet como padrão
        help="Haiku: ~$0.25/1M tokens | Sonnet: ~$3/1M tokens | Opus: ~$15/1M tokens"
    )
    
    # Toggle para modo online
    usar_online = st.sidebar.toggle(
        "🧠 Análise Online (Claude)",
        value=bool(api_key),
        help="Ativar análise premium com Claude (Anthropic)"
    )
    
    # Status dos modos
    col1, col2 = st.sidebar.columns(2)
    
    with col1:
        if usar_online and api_key:
            st.success("🟢 Claude API")
        elif usar_online and not api_key:
            st.error("🔴 API Key!")
        else:
            st.info("🔵 Offline")
    
    with col2:
        st.success("🟢 Offline+")
    
    # Sempre ter offline avançado disponível
    st.sidebar.info("✨ Versão Offline sempre ativa com indicadores avançados")
    
    return (api_key, modelo_selecionado) if usar_online else (None, None)

# ============================================
# 🧠 ANALISADOR PMO AVANÇADO COM INDICADORES DE DATA
# ============================================

class AnalisadorPMOAvancado:
    """Analisador PMO com indicadores avançados de cronograma e data"""
    
    def __init__(self):
        self.hoje = datetime.now().date()
        
    def processar_datas(self, df):
        """Processa e converte datas para análise"""
        if df.empty or 'data_prevista' not in df.columns:
            return df
        
        df = df.copy()
        
        # Converter data_prevista para datetime
        try:
            df['data_prevista'] = pd.to_datetime(df['data_prevista']).dt.date
            df['dias_para_prazo'] = df['data_prevista'].apply(
                lambda x: (x - self.hoje).days if pd.notna(x) and x is not None else None
            )
        except Exception as e:
            st.warning(f"⚠️ Erro ao processar datas: {e}")
            df['dias_para_prazo'] = None
            
        return df
    
    def calcular_indicadores_cronograma(self, df):
        """Calcula novos indicadores baseados em cronograma"""
        if df.empty:
            return {}
        
        df = self.processar_datas(df)
        
        # Filtrar apenas atividades com data
        df_com_data = df.dropna(subset=['data_prevista'])
        
        if df_com_data.empty:
            return {
                'total_com_data': 0,
                'atrasadas': 0,
                'vencendo_semana': 0,
                'futuras': 0,
                'aderencia_prazo': 0,
                'dias_atraso_medio': 0,
                'sla_validacao': 0
            }
        
        total_com_data = len(df_com_data)
        
        # 1. Atividades por situação de prazo
        atrasadas = len(df_com_data[
            (df_com_data['dias_para_prazo'] < 0) & 
            (df_com_data['status'] != 'Concluído')
        ])
        
        vencendo_semana = len(df_com_data[
            (df_com_data['dias_para_prazo'] >= 0) & 
            (df_com_data['dias_para_prazo'] <= 7) &
            (df_com_data['status'] != 'Concluído')
        ])
        
        futuras = len(df_com_data[
            (df_com_data['dias_para_prazo'] > 7) &
            (df_com_data['status'] != 'Concluído')
        ])
        
        # 2. Aderência aos prazos (% de atividades no prazo)
        concluidas = len(df_com_data[df_com_data['status'] == 'Concluído'])
        no_prazo = len(df_com_data[
            (df_com_data['status'] != 'Concluído') & 
            (df_com_data['dias_para_prazo'] >= 0)
        ])
        
        aderencia_prazo = round(((concluidas + no_prazo) / total_com_data) * 100) if total_com_data > 0 else 0
        
        # 3. Dias médios de atraso
        atividades_atrasadas = df_com_data[
            (df_com_data['dias_para_prazo'] < 0) & 
            (df_com_data['status'] != 'Concluído')
        ]
        
        if len(atividades_atrasadas) > 0:
            atraso_medio = atividades_atrasadas['dias_para_prazo'].mean()
            dias_atraso_medio = round(abs(atraso_medio)) if not pd.isna(atraso_medio) else 0
        else:
            dias_atraso_medio = 0
        
        # 4. SLA de validação (tempo médio em "Aguardando Validação")
        validacoes = df_com_data[df_com_data['status'] == 'Aguardando Validação']
        if len(validacoes) > 0:
            sla_medio = validacoes['dias_para_prazo'].mean()
            sla_validacao = round(abs(sla_medio)) if not pd.isna(sla_medio) else 0
        else:
            sla_validacao = 0
        
        return {
            'total_com_data': total_com_data,
            'atrasadas': atrasadas,
            'vencendo_semana': vencendo_semana,
            'futuras': futuras,
            'aderencia_prazo': aderencia_prazo,
            'dias_atraso_medio': dias_atraso_medio,
            'sla_validacao': sla_validacao
        }
    
    def prever_conclusao_projeto(self, df):
        """Prevê data de conclusão do projeto"""
        df = self.processar_datas(df)
        
        if df.empty or 'data_prevista' not in df.columns:
            return None
        
        # Última data prevista do projeto
        try:
            df_com_data = df.dropna(subset=['data_prevista'])
            if df_com_data.empty:
                return None
                
            ultima_data = df_com_data['data_prevista'].max()
            
            # Calcular possível atraso baseado nas atividades atrasadas
            atrasadas = df_com_data[
                (df_com_data['dias_para_prazo'] < 0) & 
                (df_com_data['status'] != 'Concluído')
            ]
            
            if len(atrasadas) > 0:
                atraso_medio = abs(atrasadas['dias_para_prazo'].mean())
                # Garantir que seja um número inteiro
                atraso_medio = int(atraso_medio) if not pd.isna(atraso_medio) else 0
                data_prevista_ajustada = ultima_data + timedelta(days=atraso_medio)
            else:
                data_prevista_ajustada = ultima_data
            
            return {
                'data_planejada': ultima_data,
                'data_prevista': data_prevista_ajustada,
                'dias_adicao': (data_prevista_ajustada - ultima_data).days
            }
        except Exception as e:
            st.warning(f"Erro ao calcular previsão: {e}")
            return None
    
    def analisar_tendencia_cronograma(self, df):
        """Analisa tendência do cronograma"""
        df = self.processar_datas(df)
        
        if df.empty:
            return {'tendencia': 'neutro', 'razao': 'Sem dados'}
        
        cronograma = self.calcular_indicadores_cronograma(df)
        
        # Lógica de tendência
        if cronograma['atrasadas'] > 5:
            return {'tendencia': 'critica', 'razao': f"{cronograma['atrasadas']} atividades atrasadas"}
        elif cronograma['aderencia_prazo'] < 60:
            return {'tendencia': 'pessima', 'razao': f"Aderência baixa ({cronograma['aderencia_prazo']}%)"}
        elif cronograma['vencendo_semana'] > 3:
            return {'tendencia': 'atencao', 'razao': f"{cronograma['vencendo_semana']} atividades vencendo"}
        elif cronograma['aderencia_prazo'] >= 80:
            return {'tendencia': 'otima', 'razao': f"Aderência alta ({cronograma['aderencia_prazo']}%)"}
        else:
            return {'tendencia': 'boa', 'razao': 'Cronograma controlado'}
    
    def gerar_recomendacoes_cronograma(self, df):
        """Gera recomendações específicas de cronograma"""
        cronograma = self.calcular_indicadores_cronograma(df)
        recomendacoes = []
        
        # Recomendações baseadas em atrasos
        if cronograma['atrasadas'] > 5:
            recomendacoes.append({
                'categoria': 'Cronograma Crítico',
                'acao': f"Reunião de emergência - {cronograma['atrasadas']} atividades atrasadas",
                'prioridade': 'crítica',
                'impacto': 'Evitar atraso total do projeto',
                'prazo': '24h'
            })
        elif cronograma['atrasadas'] > 2:
            recomendacoes.append({
                'categoria': 'Gestão de Prazo',
                'acao': f"Plano de recuperação para {cronograma['atrasadas']} atividades atrasadas",
                'prioridade': 'alta',
                'impacto': 'Reduzir impacto nos prazos finais',
                'prazo': '3 dias'
            })
        
        # Recomendações para atividades vencendo
        if cronograma['vencendo_semana'] > 3:
            recomendacoes.append({
                'categoria': 'Monitoramento',
                'acao': f"Acompanhamento diário de {cronograma['vencendo_semana']} atividades críticas",
                'prioridade': 'alta',
                'impacto': 'Evitar novos atrasos',
                'prazo': '7 dias'
            })
        
        # Recomendações para SLA de validação
        if cronograma['sla_validacao'] > 30:
            recomendacoes.append({
                'categoria': 'Processo',
                'acao': f"Melhorar SLA de validação ({cronograma['sla_validacao']} dias médios)",
                'prioridade': 'media',
                'impacto': 'Acelerar fluxo de aprovações',
                'prazo': '15 dias'
            })
        
        return recomendacoes[:3]  # Top 3
    
    def analise_completa_com_cronograma(self, df, metricas):
        """Análise completa incluindo novos indicadores de cronograma"""
        try:
            # Análise básica
            score_saude = self.calcular_score_saude_avancado(df, metricas)
            
            # Novos indicadores de cronograma
            cronograma = self.calcular_indicadores_cronograma(df)
            
            # Previsão de conclusão
            previsao_conclusao = self.prever_conclusao_projeto(df)
            
            # Tendência
            tendencia = self.analisar_tendencia_cronograma(df)
            
            # Recomendações específicas de cronograma
            recomendacoes_cronograma = self.gerar_recomendacoes_cronograma(df)
            
            # Recomendações gerais
            recomendacoes_gerais = self.gerar_recomendacoes_gerais(df, metricas)
            
            # Combinar recomendações
            todas_recomendacoes = recomendacoes_cronograma + recomendacoes_gerais
            
            return {
                "ok": True,
                "score_saude": score_saude,
                "cronograma": cronograma,
                "previsao_conclusao": previsao_conclusao,
                "tendencia": tendencia,
                "recomendacoes": todas_recomendacoes[:4],  # Top 4
                "modo": "offline-avancado"
            }
            
        except Exception as e:
            return {
                "ok": False,
                "erro": f"Erro na análise: {str(e)}"
            }
    
    def calcular_score_saude_avancado(self, df, metricas):
        """Score de saúde incluindo fatores de cronograma"""
        if df.empty:
            return 0
        
        # Score básico (40% do peso)
        conclusao = metricas['conclusao_geral']
        score_basico = conclusao * 0.4
        
        # Score de cronograma (40% do peso)
        cronograma = self.calcular_indicadores_cronograma(df)
        score_cronograma = cronograma['aderencia_prazo'] * 0.4
        
        # Score de distribuição (20% do peso)
        resp_dist = df['responsavel'].value_counts()
        if len(resp_dist) > 0:
            distribuicao_pct = min((resp_dist.max() / len(df)) * 100, 80)
            score_distribuicao = (100 - distribuicao_pct + 20) * 0.2
        else:
            score_distribuicao = 0
        
        score_final = score_basico + score_cronograma + score_distribuicao
        return round(min(score_final, 100))
    
    def gerar_recomendacoes_gerais(self, df, metricas):
        """Recomendações gerais do projeto"""
        recomendacoes = []
        
        # Análise de responsáveis
        resp_count = df['responsavel'].value_counts()
        if len(resp_count) > 0:
            max_responsavel = resp_count.index[0]
            max_count = resp_count.iloc[0]
            if max_count > len(df) * 0.6:
                recomendacoes.append({
                    'categoria': 'Gestão de Recursos',
                    'acao': f"Redistribuir atividades - {max_responsavel} sobrecarregado ({max_count} atividades)",
                    'prioridade': 'alta',
                    'impacto': 'Balancear carga de trabalho',
                    'prazo': '7 dias'
                })
        
        # Análise de validações
        validacoes = metricas['status_counts'].get('Aguardando Validação', 0)
        if validacoes > 5:
            recomendacoes.append({
                'categoria': 'Governança',
                'acao': f"Acelerar {validacoes} validações pendentes",
                'prioridade': 'alta',
                'impacto': 'Destravar fluxo do projeto',
                'prazo': '5 dias'
            })
        
        return recomendacoes

# Instância global do analisador
analisador = AnalisadorPMOAvancado()

# ============================================
# 📊 CARREGAR E PROCESSAR DADOS
# ============================================

@st.cache_data
def carregar_dados():
    """Carrega dados do arquivo Excel com novos campos"""
    try:
        df = pd.read_excel('atividades_pmo.xlsx', sheet_name='Atividades')
        
        # Validar colunas obrigatórias (incluindo nova data_prevista)
        colunas_obrigatorias = ['id', 'fase', 'atividade', 'status', 'responsavel', 'data_prevista', 'progresso', 'peso', 'delivery', 'Dimensões']
        
        if not all(col in df.columns for col in colunas_obrigatorias):
            st.error("❌ Arquivo Excel não tem as colunas corretas!")
            st.info("Colunas esperadas: " + ", ".join(colunas_obrigatorias))
            return pd.DataFrame()
        
        # Remover linhas vazias
        df = df.dropna(subset=['id'])
        
        return df
        
    except FileNotFoundError:
        st.warning("⚠️ Arquivo 'atividades_pmo.xlsx' não encontrado! Usando dados de exemplo.")
        return carregar_dados_exemplo()
        
    except Exception as e:
        st.error(f"❌ Erro ao ler Excel: {e}")
        return carregar_dados_exemplo()

def carregar_dados_exemplo():
    """Dados de exemplo com novos campos"""
    dados_exemplo = [
        {"id": "1.1", "fase": "Marco Inicial", "atividade": "Kick-Off", "status": "Concluído", "responsavel": "OS", "data_prevista": "2025-02-12", "progresso": 100, "peso": 1.0, "observacoes": "Realizado", "delivery": "Sprint 1", "Dimensões": "Processos"},
        {"id": "2.1", "fase": "Fase 1", "atividade": "Diagnóstico", "status": "Aguardando Validação", "responsavel": "DT", "data_prevista": "2025-04-07", "progresso": 80, "peso": 0.8, "observacoes": "Aguarda validação", "delivery": "Sprint 2", "Dimensões": "Tecnologia"},
        {"id": "2.2", "fase": "Fase 1", "atividade": "Análise Gaps", "status": "Em Andamento", "responsavel": "OS", "data_prevista": "2025-05-30", "progresso": 50, "peso": 0.5, "observacoes": "Em progresso", "delivery": "Sprint 3", "Dimensões": "Pessoas"},
        {"id": "3.1", "fase": "Fase 2", "atividade": "Estruturação", "status": "Identificado", "responsavel": "A Definir", "data_prevista": "2025-06-30", "progresso": 0, "peso": 0.0, "observacoes": "Pendente", "delivery": "Sprint 4", "Dimensões": "Governança"},
    ]
    return pd.DataFrame(dados_exemplo)

def calcular_metricas(df):
    """Calcula métricas incluindo novos indicadores"""
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
# 🎨 SIDEBAR HÍBRIDA
# ============================================

def criar_sidebar(df, metricas):
    """Sidebar com configuração híbrida e novos indicadores"""
    
    st.sidebar.title("🎯 PMO Digital Transformation Program")
    st.sidebar.markdown("**Digital Transformation Program**")
    st.sidebar.markdown("*Modelo Premium: Claude + Offline*")
    
    # Configuração da análise híbrida
    api_config = configurar_analise_hibrida()
    api_key = api_config[0] if api_config else None
    modelo = api_config[1] if api_config else None
    
    st.sidebar.markdown("---")
    
    # Métricas principais
    st.sidebar.metric("Total de Atividades", metricas['total_atividades'])
    st.sidebar.metric(
        "Conclusão Geral", 
        f"{metricas['conclusao_geral']}%",
        delta=f"{metricas['conclusao_geral'] - 50}% vs Meta 50%"
    )
    
    # Novos indicadores de cronograma na sidebar
    if not df.empty:
        try:
            cronograma = analisador.calcular_indicadores_cronograma(df)
            
            st.sidebar.metric(
                "Aderência aos Prazos",
                f"{cronograma['aderencia_prazo']}%",
                delta=f"{cronograma['aderencia_prazo'] - 80}% vs Meta 80%"
            )
            
            if cronograma['atrasadas'] > 0:
                st.sidebar.metric(
                    "Atividades Atrasadas",
                    cronograma['atrasadas'],
                    delta=f"-{cronograma['dias_atraso_medio']} dias médio"
                )
        except Exception as e:
            st.sidebar.warning("⚠️ Erro nos indicadores de cronograma")
    
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
    
    return df_filtrado, api_key, modelo

# ============================================
# 📊 GRÁFICOS COM INDICADORES DE CRONOGRAMA
# ============================================

def criar_gauge_aderencia(df):
    """Cria gauge de aderência aos prazos com explicação"""
    
    if df.empty:
        aderencia = 0
    else:
        cronograma = analisador.calcular_indicadores_cronograma(df)
        aderencia = cronograma['aderencia_prazo']
    
    # Determinar cor da barra
    if aderencia >= 90:
        cor_barra = "#28a745"  # Verde
        status = "Excelente"
        cor_status = "#28a745"
    elif aderencia >= 70:
        cor_barra = "#ffc107"  # Amarelo
        status = "Bom"
        cor_status = "#ffc107"
    elif aderencia >= 50:
        cor_barra = "#fd7e14"  # Laranja
        status = "Atenção"
        cor_status = "#fd7e14"
    else:
        cor_barra = "#dc3545"  # Vermelho
        status = "Crítico"
        cor_status = "#dc3545"
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=aderencia,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': f"⏰ Aderência aos Prazos<br><span style='font-size:0.6em; color:gray'>% de atividades dentro do prazo</span>", 'font': {'size': 18}},
        number={'font': {'size': 48, 'color': '#1e3a8a'}, 'suffix': '%'},  # Azul escuro
        gauge={
            'axis': {'range': [None, 100], 'tickwidth': 1},
            'bar': {'color': cor_barra},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 50], 'color': '#ffe6e6'},    # Vermelho claro
                {'range': [50, 70], 'color': '#fff9e6'},   # Amarelo muito claro
                {'range': [70, 90], 'color': '#fff3cd'},   # Amarelo claro
                {'range': [90, 100], 'color': '#e6f7e6'}   # Verde claro
            ],
            'threshold': {
                'line': {'color': "#1e3a8a", 'width': 4},  # Azul escuro
                'thickness': 0.75,
                'value': 85
            }
        }
    ))
    
    # Adicionar texto explicativo no centro
    fig.add_annotation(
        text=f"<b>{status}</b>",
        x=0.5, y=0.2,
        font_size=16,
        font_color=cor_status,
        showarrow=False
    )
    
    fig.update_layout(height=400, margin=dict(t=80, b=40, l=40, r=40))
    return fig

def criar_grafico_situacao_prazos(df):
    """Cria gráfico de situação dos prazos"""
    
    if df.empty:
        return go.Figure().add_annotation(text="Nenhum dado disponível", x=0.5, y=0.5)
    
    cronograma = analisador.calcular_indicadores_cronograma(df)
    
    # Dados para o gráfico
    situacoes = ['Atrasadas', 'Vencendo (7 dias)', 'Futuras']
    valores = [cronograma['atrasadas'], cronograma['vencendo_semana'], cronograma['futuras']]
    cores = ['#F44336', '#FF9800', '#4CAF50']
    
    fig = go.Figure([go.Bar(
        x=situacoes,
        y=valores,
        marker_color=cores,
        text=valores,
        textposition='auto'
    )])
    
    fig.update_layout(
        title='🚨 Situação dos Prazos',
        xaxis_title='Situação',
        yaxis_title='Quantidade de Atividades',
        height=400,
        margin=dict(t=60, b=40, l=40, r=40)
    )
    
    return fig

def criar_roadmap_planejado_vs_realizado(df):
    """Cria roadmap comparando planejado vs realizado por fase"""
    
    if df.empty:
        return go.Figure().add_annotation(text="Nenhum dado disponível", x=0.5, y=0.5)
    
    try:
        # Processar dados com verificações robustas
        df_processed = analisador.processar_datas(df)
        
        if 'data_prevista' not in df_processed.columns:
            return go.Figure().add_annotation(text="Campo data_prevista não encontrado", x=0.5, y=0.5)
        
        # Remover linhas com datas nulas
        df_valid = df_processed.dropna(subset=['data_prevista']).copy()
        
        if df_valid.empty:
            return go.Figure().add_annotation(text="Nenhuma data válida encontrada", x=0.5, y=0.5)
        
        # Converter explicitamente para datetime
        df_valid['data_dt'] = pd.to_datetime(df_valid['data_prevista'], errors='coerce')
        df_valid = df_valid.dropna(subset=['data_dt'])
        
        if df_valid.empty:
            return go.Figure().add_annotation(text="Erro na conversão de datas", x=0.5, y=0.5)
        
        # Agrupar por fase
        fases_resumo = []
        
        for fase in df_valid['fase'].unique():
            df_fase = df_valid[df_valid['fase'] == fase]
            
            # Calcular datas min/max da fase
            data_inicio = df_fase['data_dt'].min()
            data_fim = df_fase['data_dt'].max()
            
            # Calcular progresso da fase
            total_atividades = len(df_fase)
            peso_total = df_fase['peso'].sum()
            progresso_fase = (peso_total / total_atividades) * 100 if total_atividades > 0 else 0
            
            fases_resumo.append({
                'fase': fase,
                'data_inicio': data_inicio,
                'data_fim': data_fim,
                'progresso': min(progresso_fase, 100),  # Limitar a 100%
                'total_atividades': total_atividades
            })
        
        if not fases_resumo:
            return go.Figure().add_annotation(text="Nenhuma fase processada", x=0.5, y=0.5)
        
        # Criar o gráfico
        fig = go.Figure()
        
        for i, fase_info in enumerate(fases_resumo):
            fase = fase_info['fase']
            data_inicio = fase_info['data_inicio']
            data_fim = fase_info['data_fim']
            progresso = fase_info['progresso']
            
            # Linha cinza - cronograma planejado (100%)
            fig.add_trace(go.Scatter(
                x=[data_inicio, data_fim],
                y=[fase, fase],
                mode='lines',
                line=dict(width=20, color='rgba(180, 180, 180, 0.7)'),
                name='Planejado' if i == 0 else '',
                showlegend=True if i == 0 else False,
                hovertemplate=f'<b>Planejado</b><br>Fase: {fase}<br>Período completo<extra></extra>'
            ))
            
            # Calcular ponto final do progresso realizado
            duracao_total = (data_fim - data_inicio).total_seconds()
            duracao_progresso = duracao_total * (progresso / 100)
            data_progresso = data_inicio + pd.Timedelta(seconds=duracao_progresso)
            
            # Determinar cor baseada no progresso
            if progresso >= 80:
                cor_progresso = '#28a745'  # Verde
                status = 'No prazo'
            elif progresso >= 50:
                cor_progresso = '#ffc107'  # Amarelo
                status = 'Atenção'
            else:
                cor_progresso = '#dc3545'  # Vermelho
                status = 'Atrasado'
            
            # Linha colorida - progresso realizado
            fig.add_trace(go.Scatter(
                x=[data_inicio, data_progresso],
                y=[fase, fase],
                mode='lines',
                line=dict(width=20, color=cor_progresso),
                name='Realizado' if i == 0 else '',
                showlegend=True if i == 0 else False,
                hovertemplate=f'<b>Realizado</b><br>Fase: {fase}<br>Progresso: {progresso:.1f}%<br>Status: {status}<extra></extra>'
            ))
            
            # Marcador com percentual
            fig.add_trace(go.Scatter(
                x=[data_progresso],
                y=[fase],
                mode='markers+text',
                marker=dict(size=15, color=cor_progresso, symbol='circle'),
                text=[f'{progresso:.0f}%'],
                textposition="middle right",
                textfont=dict(size=12, color='black', family='Arial Black'),
                showlegend=False,
                hovertemplate=f'<b>{fase}</b><br>Progresso atual: {progresso:.1f}%<extra></extra>'
            ))
        
        # Adicionar linha vertical para "HOJE" se possível
        try:
            hoje = pd.Timestamp.now()
            fig.add_vline(
                x=hoje,
                line_dash="dash",
                line_color="red",
                line_width=3,
                annotation_text="HOJE",
                annotation_position="top",
                annotation_font_size=12
            )
        except:
            pass  # Se der erro, não adiciona a linha
        
        # Layout do gráfico
        fig.update_layout(
            title='🗺️ Roadmap: Planejado vs Realizado',
            height=400,
            margin=dict(t=80, b=40, l=200, r=100),
            xaxis_title="Cronograma",
            yaxis_title="Fases",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="center",
                x=0.5
            ),
            hovermode='closest'
        )
        
        return fig
        
    except Exception as e:
        # Em caso de erro, mostrar mensagem mais informativa mas sem st.error para não duplicar
        error_msg = f"Erro ao criar roadmap: {str(e)}"
        return go.Figure().add_annotation(
            text="Erro no processamento do roadmap<br>Verifique os dados de entrada", 
            x=0.5, y=0.5,
            font_size=14
        )

# ============================================
# 📊 GRÁFICOS ORIGINAIS (MANTIDOS)
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
        'Identificado': '#9E9E9E'
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

# ============================================
# 🦙 ANÁLISE ONLINE COM CLAUDE
# ============================================

def analise_claude_online(df, metricas, api_key, modelo):
    """Análise online premium com Claude (Anthropic) incluindo custos detalhados"""
    
    # Definição do User Role (MOVIDO PARA CIMA)
    user_role = """
Você é um Especialista Senior em Escritório de Projetos (PMO/EPMO) com mais de 15 anos de experiência em:

EXPERTISE:
- Implementação e maturação de PMOs e EPMOs
- Frameworks ágeis (Scrum, SAFe, Kanban, Lean)
- Governança de portfólio de projetos
- Métricas e KPIs de performance de projetos
- Gestão de mudanças organizacionais

METODOLOGIAS DE REFERÊNCIA:
- PMI (Project Management Institute) - PMP, PgMP, PfMP
- Gartner Magic Quadrant para PPM Tools
- Gartner Best Practices para PMO Excellence
- OGC PRINCE2 e MSP
- Scaled Agile Framework (SAFe)

FOCO ANALÍTICO:
- Análise baseada em dados e métricas objetivas
- Identificação de riscos e oportunidades de melhoria
- Recomendações práticas e acionáveis
- Benchmarking com melhores práticas do mercado

Sempre forneça insights estratégicos, seja direto e use linguagem executiva apropriada para stakeholders C-level.
"""
    
    try:
        # Preparar dados básicos com validação
        total = len(df) if not df.empty else 0
        if total == 0:
            return {"erro": "Nenhum dado disponível para análise", "ok": False}
        
        # Calcular conclusão de forma segura
        try:
            conclusao = int((df['peso'].sum() / total) * 100) if total > 0 else 0
        except:
            conclusao = 0
        
        # Novos dados de cronograma com tratamento de erro
        try:
            cronograma = analisador.calcular_indicadores_cronograma(df)
        except:
            cronograma = {
                'aderencia_prazo': 0, 'atrasadas': 0, 'vencendo_semana': 0,
                'dias_atraso_medio': 0, 'sla_validacao': 0
            }
        
        # Previsão com tratamento de erro
        try:
            previsao = analisador.prever_conclusao_projeto(df)
            previsao_texto = f"Data prevista: {previsao['data_prevista'].strftime('%d/%m/%Y')}" if previsao else "Sem previsão"
            if previsao and previsao['dias_adicao'] > 0:
                previsao_texto += f" (Atraso previsto: {previsao['dias_adicao']} dias)"
        except:
            previsao_texto = "Previsão indisponível"
        
        # Status com limpeza
        try:
            status_resumo = dict(metricas['status_counts'].head(5))  # Top 5 status
        except:
            status_resumo = {"Em processamento": total}
        
        # Análise de responsáveis
        try:
            resp_stats = df['responsavel'].value_counts().head(5)
            resp_resumo = dict(resp_stats)
        except:
            resp_resumo = {"Equipe": total}
        
        # Construir prompt especializado para Claude
        prompt = f"""Analise este projeto PMO Digital Transformation Program como especialista sênior:

SITUAÇÃO ATUAL DO PROJETO:
• Total de atividades: {total}
• Progresso geral: {conclusao}%
• Distribuição por status: {status_resumo}
• Distribuição por responsável: {resp_resumo}

INDICADORES DE CRONOGRAMA:
• Aderência aos prazos: {cronograma['aderencia_prazo']}%
• Atividades atrasadas: {cronograma['atrasadas']}
• Atividades vencendo (7 dias): {cronograma['vencendo_semana']}
• Média de atraso: {cronograma['dias_atraso_medio']} dias
• SLA de validação: {cronograma['sla_validacao']} dias

PREVISÕES:
• {previsao_texto}

Como especialista PMO sênior, forneça uma análise executiva com:

1. DIAGNÓSTICO CRÍTICO (2-3 pontos principais)
2. RISCOS IMINENTES (2 riscos priorizados)
3. AÇÕES ESTRATÉGICAS (3 ações específicas com prazo)
4. INDICADORES A MONITORAR (2 KPIs críticos)

Foque em insights acionáveis para tomada de decisão executiva."""
        
        # Headers para API da Anthropic
        headers = {
            "Content-Type": "application/json",
            "X-API-Key": api_key.strip(),
            "anthropic-version": "2023-06-01"
        }
        
        # Dados para API Claude
        data = {
            "model": modelo,
            "max_tokens": 800,
            "system": user_role,  # <- USER ROLE ADICIONADO AQUI
            "messages": [
                {
                    "role": "user", 
                    "content": prompt
                }
            ],
            "temperature": 0.1
        }
        
        # Estimar tokens de entrada (aproximação: 1 token ≈ 4 caracteres)
        tokens_entrada = len(prompt) // 4
        
        # Chamada para API Anthropic
        response = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers=headers,
            json=data,
            timeout=45
        )
        
        # Verificar resposta
        if response.status_code != 200:
            try:
                error_detail = response.json()
                return {"erro": f"Claude API erro {response.status_code}: {error_detail.get('error', {}).get('message', 'Erro desconhecido')}", "ok": False}
            except:
                return {"erro": f"Claude API erro {response.status_code}: Resposta inválida", "ok": False}
        
        resultado = response.json()
        analise = resultado['content'][0]['text']
        
        # Extrair informações de uso para cálculo de custo
        usage = resultado.get('usage', {})
        tokens_entrada_real = usage.get('input_tokens', tokens_entrada)
        tokens_saida = usage.get('output_tokens', 200)
        
        # Calcular custo baseado no modelo
        custos_modelo = {
            "claude-3-haiku-20240307": {"input": 0.25, "output": 1.25},      # $/1M tokens
            "claude-3-sonnet-20240229": {"input": 3.0, "output": 15.0},      # $/1M tokens  
            "claude-3-opus-20240229": {"input": 15.0, "output": 75.0}        # $/1M tokens
        }
        
        custo_input = (tokens_entrada_real / 1000000) * custos_modelo[modelo]["input"]
        custo_output = (tokens_saida / 1000000) * custos_modelo[modelo]["output"]
        custo_total = custo_input + custo_output
        
        # Formatação da análise
        analise_completa = f"**🧠 ANÁLISE CLAUDE {modelo.upper()} - PMO DIGITAL TRANSFORMATION PROGRAM:**\n\n{analise}"
        
        return {
            "analise": analise_completa,
            "custo_total": custo_total,
            "custo_input": custo_input,
            "custo_output": custo_output,
            "tokens_entrada": tokens_entrada_real,
            "tokens_saida": tokens_saida,
            "modelo": modelo,
            "modo": "claude-online",
            "ok": True
        }
            
    except requests.exceptions.Timeout:
        return {"erro": "Timeout na API Claude - tente novamente", "ok": False}
    except requests.exceptions.ConnectionError:
        return {"erro": "Erro de conexão com API Claude", "ok": False}
    except Exception as e:
        return {"erro": f"Erro interno: {str(e)}", "ok": False}

# ============================================
# 🧠 WIDGET DE ANÁLISE HÍBRIDA
# ============================================

def widget_analise_hibrida(df, metricas, api_key, modelo):
    """Widget para análise híbrida (online Claude + offline avançado)"""
    
    st.markdown("---")
    st.subheader("🤖 AI Insights Premium (Claude) + Versão Offline")
    
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        # Botões para análise
        col_online, col_offline = st.columns(2)
        
        with col_online:
            if api_key:
                modelo_nome = modelo.split('-')[1].title() if modelo else "Claude"
                if st.button(f"🧠 Claude {modelo_nome}", type="primary"):
                    with st.spinner(f"🧠 Claude {modelo_nome} analisando projeto..."):
                        resultado = analise_claude_online(df, metricas, api_key, modelo)
                    st.session_state['analise_resultado'] = resultado
            else:
                st.button("🧠 Claude Premium", disabled=True, help="API Key da Anthropic necessária")
        
        with col_offline:
            if st.button("🔧 Análise Offline", type="secondary"):
                with st.spinner("🔧 Processando análise avançada..."):
                    resultado = analisador.analise_completa_com_cronograma(df, metricas)
                st.session_state['analise_resultado'] = resultado
    
    with col2:
        modo = st.session_state.get('analise_resultado', {}).get('modo', 'N/A')
        if modo == 'claude-online':
            modelo_usado = st.session_state.get('analise_resultado', {}).get('modelo', 'N/A')
            st.metric("Modelo", modelo_usado.split('-')[1].title() if modelo_usado != 'N/A' else 'N/A')
        else:
            st.metric("Modo", modo.replace('-', ' ').title() if modo != 'N/A' else 'N/A')
    
    with col3:
        if 'analise_resultado' in st.session_state and st.session_state['analise_resultado'].get('modo') == 'claude-online':
            custo = st.session_state['analise_resultado'].get('custo_total', 0)
            st.metric("Custo", f"${custo:.4f}" if custo > 0 else "Calculando...")
        else:
            st.metric("Custo", "ZERO 💰")
    
    # Exibir resultado da análise
    if 'analise_resultado' in st.session_state:
        resultado = st.session_state['analise_resultado']
        
        if resultado.get("ok"):
            
            # Para análise online (Claude)
            if resultado.get("modo") == "claude-online":
                st.success("✅ **Análise Claude Concluída:**")
                st.markdown(resultado["analise"])
                
                # Mostrar detalhes de custo
                with st.expander("💰 Detalhes de Custo da Análise", expanded=False):
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric(
                            "Tokens Entrada", 
                            f"{resultado.get('tokens_entrada', 0):,}"
                        )
                    
                    with col2:
                        st.metric(
                            "Tokens Saída", 
                            f"{resultado.get('tokens_saida', 0):,}"
                        )
                    
                    with col3:
                        st.metric(
                            "Custo Entrada", 
                            f"${resultado.get('custo_input', 0):.5f}"
                        )
                    
                    with col4:
                        st.metric(
                            "Custo Saída", 
                            f"${resultado.get('custo_output', 0):.5f}"
                        )
                    
                    # Resumo de custo
                    custo_total = resultado.get('custo_total', 0)
                    modelo_usado = resultado.get('modelo', 'N/A')
                    
                    st.info(f"""
                    **📊 Resumo da Análise:**
                    - **Modelo:** {modelo_usado}
                    - **Custo Total:** ${custo_total:.5f} (~R$ {custo_total * 5.50:.3f})
                    - **Eficiência:** {resultado.get('tokens_saida', 0) / max(resultado.get('tokens_entrada', 1), 1):.2f} tokens saída/entrada
                    """)
                
                st.caption(f"🧠 Powered by Claude {modelo.split('-')[1].title() if modelo else ''} | Análise Premium")
            
            # Para análise offline avançada
            else:
                # Score de saúde
                score = resultado.get('score_saude', 0)
                if score >= 80:
                    st.success(f"✅ **Score de Saúde: {score}/100** - Projeto saudável")
                elif score >= 60:
                    st.warning(f"🟡 **Score de Saúde: {score}/100** - Atenção necessária")
                else:
                    st.error(f"🔴 **Score de Saúde: {score}/100** - Situação crítica")
                
                # Indicadores de cronograma
                cronograma = resultado.get('cronograma', {})
                if cronograma:
                    st.subheader("⏰ Indicadores de Cronograma")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Aderência", f"{cronograma['aderencia_prazo']}%")
                    with col2:
                        st.metric("Atrasadas", cronograma['atrasadas'])
                    with col3:
                        st.metric("Vencendo", cronograma['vencendo_semana'])
                    with col4:
                        st.metric("SLA Validação", f"{cronograma['sla_validacao']} dias")
                
                # Previsão de conclusão
                previsao = resultado.get('previsao_conclusao')
                if previsao:
                    st.subheader("🔮 Previsão de Conclusão")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.info(f"📅 **Planejado:** {previsao['data_planejada'].strftime('%d/%m/%Y')}")
                    with col2:
                        if previsao['dias_adicao'] > 0:
                            st.warning(f"⚠️ **Previsto:** {previsao['data_prevista'].strftime('%d/%m/%Y')} (+{previsao['dias_adicao']} dias)")
                        else:
                            st.success(f"✅ **Previsto:** {previsao['data_prevista'].strftime('%d/%m/%Y')} (no prazo)")
                
                # Recomendações
                recomendacoes = resultado.get('recomendacoes', [])
                if recomendacoes:
                    st.subheader("💡 Recomendações Prioritárias")
                    
                    for i, rec in enumerate(recomendacoes[:3], 1):
                        prioridade_emoji = {'crítica': '🔴', 'alta': '🟠', 'media': '🟡', 'baixa': '🟢'}
                        emoji = prioridade_emoji.get(rec['prioridade'], '🔵')
                        
                        with st.expander(f"{emoji} **{rec['categoria']}** - {rec['prioridade'].title()}", expanded=True):
                            st.write(f"**Ação:** {rec['acao']}")
                            st.write(f"**Impacto:** {rec['impacto']}")
                            if 'prazo' in rec:
                                st.write(f"**Prazo:** {rec['prazo']}")
                
                st.caption("🔧 Análise offline avançada com indicadores de cronograma | 100% PRIVADO")
                
        else:
            st.error(f"❌ {resultado.get('erro', 'Erro desconhecido')}")
            
            # Sugestões baseadas no tipo de erro
            erro_msg = resultado.get('erro', '')
            if 'API Key' in erro_msg or '401' in erro_msg:
                st.info("💡 **Dica:** Verifique se sua API Key da Anthropic está correta em console.anthropic.com")
            elif 'quota' in erro_msg.lower() or 'limit' in erro_msg.lower():
                st.info("💡 **Dica:** Limite de API atingido. Aguarde ou use a análise offline.")
            elif 'timeout' in erro_msg.lower():
                st.info("💡 **Dica:** Timeout na requisição. Tente novamente ou use modelo mais rápido (Haiku).")

# ============================================
# 📱 INTERFACE PRINCIPAL
# ============================================
def criar_grafico_dimensoes(df):
    """Cria gráfico de distribuição por Dimensões"""
    
    if df.empty or 'Dimensões' not in df.columns:
        return go.Figure().add_annotation(text="Campo 'Dimensões' não encontrado", x=0.5, y=0.5)
    
    # Remover valores nulos
    df_dimensoes = df.dropna(subset=['Dimensões'])
    
    if df_dimensoes.empty:
        return go.Figure().add_annotation(text="Nenhuma dimensão definida", x=0.5, y=0.5)
    
    # Contar atividades por dimensão
    dimensoes_count = df_dimensoes['Dimensões'].value_counts().reset_index()
    dimensoes_count.columns = ['Dimensões', 'Quantidade']
    
    # Cores personalizadas para dimensões
    cores_dimensoes = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']
    
    # Criar gráfico de barras horizontal
    fig = go.Figure([go.Bar(
        y=dimensoes_count['Dimensões'],
        x=dimensoes_count['Quantidade'],
        orientation='h',
        marker_color=cores_dimensoes[:len(dimensoes_count)],
        text=dimensoes_count['Quantidade'],
        textposition='auto',
        hovertemplate='<b>%{y}</b><br>Atividades: %{x}<br><extra></extra>'
    )])
    
    fig.update_layout(
        title='🎯 Distribuição por Dimensões',
        xaxis_title='Número de Atividades',
        yaxis_title='Dimensões',
        height=400,
        margin=dict(t=60, b=40, l=120, r=40)
    )
    
    return fig
    
def criar_grafico_dimensoes(df):
    """Cria gráfico de distribuição por Dimensões"""
    
    if df.empty or 'Dimensões' not in df.columns:
        return go.Figure().add_annotation(text="Campo 'Dimensões' não encontrado", x=0.5, y=0.5)
    
    # Remover valores nulos
    df_dimensoes = df.dropna(subset=['Dimensões'])
    
    if df_dimensoes.empty:
        return go.Figure().add_annotation(text="Nenhuma dimensão definida", x=0.5, y=0.5)
    
    # Contar atividades por dimensão
    dimensoes_count = df_dimensoes['Dimensões'].value_counts().reset_index()
    dimensoes_count.columns = ['Dimensões', 'Quantidade']
    
    # Cores personalizadas para dimensões
    cores_dimensoes = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']
    
    # Criar gráfico de barras horizontal
    fig = go.Figure([go.Bar(
        y=dimensoes_count['Dimensões'],
        x=dimensoes_count['Quantidade'],
        orientation='h',
        marker_color=cores_dimensoes[:len(dimensoes_count)],
        text=dimensoes_count['Quantidade'],
        textposition='auto',
        hovertemplate='<b>%{y}</b><br>Atividades: %{x}<br><extra></extra>'
    )])
    
    fig.update_layout(
        title='🎯 Distribuição por Dimensões',
        xaxis_title='Número de Atividades',
        yaxis_title='Dimensões',
        height=400,
        margin=dict(t=60, b=40, l=120, r=40)
    )
    
    return fig

def main():
    """Função principal do dashboard híbrido v5 com autenticação"""
    
    # ============================================
    # 🔐 VERIFICAÇÃO DE AUTENTICAÇÃO
    # ============================================
    
    if not auth.login_user():
        return
    
    # ============================================
    # 📊 DASHBOARD PRINCIPAL (Código original continua igual)
    # ============================================
    
    # Carregar dados
    df = carregar_dados()
    
    if df.empty:
        st.error("❌ Não foi possível carregar os dados do projeto!")
        st.stop()
    
    metricas = calcular_metricas(df)
    
    # Título principal
    st.title("📊 PMO - Digital Transformation Program")
    st.markdown("**Modelo Premium: Online (Claude) + Offline Avançado**")
    
    # Sidebar
    resultado_sidebar = criar_sidebar(df, metricas)
    df_filtrado = resultado_sidebar[0]
    api_key = resultado_sidebar[1] 
    modelo = resultado_sidebar[2]
    
    # Recalcular métricas com filtros
    if len(df_filtrado) != len(df):
        metricas_filtradas = calcular_metricas(df_filtrado)
    else:
        metricas_filtradas = metricas
    
    # ============================================
    # 📊 MÉTRICAS PRINCIPAIS EXPANDIDAS
    # ============================================
    
    st.markdown("---")
    st.subheader("📊 Métricas Principais")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
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
            "Concluídas", 
            concluidas
        )
    
    with col4:
        # Novo indicador: aderência aos prazos
        try:
            cronograma_info = analisador.calcular_indicadores_cronograma(df_filtrado)
            aderencia = cronograma_info['aderencia_prazo']
            st.metric(
                "Aderência Prazos",
                f"{aderencia}%",
                delta=f"{aderencia - 80}% vs Meta 80%"
            )
        except:
            st.metric("Aderência Prazos", "N/A")
    
    with col5:
        # Novo indicador: atividades atrasadas
        try:
            if 'cronograma_info' not in locals():
                cronograma_info = analisador.calcular_indicadores_cronograma(df_filtrado)
            atrasadas = cronograma_info['atrasadas']
            st.metric(
                "Atrasadas",
                atrasadas,
                delta="Crítico!" if atrasadas > 5 else "OK"
            )
        except:
            st.metric("Atrasadas", "N/A")
    
    # ============================================
    # 📊 NOVOS GRÁFICOS DE CRONOGRAMA
    # ============================================
    
    st.markdown("---")
    st.subheader("⏰ Análise de Cronograma")
    
    # Linha 1: Situação dos Prazos e Aderência
    col1, col2 = st.columns(2)
    
    with col1:
        try:
            fig_situacao = criar_grafico_situacao_prazos(df_filtrado)
            st.plotly_chart(fig_situacao, use_container_width=True)
        except Exception as e:
            st.error("⚠️ Erro no gráfico de situação")
    
    with col2:
        try:
            fig_aderencia = criar_gauge_aderencia(df_filtrado)
            st.plotly_chart(fig_aderencia, use_container_width=True)
        except Exception as e:
            st.error("⚠️ Erro no gauge de aderência")
    
    # Linha 2: Roadmap Melhorado (Planejado vs Realizado)
    st.subheader("🗺️ Roadmap: Planejado vs Realizado")
    try:
        fig_roadmap = criar_roadmap_planejado_vs_realizado(df_filtrado)
        st.plotly_chart(fig_roadmap, use_container_width=True)
    except Exception as e:
        st.warning(f"⚠️ Problema no roadmap: {str(e)}")
        st.info("🔧 Usando visualização alternativa:")
        
        # Fallback: Mostrar dados em formato tabular
        if not df_filtrado.empty and 'data_prevista' in df_filtrado.columns:
            try:
                df_roadmap_alt = df_filtrado.groupby('fase').agg({
                    'peso': 'sum',
                    'progresso': 'count',
                    'data_prevista': ['min', 'max']
                }).round(1)
                
                df_roadmap_alt.columns = ['Peso Total', 'Atividades', 'Data Início', 'Data Fim']
                df_roadmap_alt['Conclusão %'] = (df_roadmap_alt['Peso Total'] / df_roadmap_alt['Atividades'] * 100).round(1)
                
                st.dataframe(df_roadmap_alt, use_container_width=True)
            except:
                cronograma_info = analisador.calcular_indicadores_cronograma(df_filtrado)
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Atividades Atrasadas", cronograma_info['atrasadas'])
                with col2:
                    st.metric("Vencendo em 7 dias", cronograma_info['vencendo_semana'])
                with col3:
                    st.metric("Aderência aos Prazos", f"{cronograma_info['aderencia_prazo']}%")
    
    # ============================================
    # 📊 GRÁFICOS ORIGINAIS
    # ============================================
    
    st.markdown("---")
    st.subheader("📈 Visão Geral do Projeto")
    
    # Linha 1: Status e Progresso por Fase
    col1, col2 = st.columns(2)
    
    with col1:
        try:
            fig_rosca = criar_rosca_status(df_filtrado)
            st.plotly_chart(fig_rosca, use_container_width=True)
        except Exception as e:
            st.error("⚠️ Erro no gráfico de status")
    
    with col2:
        try:
            fig_barras = criar_barras_fase(df_filtrado)
            st.plotly_chart(fig_barras, use_container_width=True)
        except Exception as e:
         

    
    # ============================================
    # 🧠 ANÁLISE INTELIGENTE HÍBRIDA
    # ============================================
    
    widget_analise_hibrida(df_filtrado, metricas_filtradas, api_key, modelo)
    
    # ============================================
    # 📋 TABELA DETALHADA COM NOVAS COLUNAS
    # ============================================

    
    st.markdown("---")
    st.subheader("📋 Lista Detalhada de Atividades")
    
    if not df_filtrado.empty:
        # Processar datas para exibição
        try:
            df_tabela = analisador.processar_datas(df_filtrado).copy()
            
            # Preparar colunas para exibição
            colunas_exibir = ['id', 'fase', 'atividade', 'status', 'responsavel', 'data_prevista', 'progresso','delivery']
            
            # Adicionar coluna de situação do prazo
            if 'dias_para_prazo' in df_tabela.columns:
                def situacao_prazo(row):
                    if row['status'] == 'Concluído':
                        return '✅ Concluído'
                    elif pd.isna(row['dias_para_prazo']) or row['dias_para_prazo'] is None:
                        return '⚪ Sem data'
                    elif row['dias_para_prazo'] < 0:
                        return f'🔴 Atrasado {abs(int(row["dias_para_prazo"]))} dias'
                    elif row['dias_para_prazo'] <= 7:
                        return f'🟡 Vence em {int(row["dias_para_prazo"])} dias'
                    else:
                        return f'🟢 {int(row["dias_para_prazo"])} dias'
                
                df_tabela['situacao_prazo'] = df_tabela.apply(situacao_prazo, axis=1)
                colunas_exibir.append('situacao_prazo')
            
           
# Exibir tabela
            st.dataframe(
                df_tabela[colunas_exibir],
                column_config={
                    'id': 'ID',
                    'fase': 'Fase',
                    'atividade': 'Atividade',
                    'status': 'Status',
                    'responsavel': 'Responsável',
                    'data_prevista': st.column_config.DateColumn('Data Prevista'),
                    'progresso': st.column_config.ProgressColumn(
                        'Progresso',
                        help='Percentual de conclusão',
                        format='%d%%',
                        min_value=0,
                        max_value=100
                    ),
                    'delivery': 'Delivery',
                    'situacao_prazo': 'Situação do Prazo'
                },
                hide_index=True,
                use_container_width=True
            )
        except Exception as e:
            st.error(f"⚠️ Erro ao processar tabela: {e}")
            # Fallback: tabela simples sem processamento de datas
            colunas_basicas = ['id', 'fase', 'atividade', 'status', 'responsavel']
            if 'data_prevista' in df_filtrado.columns:
                colunas_basicas.append('data_prevista')
            if 'progresso' in df_filtrado.columns:
                colunas_basicas.append('progresso')
            if 'delivery' in df_filtrado.columns:
                colunas_basicas.append('delivery')
            
            st.dataframe(df_filtrado[colunas_basicas], use_container_width=True)
    # ============================================
    # 📈 RESUMO EXECUTIVO v5
    # ============================================
    
    st.markdown("---")
    st.subheader("📈 Resumo Executivo v5")
    
    with st.expander("🎯 Principais Insights", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **✅ Novidades v5 Premium:**
            - 🧠 Análise híbrida (Claude Premium + Offline)
            - ⏰ Indicadores de cronograma avançados
            - 📅 Timeline e roadmap visual
            - 🎯 Aderência aos prazos em tempo real
            - 🔮 Previsão de conclusão inteligente
            - 💰 Controle de custos por análise
            """)
        
        with col2:
            # Mostrar status atual do cronograma
            try:
                cronograma = analisador.calcular_indicadores_cronograma(df_filtrado)
                st.markdown(f"""
                **📊 Status Atual:**
                - ⏰ Aderência: {cronograma['aderencia_prazo']}%
                - 🔴 Atrasadas: {cronograma['atrasadas']}
                - 🟡 Vencendo: {cronograma['vencendo_semana']}
                - 📈 SLA Validação: {cronograma['sla_validacao']} dias
                """)
            except:
                st.markdown("""
                **📊 Status Atual:**
                - ⏰ Processando indicadores...
                - 🔄 Dados sendo analisados
                """)
    
    with st.expander("🚀 Como Usar o Dashboard v5"):
        st.markdown("""
        **🧠 Análise Online Premium (Claude):**
        1. Registre-se em **console.anthropic.com**
        2. Crie uma API Key (cobrança por uso)
        3. Cole sua API Key na sidebar
        4. Escolha o modelo Claude (Haiku/Sonnet/Opus)
        5. Clique em "Claude" para análise premium com IA
        
        **🔧 Análise Offline Avançada:**
        1. Sempre disponível (sem configuração)
        2. Clique em "Análise Offline"
        3. Receba insights baseados em regras avançadas
        4. Inclui novos indicadores de cronograma
        
        **⏰ Novos Indicadores de Cronograma v5:**
        - **Aderência aos Prazos:** % atividades dentro do prazo
        - **Timeline Visual:** Roadmap com datas reais
        - **Situação dos Prazos:** Atrasadas vs Futuras
        - **SLA de Validação:** Tempo médio de aprovação
        - **Previsão de Conclusão:** Data estimada final do projeto
        
        **💰 Custos Claude (referência):**
        - **Haiku:** ~$0.25/1M tokens (rápido e econômico)
        - **Sonnet:** ~$3/1M tokens (balanceado)
        - **Opus:** ~$15/1M tokens (máxima qualidade)
        
        **📊 Dica:** Use filtros na sidebar para análises específicas por fase, responsável ou status!
        """)


# ============================================
# 🚀 EXECUTAR APLICAÇÃO
# ============================================

if __name__ == "__main__":
    main()
