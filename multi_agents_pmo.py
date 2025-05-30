"""
🤖 MULTI-AGENTES PMO - CREWAI + ANTHROPIC
Módulo para integração de agentes inteligentes no Dashboard PMO
Versão: 1.0 - Integração Simples
"""

import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import json
import time
from typing import Dict, List, Optional
from dataclasses import dataclass
import os

# Imports do CrewAI (serão instalados: pip install crewai langchain-anthropic)
try:
    from crewai import Agent, Task, Crew, Process
    from langchain_anthropic import ChatAnthropic
    CREWAI_AVAILABLE = True
except ImportError:
    CREWAI_AVAILABLE = False

@dataclass
class AgentStatus:
    """Status do agente para exibição visual"""
    name: str
    role: str
    status: str  # 'idle', 'working', 'completed', 'error'
    last_execution: Optional[datetime] = None
    last_result: Optional[str] = None
    execution_time: Optional[float] = None
    avatar: str = "🤖"

class MultiAgentsPMO:
    """Orquestrador de Multi-Agentes para PMO"""
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key
        self.agents_status = {}
        self.crew = None
        self.llm = None
        
        # Configurar LLM se API key disponível
        if api_key and CREWAI_AVAILABLE:
            try:
                self.llm = ChatAnthropic(
                    model="claude-3-haiku-20240307",  # Usando Haiku por padrão (mais barato)
                    anthropic_api_key=api_key,
                    temperature=0.1
                )
                self._initialize_agents()
            except Exception as e:
                st.error(f"Erro ao inicializar agentes: {e}")
        
        # Inicializar status dos agentes
        self._initialize_agent_status()
    
    def _initialize_agent_status(self):
        """Inicializa status visual dos agentes"""
        agents_config = [
            {"name": "Data Analyst", "role": "Analista de Dados PMO", "avatar": "📊"},
            {"name": "Risk Monitor", "role": "Monitor de Riscos", "avatar": "⚠️"},
            {"name": "Report Generator", "role": "Gerador de Relatórios", "avatar": "📋"},
            {"name": "Forecast Agent", "role": "Agente de Previsões", "avatar": "🔮"}
        ]
        
        for config in agents_config:
            self.agents_status[config["name"]] = AgentStatus(
                name=config["name"],
                role=config["role"],
                status="idle",
                avatar=config["avatar"]
            )
    
    def _initialize_agents(self):
        """Inicializa os agentes CrewAI"""
        if not self.llm:
            return
        
        # Agente 1: Analista de Dados
        data_analyst = Agent(
            role="PMO Data Analyst",
            goal="Analisar dados de projetos e identificar padrões, tendências e insights quantitativos",
            backstory="""Você é um especialista em análise de dados de PMO com 10+ anos de experiência.
            Foca em métricas objetivas, KPIs e indicadores de performance de projetos.
            Sempre baseia conclusões em dados quantitativos e benchmarks de mercado.""",
            verbose=False,
            allow_delegation=False,
            llm=self.llm
        )
        
        # Agente 2: Monitor de Riscos
        risk_monitor = Agent(
            role="Risk Monitor Specialist",
            goal="Identificar riscos, alertas e problemas potenciais no projeto antes que se tornem críticos",
            backstory="""Especialista em gestão de riscos de projetos. Monitora constantemente
            indicadores de alerta precoce e identifica ameaças antes que se tornem problemas.
            Foca em cronogramas, recursos, dependências críticas e gargalos.""",
            verbose=False,
            allow_delegation=False,
            llm=self.llm
        )
        
        # Agente 3: Gerador de Relatórios
        report_generator = Agent(
            role="Executive Report Generator",
            goal="Criar resumos executivos claros e acionáveis para tomadores de decisão",
            backstory="""Especialista em comunicação executiva. Transforma dados complexos
            em insights claros para stakeholders C-level. Foca em recomendações
            práticas, acionáveis e priorizadas por impacto.""",
            verbose=False,
            allow_delegation=False,
            llm=self.llm
        )
        
        # Agente 4: Previsões
        forecast_agent = Agent(
            role="Project Forecasting Specialist",
            goal="Fazer previsões precisas sobre cronogramas, custos e probabilidades de sucesso",
            backstory="""Especialista em previsões de projetos usando dados históricos
            e tendências atuais. Utiliza métodos quantitativos para projetar cenários
            futuros e estimar probabilidades de sucesso com base em padrões.""",
            verbose=False,
            allow_delegation=False,
            llm=self.llm
        )
        
        self.agents = {
            "data_analyst": data_analyst,
            "risk_monitor": risk_monitor,
            "report_generator": report_generator,
            "forecast_agent": forecast_agent
        }
    
    def create_analysis_tasks(self, df: pd.DataFrame, metricas: dict) -> List[Task]:
        """Cria tasks para análise dos dados"""
        if not self.llm or df.empty:
            return []
        
        # Preparar resumo dos dados
        data_summary = self._prepare_data_summary(df, metricas)
        
        # Task 1: Análise de Dados
        data_analysis_task = Task(
            description=f"""Analise os seguintes dados do projeto PMO Digital Transformation:
            
            {data_summary}
            
            Como Data Analyst, forneça:
            1. Análise quantitativa dos KPIs principais (seja específico com números)
            2. Identificação de 2-3 tendências mais importantes nos dados
            3. Comparação com benchmarks típicos de projetos similares
            4. Top 3 métricas que precisam de atenção imediata
            
            Seja conciso, objetivo e focado em números. Máximo 150 palavras.""",
            agent=self.agents["data_analyst"],
            expected_output="Análise quantitativa com insights baseados em dados"
        )
        
        # Task 2: Monitoramento de Riscos
        risk_analysis_task = Task(
            description=f"""Com base nos dados do projeto PMO:
            
            {data_summary}
            
            Como Risk Monitor, identifique:
            1. Top 3 riscos mais críticos no momento (próximos 15-30 dias)
            2. Indicadores de alerta precoce sendo violados
            3. Dependências críticas que podem causar atrasos em cascata
            4. 2 ações preventivas específicas e urgentes
            
            Foque em riscos iminentes e acionáveis. Máximo 150 palavras.""",
            agent=self.agents["risk_monitor"],
            expected_output="Lista priorizada de riscos com ações preventivas específicas"
        )
        
        # Task 3: Previsões
        forecast_task = Task(
            description=f"""Baseado nos dados atuais do projeto:
            
            {data_summary}
            
            Como Forecasting Specialist, gere:
            1. Data mais provável de conclusão do projeto (seja específico)
            2. Probabilidade % de cumprir prazo original
            3. Cenário realista vs pessimista (datas específicas)
            4. Recursos/ações necessárias para acelerar entrega
            
            Use tendências atuais para projeções. Máximo 150 palavras.""",
            agent=self.agents["forecast_agent"],
            expected_output="Previsões quantitativas com cenários e datas específicas"
        )
        
        return [data_analysis_task, risk_analysis_task, forecast_task]
    
    def _prepare_data_summary(self, df: pd.DataFrame, metricas: dict) -> str:
        """Prepara resumo dos dados para os agentes"""
        try:
            # Resumo básico
            total = len(df)
            conclusao = metricas.get('conclusao_geral', 0)
            
            # Status breakdown
            status_counts = df['status'].value_counts().to_dict()
            
            # Responsáveis
            resp_counts = df['responsavel'].value_counts().head(3).to_dict()
            
            # Fases
            fase_counts = df['fase'].value_counts().to_dict()
            
            # Cronograma (tentar importar do módulo principal)
            cronograma_info = ""
            try:
                # Tentar calcular indicadores de cronograma
                if 'data_prevista' in df.columns:
                    hoje = datetime.now().date()
                    df_temp = df.copy()
                    df_temp['data_prevista'] = pd.to_datetime(df_temp['data_prevista'], errors='coerce').dt.date
                    df_temp['dias_para_prazo'] = df_temp['data_prevista'].apply(
                        lambda x: (x - hoje).days if pd.notna(x) and x is not None else None
                    )
                    
                    atrasadas = len(df_temp[(df_temp['dias_para_prazo'] < 0) & (df_temp['status'] != 'Concluído')])
                    vencendo = len(df_temp[(df_temp['dias_para_prazo'] >= 0) & (df_temp['dias_para_prazo'] <= 7) & (df_temp['status'] != 'Concluído')])
                    
                    cronograma_info = f"""
                    INDICADORES DE CRONOGRAMA:
                    - Atividades atrasadas: {atrasadas}
                    - Atividades vencendo em 7 dias: {vencendo}
                    - Data de hoje: {hoje.strftime('%d/%m/%Y')}
                    """
                else:
                    cronograma_info = "Dados de cronograma limitados"
            except:
                cronograma_info = "Cronograma: dados não disponíveis para análise detalhada"
            
            summary = f"""
            PROJETO: Digital Transformation Program PMO
            
            SITUAÇÃO ATUAL:
            - Total de atividades: {total}
            - Progresso geral: {conclusao}%
            - Distribuição por status: {status_counts}
            - Principais responsáveis: {resp_counts}
            - Fases do projeto: {fase_counts}
            
            {cronograma_info}
            
            CONTEXTO: Este é um programa de transformação digital corporativa gerenciado via PMO, 
            com múltiplas atividades, responsáveis e entregas distribuídas ao longo do tempo.
            """
            
            return summary
            
        except Exception as e:
            return f"DADOS BÁSICOS DISPONÍVEIS:\n- Total atividades: {len(df)}\n- Erro no processamento: {str(e)}"
    
    def execute_analysis(self, df: pd.DataFrame, metricas: dict) -> dict:
        """Executa análise com multi-agentes"""
        
        if not CREWAI_AVAILABLE:
            return {
                "success": False,
                "error": "CrewAI não disponível. Execute: pip install crewai langchain-anthropic"
            }
        
        if not self.llm:
            return {
                "success": False,
                "error": "API Key da Anthropic necessária para usar multi-agentes"
            }
        
        try:
            # Atualizar status dos agentes
            for agent_name in self.agents_status:
                self.agents_status[agent_name].status = "working"
                self.agents_status[agent_name].last_execution = datetime.now()
            
            start_time = time.time()
            
            # Criar tasks
            tasks = self.create_analysis_tasks(df, metricas)
            
            if not tasks:
                return {"success": False, "error": "Não foi possível criar tasks de análise"}
            
            # Criar e executar crew
            crew = Crew(
                agents=list(self.agents.values())[:3],  # Usar apenas 3 agentes para economizar
                tasks=tasks,
                verbose=False,
                process=Process.sequential
            )
            
            # Executar análise
            result = crew.kickoff()
            
            execution_time = time.time() - start_time
            
            # Atualizar status dos agentes
            for agent_name in self.agents_status:
                self.agents_status[agent_name].status = "completed"
                self.agents_status[agent_name].execution_time = execution_time
                self.agents_status[agent_name].last_result = "Análise concluída com sucesso"
            
            return {
                "success": True,
                "result": result,
                "execution_time": execution_time,
                "agents_used": 3,  # Usamos 3 agentes
                "tasks_completed": len(tasks)
            }
            
        except Exception as e:
            # Atualizar status para erro
            for agent_name in self.agents_status:
                self.agents_status[agent_name].status = "error"
                self.agents_status[agent_name].last_result = f"Erro: {str(e)}"
            
            return {
                "success": False,
                "error": str(e)
            }
    
    def get_agents_status(self) -> Dict[str, AgentStatus]:
        """Retorna status atual dos agentes"""
        return self.agents_status
    
    def simulate_agent_activity(self):
        """Simula atividade dos agentes para demonstração"""
        import random
        
        for agent_name, status in self.agents_status.items():
            # Simular atividade aleatória
            if random.random() < 0.3:  # 30% chance de mudança
                current_status = status.status
                if current_status == "idle":
                    status.status = "working"
                elif current_status == "working":
                    status.status = "completed"
                    status.last_execution = datetime.now()
                    status.last_result = f"Análise simulada concluída para {agent_name}"


# ============================================
# 🎨 COMPONENTES VISUAIS PARA STREAMLIT
# ============================================

def render_agents_sidebar(multi_agents: MultiAgentsPMO):
    """Renderiza sidebar com status dos agentes"""
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("🤖 AI Agents")
    
    agents_status = multi_agents.get_agents_status()
    
    for agent_name, status in agents_status.items():
        # Container para cada agente
        with st.sidebar.container():
            col1, col2 = st.sidebar.columns([1, 3])
            
            with col1:
                st.write(status.avatar)
            
            with col2:
                st.write(f"**{status.name}**")
                
                # Status com cores
                if status.status == "idle":
                    st.caption("🔵 Idle")
                elif status.status == "working":
                    st.caption("🟡 Working...")
                elif status.status == "completed":
                    st.caption("🟢 Completed")
                elif status.status == "error":
                    st.caption("🔴 Error")
                
                # Última execução
                if status.last_execution:
                    time_ago = datetime.now() - status.last_execution
                    if time_ago.seconds < 60:
                        st.caption(f"Last: {time_ago.seconds}s ago")
                    else:
                        st.caption(f"Last: {time_ago.seconds//60}m ago")

def render_agents_status_widget(multi_agents: MultiAgentsPMO):
    """Widget compacto de status dos agentes para página principal"""
    
    with st.expander("🤖 AI Agents Status", expanded=False):
        agents_status = multi_agents.get_agents_status()
        
        # Status resumido em colunas
        cols = st.columns(len(agents_status))
        
        for i, (agent_name, status) in enumerate(agents_status.items()):
            with cols[i]:
                # Status emoji
                status_emoji = {
                    "idle": "🔵",
                    "working": "🟡", 
                    "completed": "🟢",
                    "error": "🔴"
                }
                
                st.markdown(f"""
                <div style="text-align: center; padding: 10px; border: 1px solid #ddd; border-radius: 8px;">
                    <div style="font-size: 24px;">{status.avatar}</div>
                    <div style="font-size: 11px; font-weight: bold;">{status.name}</div>
                    <div style="font-size: 18px;">{status_emoji[status.status]}</div>
                </div>
                """, unsafe_allow_html=True)
        
        # Informações adicionais
        st.markdown("**Legend:** 🔵 Idle | 🟡 Working | 🟢 Completed | 🔴 Error")
        
        if any(status.last_execution for status in agents_status.values()):
            last_run = max((status.last_execution for status in agents_status.values() if status.last_execution))
            time_since = datetime.now() - last_run
            st.caption(f"Last multi-agent analysis: {time_since.seconds//60}m ago")


# ============================================
# 🔧 FUNÇÕES DE INTEGRAÇÃO
# ============================================

def integrate_agents_to_dashboard(df: pd.DataFrame, metricas: dict, api_key: str = None):
    """Função principal para integrar agentes ao dashboard existente"""
    
    # Inicializar multi-agentes
    if 'multi_agents' not in st.session_state:
        st.session_state.multi_agents = MultiAgentsPMO(api_key)
    
    # Atualizar API key se mudou
    if api_key and st.session_state.multi_agents.api_key != api_key:
        st.session_state.multi_agents = MultiAgentsPMO(api_key)
    
    return st.session_state.multi_agents


# ============================================
# 📦 VERIFICAÇÃO DE DEPENDÊNCIAS
# ============================================

def check_dependencies():
    """Verifica dependências necessárias"""
    missing = []
    
    try:
        import crewai
    except ImportError:
        missing.append("crewai")
    
    try:
        import langchain_anthropic
    except ImportError:
        missing.append("langchain-anthropic")
    
    if missing:
        st.error(f"❌ Dependências em falta: {', '.join(missing)}")
        st.info("Execute no terminal: pip install " + " ".join(missing))
        return False
    
    return True


# ============================================
# 🎯 TESTE RÁPIDO (se executar este arquivo diretamente)
# ============================================

if __name__ == "__main__":
    st.title("🤖 Multi-Agents PMO - Teste")
    
    # Verificar dependências
    if not check_dependencies():
        st.stop()
    
    # Demo básico
    api_key = st.text_input("API Key Anthropic:", type="password")
    
    if api_key:
        multi_agents = MultiAgentsPMO(api_key)
        
        # Dados dummy para teste
        demo_data = pd.DataFrame([
            {"id": "1", "fase": "Início", "atividade": "Setup", "status": "Concluído", "responsavel": "Team A", "data_prevista": "2025-01-15"},
            {"id": "2", "fase": "Desenvolvimento", "atividade": "Coding", "status": "Em Andamento", "responsavel": "Team B", "data_prevista": "2025-06-30"}
        ])
        
        demo_metricas = {"conclusao_geral": 65, "total_atividades": 2}
        
        # Mostrar widget
        render_agents_status_widget(multi_agents)
        
        # Teste de execução
        if st.button("🧪 Testar Multi-Agentes"):
            with st.spinner("Testando agentes..."):
                result = multi_agents.execute_analysis(demo_data, demo_metricas)
            
            if result["success"]:
                st.success("✅ Teste bem-sucedido!")
                st.write(result["result"])
            else:
                st.error(f"❌ Erro: {result['error']}")
    else:
        st.info("👆 Configure sua API Key da Anthropic para testar")