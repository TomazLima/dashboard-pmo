import streamlit as st
import pandas as pd
import plotly.graph_objects as go

st.set_page_config(page_title="PMO Dashboard", layout="wide")

st.title("📊 PMO Dashboard")

data = [
    ["1.1", "Kick-Off", "Concluído", "OS", 100],
    ["1.2", "Workshop", "Concluído", "OS", 100], 
    ["2.1", "Diagnóstico", "Concluído", "OS", 100],
    ["2.2", "Análise Gaps", "Aguardando Validação", "DT", 80],
    ["2.3", "Quick Wins", "Aguardando Validação", "DT", 80],
    ["3.1", "Sponsor", "Concluído", "OS", 100]
]

df = pd.DataFrame(data, columns=["ID", "Atividade", "Status", "Responsável", "Progresso"])

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Total", len(df))
with col2:
    completed = len(df[df["Status"] == "Concluído"])
    st.metric("Completed", completed)
with col3:
    rate = round(completed / len(df) * 100)
    st.metric("Rate", f"{rate}%")

status_counts = df["Status"].value_counts()
colors = ["#E20074", "#F899C9"]

fig = go.Figure(data=[go.Pie(
    labels=status_counts.index,
    values=status_counts.values,
    marker_colors=colors
)])

st.plotly_chart(fig, use_container_width=True)
st.dataframe(df, use_container_width=True)
st.success("Dashboard funcionando!")
