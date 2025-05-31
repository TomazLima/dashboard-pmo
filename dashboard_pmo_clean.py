import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta
import openpyxl

# ============================================
# 🔐 IMPORT AUTHENTICATION
# ============================================
import auth

# ============================================
# 🔧 PAGE CONFIGURATION
# ============================================

st.set_page_config(
    page_title="PMO - Digital Transformation Program",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================
# 📊 LOAD AND PROCESS DATA
# ============================================

@st.cache_data
def load_data():
    """Load data from Excel file"""
    try:
        df = pd.read_excel('atividades_pmo.xlsx', sheet_name='Atividades')
        
        # Validate required columns
        required_columns = ['id', 'fase', 'atividade', 'status', 'responsavel', 'data_prevista', 'progresso', 'peso', 'delivery', 'Dimensões']
        
        if not all(col in df.columns for col in required_columns):
            st.error("❌ Excel file doesn't have the correct columns!")
            st.info("Expected columns: " + ", ".join(required_columns))
            return pd.DataFrame()
        
        # Remove empty rows
        df = df.dropna(subset=['id'])
        
        return df
        
    except FileNotFoundError:
        st.warning("⚠️ File 'atividades_pmo.xlsx' not found! Using sample data.")
        return load_sample_data()
        
    except Exception as e:
        st.error(f"❌ Error reading Excel: {e}")
        return load_sample_data()

def load_sample_data():
    """Sample data for demonstration"""
    sample_data = [
        {"id": "1.1", "fase": "Initial Milestone", "atividade": "Kick-Off", "status": "Concluído", "responsavel": "OS", "data_prevista": "2025-02-12", "progresso": 100, "peso": 1.0, "observacoes": "Done", "delivery": "Sprint 1", "Dimensões": "Process"},
        {"id": "2.1", "fase": "Phase 1", "atividade": "Diagnosis", "status": "Aguardando Validação", "responsavel": "DT", "data_prevista": "2025-04-07", "progresso": 80, "peso": 0.8, "observacoes": "Awaiting validation", "delivery": "Sprint 2", "Dimensões": "Technology"},
        {"id": "2.2", "fase": "Phase 1", "atividade": "Gap Analysis", "status": "Em Andamento", "responsavel": "OS", "data_prevista": "2025-05-30", "progresso": 50, "peso": 0.5, "observacoes": "In progress", "delivery": "Sprint 3", "Dimensões": "People"},
        {"id": "3.1", "fase": "Phase 2", "atividade": "Structuring", "status": "Identificado", "responsavel": "To Be Defined", "data_prevista": "2025-06-30", "progresso": 0, "peso": 0.0, "observacoes": "Pending", "delivery": "Sprint 4", "Dimensões": "Governance"},
    ]
    return pd.DataFrame(sample_data)

def calculate_metrics(df):
    """Calculate project metrics"""
    if df.empty:
        return {
            'total_activities': 0,
            'overall_completion': 0,
            'status_counts': pd.Series(dtype=int),
            'phase_stats': pd.DataFrame(),
            'resp_stats': pd.DataFrame()
        }
    
    total_activities = len(df)
    overall_completion = round((df['peso'].sum() / total_activities) * 100)
    
    # Count by status
    status_counts = df['status'].value_counts()
    
    # Analysis by phase
    phase_stats = df.groupby('fase').agg({
        'peso': ['sum', 'count'],
        'progresso': 'mean'
    }).round(1)
    
    phase_stats.columns = ['total_weight', 'quantity', 'avg_progress']
    phase_stats['phase_completion'] = (phase_stats['total_weight'] / phase_stats['quantity'] * 100).round(1)
    
    # Analysis by responsible
    resp_stats = df.groupby(['responsavel', 'status']).size().unstack(fill_value=0)
    
    return {
        'total_activities': total_activities,
        'overall_completion': overall_completion,
        'status_counts': status_counts,
        'phase_stats': phase_stats,
        'resp_stats': resp_stats
    }

def process_dates(df):
    """Process dates for timeline analysis"""
    if df.empty or 'data_prevista' not in df.columns:
        return df
    
    df = df.copy()
    today = datetime.now().date()
    
    # Convert data_prevista to datetime
    try:
        df['data_prevista'] = pd.to_datetime(df['data_prevista']).dt.date
        df['days_to_deadline'] = df['data_prevista'].apply(
            lambda x: (x - today).days if pd.notna(x) and x is not None else None
        )
    except Exception as e:
        st.warning(f"⚠️ Error processing dates: {e}")
        df['days_to_deadline'] = None
        
    return df

def calculate_schedule_indicators(df):
    """Calculate schedule-based indicators"""
    if df.empty:
        return {}
    
    df = process_dates(df)
    
    # Filter only activities with dates
    df_with_dates = df.dropna(subset=['data_prevista'])
    
    if df_with_dates.empty:
        return {
            'total_with_dates': 0,
            'overdue': 0,
            'due_this_week': 0,
            'future': 0,
            'schedule_adherence': 0,
            'avg_delay_days': 0
        }
    
    total_with_dates = len(df_with_dates)
    
    # Activities by deadline situation
    overdue = len(df_with_dates[
        (df_with_dates['days_to_deadline'] < 0) & 
        (df_with_dates['status'] != 'Concluído')  # 🔧 CORREÇÃO: status em português
    ])
    
    due_this_week = len(df_with_dates[
        (df_with_dates['days_to_deadline'] >= 0) & 
        (df_with_dates['days_to_deadline'] <= 7) &
        (df_with_dates['status'] != 'Concluído')  # 🔧 CORREÇÃO: status em português
    ])
    
    future = len(df_with_dates[
        (df_with_dates['days_to_deadline'] > 7) &
        (df_with_dates['status'] != 'Concluído')  # 🔧 CORREÇÃO: status em português
    ])
    
    # Schedule adherence (% of activities on time)
    completed = len(df_with_dates[df_with_dates['status'] == 'Concluído'])  # 🔧 CORREÇÃO: status em português
    on_time = len(df_with_dates[
        (df_with_dates['status'] != 'Concluído') &  # 🔧 CORREÇÃO: status em português
        (df_with_dates['days_to_deadline'] >= 0)
    ])
    
    schedule_adherence = round(((completed + on_time) / total_with_dates) * 100) if total_with_dates > 0 else 0
    
    # Average delay days
    overdue_activities = df_with_dates[
        (df_with_dates['days_to_deadline'] < 0) & 
        (df_with_dates['status'] != 'Concluído')  # 🔧 CORREÇÃO: status em português
    ]
    
    if len(overdue_activities) > 0:
        avg_delay = overdue_activities['days_to_deadline'].mean()
        avg_delay_days = round(abs(avg_delay)) if not pd.isna(avg_delay) else 0
    else:
        avg_delay_days = 0
    
    return {
        'total_with_dates': total_with_dates,
        'overdue': overdue,
        'due_this_week': due_this_week,
        'future': future,
        'schedule_adherence': schedule_adherence,
        'avg_delay_days': avg_delay_days
    }

# ============================================
# 🎨 SIDEBAR
# ============================================

def create_sidebar(df, metrics):
    """Create sidebar with filters and metrics"""
    
    st.sidebar.title("🎯 PMO - Digital Transformation")
    st.sidebar.markdown("**Project Management Dashboard**")
    
    st.sidebar.markdown("---")
    
    # Main metrics
    st.sidebar.metric("Total Activities", metrics['total_activities'])
    st.sidebar.metric(
        "Overall Completion", 
        f"{metrics['overall_completion']}%",
        delta=f"{metrics['overall_completion'] - 50}% vs Target 50%"
    )
    
    # Schedule indicators
    if not df.empty:
        try:
            schedule = calculate_schedule_indicators(df)
            
            st.sidebar.metric(
                "Schedule Adherence",
                f"{schedule['schedule_adherence']}%",
                delta=f"{schedule['schedule_adherence'] - 80}% vs Target 80%"
            )
            
            if schedule['overdue'] > 0:
                st.sidebar.metric(
                    "Overdue Activities",
                    schedule['overdue'],
                    delta=f"-{schedule['avg_delay_days']} avg days"
                )
        except Exception as e:
            st.sidebar.warning("⚠️ Error in schedule indicators")
    
    st.sidebar.markdown("---")
    
    # Filters
    st.sidebar.subheader("🔍 Filters")
    
    if df.empty:
        st.sidebar.warning("⚠️ No data available")
        return df
    
    selected_phases = st.sidebar.multiselect(
        "Select Phases:",
        options=df['fase'].unique(),
        default=df['fase'].unique()
    )
    
    selected_status = st.sidebar.multiselect(
        "Select Status:",
        options=df['status'].unique(),
        default=df['status'].unique()
    )
    
    selected_responsible = st.sidebar.multiselect(
        "Select Responsible:",
        options=df['responsavel'].unique(),
        default=df['responsavel'].unique()
    )
    
    # Apply filters
    df_filtered = df[
        (df['fase'].isin(selected_phases)) &
        (df['status'].isin(selected_status)) &
        (df['responsavel'].isin(selected_responsible))
    ]
    
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"**Updated:** {datetime.now().strftime('%m/%d/%Y %H:%M')}")
    
    return df_filtered

# ============================================
# 📊 CHARTS
# ============================================

def create_status_donut(df):
    """Create donut chart for status distribution"""
    
    if df.empty:
        return go.Figure().add_annotation(text="No data available", x=0.5, y=0.5)
    
    status_counts = df['status'].value_counts()
    
    # 🔧 CORREÇÃO: Mapeamento de cores para status em português
    status_colors = {
        'Concluído': '#E20074',           # Magenta principal - Completed
        'Em Andamento': '#F066A7',        # Magenta claro - In Progress
        'Aguardando Validação': '#F899C9', # Magenta pastel - Awaiting Validation
        'Identificado': '#8E8E93'         # Cinza moderno - Identified
    }
    
    colors = [status_colors.get(status, '#CCCCCC') for status in status_counts.index]
    
    fig = go.Figure(data=[go.Pie(
        labels=status_counts.index,
        values=status_counts.values,
        hole=0.6,
        marker_colors=colors,
        textinfo='label+percent',
        textfont_size=10,
        textposition="outside",
        hovertemplate='<b>%{label}</b><br>Quantity: %{value}<br>Percentage: %{percent}<extra></extra>',
        pull=[0.05 if status == 'Concluído' else 0.02 for status in status_counts.index]  # 🔧 CORREÇÃO: destaque para "Concluído"
    )])
    
    # Center text
    total = len(df)
    completion = round((df['peso'].sum() / total) * 100)
    
    fig.add_annotation(
        text=f"<b style='font-size:20px'>{total}</b><br><span style='font-size:12px'>ACTIVITIES</span><br><br><b style='font-size:18px'>{completion}%</b><br><span style='font-size:11px'>COMPLETION</span>",
        x=0.5, y=0.5,
        font_size=14,
        showarrow=False
    )
    
    fig.update_layout(
        title='🍩 Activity Status',
        showlegend=True,
        height=450,
        margin=dict(t=60, b=60, l=60, r=60),
        legend=dict(
            orientation="v",
            yanchor="middle",
            y=0.5,
            xanchor="left",
            x=1.05
        )
    )
    
    return fig

def create_phase_bars(df):
    """Create horizontal bar chart by phase"""
    
    if df.empty:
        return go.Figure().add_annotation(text="No data available", x=0.5, y=0.5)
    
    phase_stats = df.groupby('fase').agg({
        'peso': ['sum', 'count']
    }).round(1)
    
    phase_stats.columns = ['total_weight', 'quantity']
    phase_stats['phase_completion'] = (phase_stats['total_weight'] / phase_stats['quantity'] * 100).round(1)
    
    # Colors by level
    colors = []
    for completion in phase_stats['phase_completion']:
        if completion >= 70:
            colors.append('#E20074')  # Main magenta
        elif completion >= 30:
            colors.append('#F066A7')  # Light magenta
        else:
            colors.append('#8E8E93')  # Gray
    
    fig = go.Figure([go.Bar(
        x=phase_stats['phase_completion'],
        y=phase_stats.index,
        orientation='h',
        marker_color=colors,
        text=[f'{c:.0f}%' for c in phase_stats['phase_completion']],
        textposition='inside'
    )])
    
    fig.update_layout(
        title='📊 Progress by Phase',
        xaxis_title='Completion Percentage (%)',
        height=400,
        margin=dict(t=60, b=40, l=200, r=40)
    )
    
    fig.add_vline(x=50, line_dash="dash", line_color="gray", 
                  annotation_text="Target 50%")
    
    return fig

def create_dimensions_chart(df):
    """Create chart for dimensions distribution"""
    
    if df.empty or 'Dimensões' not in df.columns:
        return go.Figure().add_annotation(text="'Dimensions' field not found", x=0.5, y=0.5)
    
    # Remove null values
    df_dimensions = df.dropna(subset=['Dimensões'])
    
    if df_dimensions.empty:
        return go.Figure().add_annotation(text="No dimensions defined", x=0.5, y=0.5)
    
    # Count activities by dimension
    dimensions_count = df_dimensions['Dimensões'].value_counts().reset_index()
    dimensions_count.columns = ['Dimensions', 'Quantity']
    
    # Custom colors for dimensions
    dimension_colors = [
        '#E20074',  # Main magenta
        '#F066A7',  # Light magenta
        '#F899C9',  # Pastel magenta
        '#FCCCEB',  # Very light magenta
        '#8E8E93',  # Modern gray
        '#C7C7CC',  # Light gray
        '#F2F2F7'   # Very light gray
    ]
    
    # Create horizontal bar chart
    fig = go.Figure([go.Bar(
        y=dimensions_count['Dimensions'],
        x=dimensions_count['Quantity'],
        orientation='h',
        marker_color=dimension_colors[:len(dimensions_count)],
        text=dimensions_count['Quantity'],
        textposition='auto',
        hovertemplate='<b>%{y}</b><br>Activities: %{x}<br><extra></extra>'
    )])
    
    fig.update_layout(
        title='🎯 Distribution by Dimensions',
        xaxis_title='Number of Activities',
        yaxis_title='Dimensions',
        height=400,
        margin=dict(t=60, b=40, l=120, r=40)
    )
    
    return fig

def create_schedule_situation_chart(df):
    """Create chart showing schedule situation"""
    
    if df.empty:
        return go.Figure().add_annotation(text="No data available", x=0.5, y=0.5)
    
    schedule = calculate_schedule_indicators(df)
    
    # Data for chart
    situations = ['Overdue', 'Due (7 days)', 'Future']
    values = [schedule['overdue'], schedule['due_this_week'], schedule['future']]
    colors = ['#8E8E93', '#F066A7', '#E20074']  # Gray, Light magenta, Main magenta
    
    fig = go.Figure([go.Bar(
        x=situations,
        y=values,
        marker_color=colors,
        text=values,
        textposition='auto'
    )])
    
    fig.update_layout(
        title='🚨 Schedule Situation',
        xaxis_title='Situation',
        yaxis_title='Number of Activities',
        height=400,
        margin=dict(t=60, b=40, l=40, r=40)
    )
    
    return fig

def create_adherence_gauge(df):
    """Create gauge for schedule adherence"""
    
    if df.empty:
        adherence = 0
    else:
        schedule = calculate_schedule_indicators(df)
        adherence = schedule['schedule_adherence']
    
    # Determine bar color
    if adherence >= 90:
        bar_color = "#E20074"      # Main magenta
        status = "Excellent"
        status_color = "#E20074"
    elif adherence >= 70:
        bar_color = "#F066A7"      # Light magenta
        status = "Good"
        status_color = "#F066A7"
    elif adherence >= 50:
        bar_color = "#F899C9"      # Pastel magenta
        status = "Attention"
        status_color = "#F899C9"
    else:
        bar_color = "#8E8E93"      # Gray
        status = "Critical"
        status_color = "#8E8E93"
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=adherence,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': f"⏰ Schedule Adherence<br><span style='font-size:0.6em; color:gray'>% of activities on time</span>", 'font': {'size': 18}},
        number={'font': {'size': 48, 'color': '#E20074'}, 'suffix': '%'},
        gauge={
            'axis': {'range': [None, 100], 'tickwidth': 1},
            'bar': {'color': bar_color},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 50], 'color': '#F2F2F7'},    # Very light gray
                {'range': [50, 70], 'color': '#FCCCEB'},   # Very light magenta
                {'range': [70, 90], 'color': '#F899C9'},   # Pastel magenta
                {'range': [90, 100], 'color': '#F066A7'}   # Light magenta
            ],
            'threshold': {
                'line': {'color': "#1e3a8a", 'width': 4},
                'thickness': 0.75,
                'value': 85
            }
        }
    ))
    
    # Add explanatory text in center
    fig.add_annotation(
        text=f"<b>{status}</b>",
        x=0.5, y=0.2,
        font_size=16,
        font_color=status_color,
        showarrow=False
    )
    
    fig.update_layout(height=400, margin=dict(t=80, b=40, l=40, r=40))
    return fig

def create_timeline_roadmap(df):
    """Create timeline roadmap by phase"""
    
    if df.empty:
        return go.Figure().add_annotation(text="No data available", x=0.5, y=0.5)
    
    try:
        # Process data with robust checks
        df_processed = process_dates(df)
        
        if 'data_prevista' not in df_processed.columns:
            return go.Figure().add_annotation(text="'data_prevista' field not found", x=0.5, y=0.5)
        
        # Remove rows with null dates
        df_valid = df_processed.dropna(subset=['data_prevista']).copy()
        
        if df_valid.empty:
            return go.Figure().add_annotation(text="No valid dates found", x=0.5, y=0.5)
        
        # Convert explicitly to datetime
        df_valid['data_dt'] = pd.to_datetime(df_valid['data_prevista'], errors='coerce')
        df_valid = df_valid.dropna(subset=['data_dt'])
        
        if df_valid.empty:
            return go.Figure().add_annotation(text="Error in date conversion", x=0.5, y=0.5)
        
        # Group by phase
        phase_summary = []
        
        for phase in df_valid['fase'].unique():
            df_phase = df_valid[df_valid['fase'] == phase]
            
            # Calculate min/max dates of phase
            start_date = df_phase['data_dt'].min()
            end_date = df_phase['data_dt'].max()
            
            # Calculate phase progress
            total_activities = len(df_phase)
            total_weight = df_phase['peso'].sum()
            phase_progress = (total_weight / total_activities) * 100 if total_activities > 0 else 0
            
            phase_summary.append({
                'phase': phase,
                'start_date': start_date,
                'end_date': end_date,
                'progress': min(phase_progress, 100),  # Limit to 100%
                'total_activities': total_activities
            })
        
        if not phase_summary:
            return go.Figure().add_annotation(text="No phases processed", x=0.5, y=0.5)
        
        # Create chart
        fig = go.Figure()
        
        for i, phase_info in enumerate(phase_summary):
            phase = phase_info['phase']
            start_date = phase_info['start_date']
            end_date = phase_info['end_date']
            progress = phase_info['progress']
            
            # Gray line - planned schedule (100%)
            fig.add_trace(go.Scatter(
                x=[start_date, end_date],
                y=[phase, phase],
                mode='lines',
                line=dict(width=20, color='rgba(180, 180, 180, 0.7)'),
                name='Planned' if i == 0 else '',
                showlegend=True if i == 0 else False,
                hovertemplate=f'<b>Planned</b><br>Phase: {phase}<br>Complete period<extra></extra>'
            ))
            
            # Calculate progress endpoint
            total_duration = (end_date - start_date).total_seconds()
            progress_duration = total_duration * (progress / 100)
            progress_date = start_date + pd.Timedelta(seconds=progress_duration)
            
            # Determine color based on progress
            if progress >= 80:
                progress_color = '#28a745'  # Green
                status = 'On time'
            elif progress >= 50:
                progress_color = '#ffc107'  # Yellow
                status = 'Attention'
            else:
                progress_color = '#dc3545'  # Red
                status = 'Delayed'
            
            # Colored line - actual progress
            fig.add_trace(go.Scatter(
                x=[start_date, progress_date],
                y=[phase, phase],
                mode='lines',
                line=dict(width=20, color=progress_color),
                name='Actual' if i == 0 else '',
                showlegend=True if i == 0 else False,
                hovertemplate=f'<b>Actual</b><br>Phase: {phase}<br>Progress: {progress:.1f}%<br>Status: {status}<extra></extra>'
            ))
            
            # Marker with percentage
            fig.add_trace(go.Scatter(
                x=[progress_date],
                y=[phase],
                mode='markers+text',
                marker=dict(size=15, color=progress_color, symbol='circle'),
                text=[f'{progress:.0f}%'],
                textposition="middle right",
                textfont=dict(size=12, color='black', family='Arial Black'),
                showlegend=False,
                hovertemplate=f'<b>{phase}</b><br>Current progress: {progress:.1f}%<extra></extra>'
            ))
        
        # Add vertical line for "TODAY"
        try:
            today = pd.Timestamp.now()
            fig.add_vline(
                x=today,
                line_dash="dash",
                line_color="#E20074",
                line_width=3,
                annotation_text="TODAY",
                annotation_position="top",
                annotation_font_size=12,
                annotation_font_color="#E20074"
            )
        except:
            pass
        
        # Chart layout
        fig.update_layout(
            title='🗺️ Timeline: Planned vs Actual',
            height=400,
            margin=dict(t=80, b=40, l=200, r=100),
            xaxis_title="Timeline",
            yaxis_title="Phases",
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
        error_msg = f"Error creating roadmap: {str(e)}"
        return go.Figure().add_annotation(
            text="Error processing roadmap<br>Check input data", 
            x=0.5, y=0.5,
            font_size=14
        )

def deadline_situation_with_colors(row):
    """Function to determine deadline situation with colors"""
    if row['status'] == 'Concluído':  # 🔧 CORREÇÃO: status em português
        return '✅ Completed'
    elif pd.isna(row['days_to_deadline']) or row['days_to_deadline'] is None:
        return '⚪ No date'
    elif row['days_to_deadline'] < 0:
        return f'🔴 Overdue {abs(int(row["days_to_deadline"]))} days'
    elif row['days_to_deadline'] <= 7:
        return f'🟡 Due in {int(row["days_to_deadline"])} days'
    else:
        return f'🟢 {int(row["days_to_deadline"])} days'

# ============================================
# 📱 MAIN INTERFACE
# ============================================

def main():
    """Main dashboard function"""
    
    # ============================================
    # 🔐 AUTHENTICATION CHECK
    # ============================================
    
    if not auth.login_user():
        return
    
    # ============================================
    # 📊 MAIN DASHBOARD
    # ============================================
    
    # Load data
    df = load_data()
    
    if df.empty:
        st.error("❌ Could not load project data!")
        st.stop()
    
    metrics = calculate_metrics(df)
    
    # Main title
    st.title("📊 PMO - Digital Transformation Program")
    st.markdown("**Project Management Dashboard - Visualization Focus**")
    
    # Sidebar
    df_filtered = create_sidebar(df, metrics)
    
    # Recalculate metrics with filters
    if len(df_filtered) != len(df):
        filtered_metrics = calculate_metrics(df_filtered)
    else:
        filtered_metrics = metrics
    
    # ============================================
    # 📊 MAIN METRICS
    # ============================================
    
    st.markdown("---")
    st.subheader("📊 Key Metrics")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric(
            "Total Activities", 
            filtered_metrics['total_activities']
        )
    
    with col2:
        completion = filtered_metrics['overall_completion']
        delta_completion = completion - 50
        st.metric(
            "Overall Completion", 
            f"{completion}%",
            delta=f"{delta_completion:+.0f}% vs Target 50%"
        )
    
    with col3:
        # 🔧 CORREÇÃO: Procurar por "Concluído" ao invés de "Completed"
        completed = filtered_metrics['status_counts'].get('Concluído', 0)
        st.metric(
            "Completed", 
            completed
        )
    
    with col4:
        # Schedule adherence indicator
        try:
            schedule_info = calculate_schedule_indicators(df_filtered)
            adherence = schedule_info['schedule_adherence']
            st.metric(
                "Schedule Adherence",
                f"{adherence}%",
                delta=f"{adherence - 80}% vs Target 80%"
            )
        except:
            st.metric("Schedule Adherence", "N/A")
    
    with col5:
        # Overdue activities indicator
        try:
            if 'schedule_info' not in locals():
                schedule_info = calculate_schedule_indicators(df_filtered)
            overdue = schedule_info['overdue']
            st.metric(
                "Overdue",
                overdue,
                delta=f"-{overdue}" if overdue > 0 else "OK"
            )
        except:
            st.metric("Overdue", "N/A")

    # ============================================
    # ⏰ SCHEDULE ANALYSIS
    # ============================================
    
    st.markdown("---")
    st.subheader("⏰ Schedule Analysis")
    
    # Row 1: Schedule Situation and Adherence
    col1, col2 = st.columns(2)
    
    with col1:
        try:
            fig_situation = create_schedule_situation_chart(df_filtered)
            st.plotly_chart(fig_situation, use_container_width=True)
        except Exception as e:
            st.error("⚠️ Error in situation chart")
    
    with col2:
        try:
            fig_adherence = create_adherence_gauge(df_filtered)
            st.plotly_chart(fig_adherence, use_container_width=True)
        except Exception as e:
            st.error("⚠️ Error in adherence gauge")
    
    # Row 2: Improved Timeline (Planned vs Actual)
    st.subheader("🗺️ Timeline: Planned vs Actual")
    try:
        fig_timeline = create_timeline_roadmap(df_filtered)
        st.plotly_chart(fig_timeline, use_container_width=True)
    except Exception as e:
        st.warning(f"⚠️ Timeline issue: {str(e)}")
        st.info("🔧 Using alternative visualization:")
        
        # Fallback: Show data in tabular format
        if not df_filtered.empty and 'data_prevista' in df_filtered.columns:
            try:
                df_timeline_alt = df_filtered.groupby('fase').agg({
                    'peso': 'sum',
                    'progresso': 'count',
                    'data_prevista': ['min', 'max']
                }).round(1)
                
                df_timeline_alt.columns = ['Total Weight', 'Activities', 'Start Date', 'End Date']
                df_timeline_alt['Completion %'] = (df_timeline_alt['Total Weight'] / df_timeline_alt['Activities'] * 100).round(1)
                
                st.dataframe(df_timeline_alt, use_container_width=True)
            except:
                schedule_info = calculate_schedule_indicators(df_filtered)
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Overdue Activities", schedule_info['overdue'])
                with col2:
                    st.metric("Due in 7 days", schedule_info['due_this_week'])
                with col3:
                    st.metric("Schedule Adherence", f"{schedule_info['schedule_adherence']}%")
    
    # ============================================
    # 📊 PROJECT OVERVIEW CHARTS
    # ============================================
    
    st.markdown("---")
    st.subheader("📈 Project Overview")
    
    # Row 1: Status and Progress by Phase
    col1, col2 = st.columns(2)
    
    with col1:
        try:
            fig_donut = create_status_donut(df_filtered)
            st.plotly_chart(fig_donut, use_container_width=True)
        except Exception as e:
            st.error("⚠️ Error in status chart")
    
    with col2:
        try:
            fig_bars = create_phase_bars(df_filtered)
            st.plotly_chart(fig_bars, use_container_width=True)
        except Exception as e:
            st.error("⚠️ Error in phase chart")
    
    # Row 2: Dimensions Analysis
    st.subheader("🎯 Analysis by Dimensions")
    try:
        fig_dimensions = create_dimensions_chart(df_filtered)
        st.plotly_chart(fig_dimensions, use_container_width=True)
    except Exception as e:
        st.error(f"⚠️ Error in dimensions chart: {e}")
    
    # ============================================
    # 📋 DETAILED TABLE
    # ============================================
    
    st.markdown("---")
    st.subheader("📋 Detailed Activity List")
    
    if not df_filtered.empty:
        # Process dates for display
        try:
            df_table = process_dates(df_filtered).copy()
            
            # Columns to display
            display_columns = ['id', 'fase', 'atividade', 'status', 'responsavel', 'data_prevista', 'progresso', 'delivery', 'Dimensões']
            
            # Add deadline situation column
            if 'days_to_deadline' in df_table.columns:
                df_table['deadline_situation'] = df_table.apply(deadline_situation_with_colors, axis=1)
                display_columns.append('deadline_situation')
           
            # Display table
            st.dataframe(
                df_table[display_columns],
                column_config={
                    'id': 'ID',
                    'fase': 'Phase',
                    'atividade': 'Activity',
                    'status': 'Status',
                    'responsavel': 'Responsible',
                    'data_prevista': st.column_config.DateColumn('Expected Date'),
                    'progresso': st.column_config.ProgressColumn(
                        'Progress',
                        help='Completion percentage',
                        format='%d%%',
                        min_value=0,
                        max_value=100
                    ),
                    'delivery': 'Delivery',
                    'Dimensões': 'Dimensions',
                    'deadline_situation': 'Deadline Situation'
                },
                hide_index=True,
                use_container_width=True
            )
        except Exception as e:
            st.error(f"⚠️ Error processing table: {e}")
            # Fallback: simple table
            basic_columns = ['id', 'fase', 'atividade', 'status', 'responsavel']
            if 'data_prevista' in df_filtered.columns:
                basic_columns.append('data_prevista')
            if 'progresso' in df_filtered.columns:
                basic_columns.append('progresso')
            if 'delivery' in df_filtered.columns:
                basic_columns.append('delivery')
            if 'Dimensões' in df_filtered.columns:
                basic_columns.append('Dimensões')
            
            st.dataframe(df_filtered[basic_columns], use_container_width=True)
    
    # ============================================
    # 📈 EXECUTIVE SUMMARY
    # ============================================
    
    st.markdown("---")
    st.subheader("📈 Executive Summary")
    
    with st.expander("🎯 Key Insights", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **✅ Dashboard Features:**
            - 📊 Visual project metrics
            - ⏰ Advanced schedule indicators
            - 📅 Visual timeline and roadmap
            - 🎯 Real-time deadline adherence
            - 🔮 Project completion tracking
            - 📱 Responsive design for all devices
            """)
        
        with col2:
            # Show current schedule status
            try:
                schedule = calculate_schedule_indicators(df_filtered)
                st.markdown(f"""
                **📊 Current Status:**
                - ⏰ Adherence: {schedule['schedule_adherence']}%
                - 🔴 Overdue: {schedule['overdue']}
                - 🟡 Due Soon: {schedule['due_this_week']}
                - 📈 Future Activities: {schedule['future']}
                """)
            except:
                st.markdown("""
                **📊 Current Status:**
                - ⏰ Processing indicators...
                - 🔄 Data being analyzed
                """)
    
    with st.expander("🚀 How to Use the Dashboard"):
        st.markdown("""
        **📊 Navigation:**
        1. Use the **sidebar filters** to analyze specific phases, status, or responsible parties
        2. Monitor **schedule adherence** in real-time
        3. Track **project timeline** with planned vs actual progress
        4. Analyze **activity distribution** by dimensions
        
        **⏰ Schedule Indicators:**
        - **Schedule Adherence:** % of activities on time
        - **Visual Timeline:** Roadmap with actual dates
        - **Deadline Situation:** Overdue vs Future activities
        - **Progress Tracking:** Visual progress by phase
        
        **📊 Chart Features:**
        - **Status Distribution:** Visual breakdown of activity status
        - **Phase Progress:** Horizontal progress bars by phase
        - **Dimensions Analysis:** Activity distribution by business dimension
        - **Timeline Roadmap:** Planned vs actual timeline visualization
        
        **📋 Data Table:**
        - **Complete view** of all activities with filtering
        - **Deadline indicators** with color coding
        - **Export capabilities** for further analysis
        
        **💡 Tips:** 
        - Use filters in the sidebar for specific analysis
        - Hover over charts for detailed information
        - Check the detailed table for comprehensive activity view
        """)


# ============================================
# 🚀 RUN APPLICATION
# ============================================

if __name__ == "__main__":
    main()