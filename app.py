import streamlit as st
import pandas as pd
import nltk
import base64
import io
from nltk.corpus import stopwords
from wordcloud import WordCloud
import numpy as np
import networkx as nx

import plotly.express as px
import plotly.graph_objects as go

# Check if stopwords are already downloaded, if not, download them
try:
    stopwords.words('spanish')
except LookupError:
    nltk.download('stopwords')

# Preload mask and stopwords outside the function if reused
stopwords_es = set(stopwords.words('spanish'))

def plot_donut_chart(df):
    # Calculate value counts
    value_counts = df['Tipo expediente'].value_counts()
    
    # Create a donut chart using Plotly
    fig = px.pie(
        values=value_counts.values,
        names=value_counts.index,
        title='Cantidad de expedientes por tipo',
        hole=0.3  # Makes it a donut chart
    )
    
    # Update layout to show class names and values on hover
    fig.update_traces(
        hovertemplate='Tipo EE: %{label}<br>Cantidad: %{value}',  # Show label and value on hover in different lines
        textinfo='none',   
        showlegend=False    
    )
    
    # Display the figure
    st.plotly_chart(fig)

def plot_donut_chart_two(df):
    # Calculate value counts
    value_counts = df['Estado'].value_counts()
    
    # Create a donut chart using Plotly
    fig = px.pie(
        values=value_counts.values,
        names=value_counts.index,
        title='Estado de los expediente a la fecha',
        hole=0.3  # Makes it a donut chart
    )
    
    # Update layout to show class names and values on hover
    fig.update_traces(
        hovertemplate='Estado: %{label}<br>Cantidad: %{value}',  # Show label and percentage on hover in different lines
        textinfo='percent',   
        showlegend=True
    )

    st.plotly_chart(fig)

def create_wordcloud(df, column_name):
    # Generate a cache key based on the column name and data content to avoid regenerating the figure unnecessarily
    cache_key = f"wordcloud_{column_name}_{hash(tuple(df[column_name]))}"

    # Check if the figure is already cached in session state
    if cache_key in st.session_state:
        st.plotly_chart(st.session_state[cache_key], use_container_width=True)
        return

    # Create a string of all words from the specified column
    words = ' '.join(df[column_name].dropna())
    
    # Create a list of Spanish stopwords
    stopwords_es = stopwords.words('spanish')
    
    # Generate the word cloud
    wordcloud = WordCloud(
        width=1600,  # Larger width for better resolution
        height=1200,  # Larger height for better resolution
        background_color='white',
        stopwords=stopwords_es,
        colormap='rainbow'  # Set the color palette to rainbow
    ).generate(words)
    
    # Convert word cloud image to a base64 string
    buffer = io.BytesIO()
    wordcloud.to_image().save(buffer, format='PNG')
    img_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
    
    # Create a Plotly figure
    fig = go.Figure()
    fig.add_layout_image(
        dict(
            source=f'data:image/png;base64,{img_str}',
            x=0,
            y=1,
            sizex=1.5,  # Adjust to control the width
            sizey=1.2,  # Adjust to control the height
            xanchor='left',
            yanchor='top',
            opacity=1
        )
    )
    
    # Update the layout to remove axes and title, and set the figure size
    fig.update_layout(
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        title=f'Principales términos en {column_name}',
        title_x=0.0,  # Left justify the title
        title_y=0.95,
        width=450,  # Set the width of the figure in Streamlit
        height=450,  # Set the height of the figure in Streamlit
        margin=dict(l=0, r=0, t=50, b=0)  # Remove any extra margins
    )
    
    # Store the figure in session state for future use
    st.session_state[cache_key] = fig
    
    # Show the plot in Streamlit
    st.plotly_chart(fig, use_container_width=True)

def plot_cant_dias_distribution(df, percentil):
    # Select Cant. días column
    cant_dias = df['Cant. días']

    # Calculate the threshold for the given percentile
    threshold = cant_dias.quantile(percentil)

    # Create the histogram
    fig = go.Figure()

    # Add traces for values below and above the threshold
    fig.add_trace(go.Histogram(
        x=cant_dias[cant_dias <= threshold],
        xbins=dict(size=2),
        name='Por debajo',
        marker_color='blue'
    ))

    fig.add_trace(go.Histogram(
        x=cant_dias[cant_dias > threshold],
        xbins=dict(size=2),
        name='Por encima',
        marker_color='red'
    ))

    # Update layout
    fig.update_layout(
        title='Distribución de la duración de los expedientes',
        xaxis_title='Duración en días',
        yaxis_title='Cantidad de expedientes',
        barmode='overlay'
    )

    # Return the plot 
    return fig

def plot_activity_by_day_of_week(df):
    df['Fecha C.'] = pd.to_datetime(df['Fecha creación'])
    df['Fecha P.'] = pd.to_datetime(df['Fecha pase'])
    
    # Count the occurrences of each day of the week for 'Fecha C.' and 'Fecha P.'
    counts_c = df['Fecha C.'].dt.dayofweek.value_counts().sort_index()
    counts_p = df['Fecha P.'].dt.dayofweek.value_counts().sort_index()

    # Ensure both counts include all days of the week (0=Monday, ..., 6=Sunday)
    counts_c = counts_c.reindex(range(7), fill_value=0)
    counts_p = counts_p.reindex(range(7), fill_value=0)

    # Create the grouped bar chart
    fig = go.Figure()

    # Add the traces
    fig.add_trace(
        go.Bar(
            x=['Lun', 'Mar', 'Mié', 'Jue', 'Vie', 'Sáb', 'Dom'],
            y=counts_c,
            name='Creaciones'
        )
    )
    
    fig.add_trace(
        go.Bar(
            x=['Lun', 'Mar', 'Mié', 'Jue', 'Vie', 'Sáb', 'Dom'],
            y=counts_p,
            name='Pases'
        )
    )

    # Update layout
    fig.update_layout(
        title='Promedio de actuaciones por día de la semana',
        xaxis_title='Día de la semana',
        yaxis_title='Cantidad de expedientes',
        barmode='group',  # Grouped bar chart
        xaxis=dict(
            tickvals=list(range(7)),
            ticktext=['Lun', 'Mar', 'Mié', 'Jue', 'Vie', 'Sáb', 'Dom']
        )
    )

    # Show the plot
    return fig

def plot_activity_by_hour(df):
    # Make a copy of the DataFrame
    df_copy = df.copy()

    # Convert 'Fecha creación' and 'Fecha pase' to datetime
    df_copy['Fecha C.'] = pd.to_datetime(df_copy['Fecha creación'])
    df_copy['Fecha P.'] = pd.to_datetime(df_copy['Fecha pase'])

    # Create hour columns
    df_copy['Hora C.'] = df_copy['Fecha C.'].dt.hour
    df_copy['Hora P.'] = df_copy['Fecha P.'].dt.hour

    # Group the counts by hour of the day and fill missing values with 0
    counts_c_hour = df_copy.groupby('Hora C.').size().reindex(range(24), fill_value=0)
    counts_p_hour = df_copy.groupby('Hora P.').size().reindex(range(24), fill_value=0)

    # Filter out the hours with zero counts in both datasets
    non_zero_hours = (counts_c_hour > 0) | (counts_p_hour > 0)
    counts_c_hour = counts_c_hour[non_zero_hours]
    counts_p_hour = counts_p_hour[non_zero_hours]

    # Drop temporary columns
    df_copy.drop(columns=['Fecha C.', 'Fecha P.', 'Hora C.', 'Hora P.'], inplace=True)

    # Create the stacked area plot
    fig = go.Figure()

    # Add traces for creation and pase
    fig.add_trace(
        go.Scatter(
            x=counts_c_hour.index,
            y=counts_c_hour,
            fill='tozeroy',
            name='Creaciones'
        )
    )
    
    fig.add_trace(
        go.Scatter(
            x=counts_p_hour.index,
            y=counts_p_hour,
            fill='tonexty',
            name='Pases'
        )
    )

    # Update layout
    fig.update_layout(
        title='Promedio de actuaciones por hora',
        xaxis_title='Hora del día',
        yaxis_title='Cantidad de expedientes',
        xaxis=dict(
            tickmode='array',
            tickvals=list(range(24)),
            ticktext=[f'{(h % 12) or 12}{" AM" if h < 12 else " PM"}' for h in range(24)],
            tickangle=90
        ),
        yaxis=dict(
            autorange=True
        ),
        legend_title='Tipo de actividad'
    )

    # Show the plot
    return fig

def plot_top_nodes_graph(df, top_n):
    # Group by source and destination and count occurrences
    edge_weights = df.groupby(['oficina_origen', 'oficina_destino']).size().reset_index(name='weight')

    # Rename column names to source and destination
    edge_weights.columns = ['source', 'destination', 'weights']

    # Create graph 
    G = nx.DiGraph()

    # Add edges to the graph
    for i, elrow in edge_weights.iterrows():
        G.add_edge(elrow['source'], elrow['destination'], weight=elrow['weights'])

    # Add node attribute from 'cantidad_creaciones'
    node_creations = df.groupby('oficina_origen')['cantidad_creaciones'].first().to_dict()

    # Calculate in-degree and out-degree weighted values (total in+out weighted degree)
    node_weights = {}
    for node in G.nodes():
        in_weight = sum([d['weight'] for u, v, d in G.in_edges(node, data=True)])
        out_weight = sum([d['weight'] for u, v, d in G.out_edges(node, data=True)])
        total_weight = in_weight + out_weight
        node_weights[node] = total_weight

    # Sort nodes by total weight and select the top 'top_n'
    top_nodes = sorted(node_weights, key=node_weights.get, reverse=True)[:top_n]

    # Create a subgraph with only the top nodes
    G_top = G.subgraph(top_nodes)

    # Use Kamada-Kawai layout for better distribution
    pos = nx.kamada_kawai_layout(G_top)

    # Initialize lists for nodes and edges
    node_x = []
    node_y = []
    node_text = []
    node_color = []
    node_size = []
    edge_x = []
    edge_y = []

    # Calculate hover information and graph data for nodes and edges
    for node in G_top.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)

        # Sum of incoming and outgoing weights for top nodes
        in_weight = sum([d['weight'] for u, v, d in G_top.in_edges(node, data=True)])
        out_weight = sum([d['weight'] for u, v, d in G_top.out_edges(node, data=True)])
        total_weight = in_weight + out_weight

        # Add 'cantidad_creaciones' to the hover information
        node_creation_count = node_creations.get(node, 0)

        # Hover text including node name, in_weight, out_weight, and cantidad_creaciones
        hover_text = (
            f"Departamento: {node}<br>"
            f"Pases entrantes: {in_weight}<br>"
            f"Pases salientes: {out_weight}<br>"
            f"Expedientes creados: {node_creation_count}"
        )
        node_text.append(hover_text)

        # Append total_weight to node_color and node_size to use for visualization
        node_color.append(total_weight)
        node_size.append(total_weight * 0.005)  # Adjust size scaling factor as needed

    # Add edges to edge_x and edge_y for visualization
    for edge in G_top.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.append(x0)
        edge_x.append(x1)
        edge_x.append(None)
        edge_y.append(y0)
        edge_y.append(y1)
        edge_y.append(None)

    # Create the Plotly graph object for edges
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=1, color='rgba(136, 136, 136, 0.3)'),
        hoverinfo='none',
        mode='lines'
    )

    # Create the Plotly graph object for nodes
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        marker=dict(
            size=node_size,  
            color=node_color,  
            colorscale='Rainbow',
            colorbar=dict(
                title='Total de pases',  
            ),
            cmin=min(node_color),
            cmax=max(node_color),
            showscale=True
        ),
        text=node_text
    )

    # Set up the layout for the plot
    layout = go.Layout(
        title=f"Top {top_n} departamentos con más pases",
        title_x=0.5,
        showlegend=False,
        hovermode='closest',
        margin=dict(b=0, l=0, r=0, t=40),
        xaxis=dict(showgrid=True, zeroline=False),
        yaxis=dict(showgrid=False, zeroline=False)
    )

    # Create figure and display it
    fig = go.Figure(data=[edge_trace, node_trace], layout=layout)
    st.plotly_chart(fig)

def display_dossier_count(df):
    # Count the number of rows in the DataFrame
    dossier_count = len(df)
    
    # Display a message with the dossier count
    st.markdown(f"#### Total de expedientes creados: **{dossier_count}**")

def plot_activity_by_month_area(df):
    df = df.copy()

    # Convert 'Fecha creación' and 'Fecha pase' to datetime format
    df['Fecha C.'] = pd.to_datetime(df['Fecha creación'])
    df['Fecha P.'] = pd.to_datetime(df['Fecha pase'])
    
    # Extract month and year from 'Fecha C.' and 'Fecha P.'
    df['Mes C.'] = df['Fecha C.'].dt.to_period('M')
    df['Mes P.'] = df['Fecha P.'].dt.to_period('M')
    
    # Count occurrences by month and fill missing values with 0
    counts_c_month = df['Mes C.'].value_counts().sort_index()
    counts_p_month = df['Mes P.'].value_counts().sort_index()

    # Create a figure for the area chart
    fig = go.Figure()

    # Add traces for Creaciones and Pases
    fig.add_trace(
        go.Scatter(
            x=counts_c_month.index.astype(str),
            y=counts_c_month,
            fill='tozeroy',
            name='Creaciones'
        )
    )
    
    fig.add_trace(
        go.Scatter(
            x=counts_p_month.index.astype(str),
            y=counts_p_month,
            fill='tonexty',
            name='Pases'
        )
    )

    # Update layout
    fig.update_layout(
        title='Actividad en el último año agrupado por mes',
        xaxis_title='Mes',
        yaxis_title='Cantidad',
        xaxis=dict(
            tickvals=counts_c_month.index.astype(str),
            ticktext=counts_c_month.index.strftime('%b %Y')  # Format as abbreviated month and year
        ),
        yaxis=dict(
            autorange=True
        ),
        legend_title='Tipo de actividad'
    )

    # Show the plot
    return fig

def plot_pivot_table(df, tipo):
    # Filter df by selected Tipo expediente
    df_filtered = df[df['Tipo expediente'] == tipo]

    # Groupby 'Oficina creadora' and calculate average Cant. días 
    pivot_table = df_filtered.groupby('Oficina creadora')['Cant. días'].mean().reset_index()

    # Round the average to the next integer
    pivot_table['Cant. días'] = pivot_table['Cant. días'].apply(np.ceil)

    # Detect outliers with IQR method
    Q1 = pivot_table['Cant. días'].quantile(0.25)
    Q3 = pivot_table['Cant. días'].quantile(0.75)
    IQR = Q3 - Q1
    upper_bound = Q3 + 1.5 * IQR

    # Color the outliers in red and the rest in blue
    colors = ['red' if x > upper_bound else 'blue' for x in pivot_table['Cant. días']] 

    # Create the bar chart
    fig = go.Figure(data=[go.Bar(
        x=pivot_table['Oficina creadora'],
        y=pivot_table['Cant. días'],
        marker_color=colors
    )])

    # Add title to the chart
    fig.update_layout(title="Tiempo total de duración de un expediente hasta su archivamiento por oficina creadora")

    st.plotly_chart(fig)

def display_timespan(df):
    # Calculate the timespan between 'Fecha creación' and 'Fecha pase'
    df['Fecha C.'] = pd.to_datetime(df['Fecha creación'])
    tiempo = df['Fecha C.'].max() - df['Fecha C.'].min()

    st.markdown(f"#### Rango de expedientes: **{tiempo.days} días**")

def pivot_table_graph(df):
    # Create a pivot table, index is 'Ubicación actual', columns is 'Estado', values is count of 'Oficina creadora'
    pivot_table = df.pivot_table(index='Ubicación actual', columns='Estado', values='Oficina creadora', aggfunc='count', fill_value=0)

    # Reset the index to turn it into a normal column
    pivot_table.reset_index(inplace=True)

    return pivot_table

def plot_stacked_graphs(df):
     # Check if 'Ubicación actual' column exists
    if 'Ubicación actual' not in df.columns:
        raise ValueError("The DataFrame must contain a 'Ubicación actual' column.")
    
    # Filter for numerical columns
    numerical_cols = df.select_dtypes(include=['number']).columns
    
    # Create a new DataFrame for plotting
    plot_df = df[numerical_cols].copy()
    
    # Add the 'Ubicación actual' column to the DataFrame
    plot_df['Ubicación actual'] = df['Ubicación actual']
    
    # Melt the DataFrame to long format for Plotly
    plot_df = plot_df.melt(id_vars='Ubicación actual', var_name='Variable', value_name='Value')
    
    # Create the stacked bar plot
    fig = px.bar(plot_df, x='Ubicación actual', y='Value', color='Variable', 
                 title='Gráficos de barras apiladas para filas seleccionadas',
                 labels={'Ubicación actual': 'Ubicación Actual', 'Value': 'Valor', 'Variable': 'Columnas Numéricas'},
                 barmode='stack')
    
    # Show the plot
    return fig

def display_general_info(df, df_archived):
    with st.expander("🔍 Ver detalles", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            display_dossier_count(df)
            st.markdown("#### **Tiempo de vida promedio de un expediente**: {:.2f} días".format(df_archived['Cant. días'].mean()))
        with col2:
            display_timespan(df)
            st.markdown("#### **Ratio de pases vs creaciones**: {:.2f}".format(len(df) / len(df_two)))

# Set initial configurations
st.set_page_config(layout="wide", page_title="INE ", page_icon=':bar_chart:')

# Display the image
col1, col2= st.columns(2)
with col1:
    st.image("assets/ine.png", width=300)
with col2:
    st.image("assets/virtu.png", width=250)

# Title and dashboard header
st.title(":bar_chart: Analitica de uso de expedientes en la institución")

# Load data 
df = pd.read_excel('Consulta_global.xls', skiprows=31)
counts_df = df['Oficina creadora'].value_counts()
df_two = pd.read_csv('historico_de_pases.csv')
merged_df = df_two.merge(counts_df, left_on='oficina_origen', right_on='Oficina creadora', how='left')
merged_df.fillna(0, inplace=True)
merged_df.columns = ['oficina_origen', 'oficina_destino', 'tipo_expediente', 'urgente', 'tiempo', 'cantidad_creaciones']
df_archived = df[df['Estado'] == 'ARCHIVADO']

# Only show the expander if df is not None (i.e., after file upload)
if df is not None:

    display_general_info(df, df_archived)

    # Create columns for the plots 
    col1, col2= st.columns(2)

    with col1:
        plot_donut_chart(df)
        st.caption("Esta función calcula la cantidad de ocurrencias de cada valor en la columna 'Tipo expediente' de los datos obtenidos")
        st.plotly_chart(plot_activity_by_day_of_week(df))
        plot_donut_chart_two(df)
        st.plotly_chart(plot_activity_by_month_area(df))

    with col2:
        create_wordcloud(df, 'Asunto')
        st.caption("Esta función crea una nube de palabras a partir de los valores de la columna 'Asunto' de los datos obtenidos")
        st.plotly_chart(plot_activity_by_hour(df))
    
        # Create a placeholder for the plot
        plot_placeholder = st.empty()

        # Slider for percentile
        percentil = st.slider("Seleccione un percentil para la distribución de la duración de los expedientes", 0.0, 1.0, 0.95, 0.05)

        # Update the plot in the placeholder with the selected percentile
        plot_placeholder.plotly_chart(plot_cant_dias_distribution(df, percentil))

    st.divider()

    # Add a title for the network graph
    plot_top_nodes_graph(merged_df, 50)

    # Space out the components
    for _ in range(7):
        st.write(" ")

    # Create a layout with many columns
    col1, col2= st.columns([4, 1])

    # In the fourth column, display the selectbox and the caption
    with col2:
        unique_tipos = df['Tipo expediente'].unique()
        selected_tipo = st.selectbox("Seleccione un Tipo expediente", unique_tipos, key="tipo_exp")
        st.caption("Esta tabla muestra el tiempo de respuesta promedio por tipo de trámite (seleccionado arriba) para cada oficina creadora")

    # In the first three columns, display the pivot table
    with col1:
        plot_pivot_table(df, selected_tipo)

    # Space out the components
    for _ in range(7):
        st.write(" ")

    # Add a title for the last graph
    st.markdown("###### **Ubicación Actual de los expedientes agrupados por Estado y Oficina Actual**")

    # Load the dataset and create a pivot table
    pivot_table = pivot_table_graph(df)
    
    # Display the pivot table with a row selection option
    event = st.dataframe(
        pivot_table,
        use_container_width=True,
        hide_index=True,
        on_select="rerun",
        selection_mode="multi-row",
    )

    # Get the indices of selected rows
    selected_rows = event.selection.rows

    # If there are selected rows, filter the pivot_table DataFrame
    if selected_rows:
        selected_df = pivot_table.iloc[selected_rows]
        st.plotly_chart(plot_stacked_graphs(selected_df))
    else:
        st.write("No rows selected.")

# Eliminate the need to re-run the entire script, save hashed graphs in the session state ?