"""
Red Social - Interfaz Streamlit (SIN compilación CUDA)
Lee resultados pre-generados de un archivo JSON
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json
from pathlib import Path

# Configuración de la página
st.set_page_config(
    page_title="Red Social CUDA",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilos personalizados
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1E88E5;
        text-align: center;
        padding: 1rem 0;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #1E88E5;
    }
    </style>
""", unsafe_allow_html=True)

# Título principal
st.markdown('<p class="main-header">🚀 Red Social con CUDA</p>', unsafe_allow_html=True)
st.markdown("---")

# Sidebar con controles
with st.sidebar:
    st.header("⚙️ Configuración")

    st.subheader("📂 Cargar Datos")

    # Opción 1: Archivo JSON subido
    uploaded_file = st.file_uploader(
        "Sube el archivo resultados.json generado en Colab",
        type=['json']
    )

    # Opción 2: Archivo local
    st.markdown("**O usa archivo local:**")
    json_path = st.text_input(
        "Ruta del archivo JSON",
        value="resultados.json"
    )

    if st.button("📥 Cargar desde archivo local", type="primary"):
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                st.session_state['data'] = data
                st.success("✓ Datos cargados correctamente!")
                st.rerun()
        except FileNotFoundError:
            st.error(f"Archivo no encontrado: {json_path}")
        except json.JSONDecodeError:
            st.error("Error al leer el archivo JSON")
        except Exception as e:
            st.error(f"Error: {str(e)}")

    # Procesar archivo subido
    if uploaded_file is not None:
        try:
            data = json.load(uploaded_file)
            st.session_state['data'] = data
            st.success("✓ Archivo cargado correctamente!")
        except Exception as e:
            st.error(f"Error al cargar archivo: {str(e)}")

    st.markdown("---")

    st.subheader("📊 Visualizaciones")
    view_option = st.radio(
        "Selecciona una vista:",
        [
            "📈 Dashboard General",
            "👥 Seguidores",
            "❤️ Reacciones",
            "🏆 Top Publicaciones",
            "#️⃣ Hashtags",
            "🚫 Usuarios Bloqueados",
            "💼 Recomendaciones Empresas",
            "📄 Output Completo"
        ]
    )

# Contenido principal
if 'data' not in st.session_state:
    st.info("👈 Carga un archivo JSON con los resultados del análisis CUDA")

    # Información sobre el proyecto
    with st.expander("ℹ️ Cómo usar esta aplicación"):
        st.markdown("""
        ### Pasos para usar:

        1. **En Google Colab:**
           - Ejecuta tu código CUDA
           - Al final del notebook, descarga el archivo `resultados.json`

        2. **En esta aplicación local:**
           - Sube el archivo JSON usando el botón de arriba
           - O colócalo en la misma carpeta y cárgalo

        3. **Explora:**
           - Usa la barra lateral para ver diferentes visualizaciones

        ### Ventajas de este método:
        - ✅ No necesitas GPU en tu PC local
        - ✅ No necesitas instalar CUDA Toolkit
        - ✅ Ejecuta CUDA en Colab, visualiza localmente
        - ✅ Puedes guardar y reutilizar resultados
        """)

    with st.expander("🚀 Generar resultados.json en Colab"):
        st.markdown("""
        Ejecuta esto en Google Colab al final de tu notebook:

        ```python
        import json
        from google.colab import files

        # Guardar datos en JSON
        with open('resultados.json', 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        # Descargar el archivo
        files.download('resultados.json')
        ```
        """)

else:
    data = st.session_state['data']

    # Dashboard General
    if view_option == "📈 Dashboard General":
        st.header("📈 Dashboard General")

        # Métricas principales
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            total_personas = len(data['seguidores']['personas'])
            st.metric("👥 Personas", total_personas)

        with col2:
            total_empresas = len(data['seguidores']['empresas'])
            st.metric("💼 Empresas", total_empresas)

        with col3:
            total_posts = len(data['reacciones'])
            st.metric("📝 Publicaciones", total_posts)

        with col4:
            total_hashtags = len(data['hashtags']['conteo'])
            st.metric("#️⃣ Hashtags", total_hashtags)

        st.markdown("---")

        # Gráficos en dos columnas
        col1, col2 = st.columns(2)

        with col1:
            # Top 5 personas por seguidores
            st.subheader("🏆 Top 5 Personas - Seguidores")
            personas_df = pd.DataFrame(data['seguidores']['personas'])
            if not personas_df.empty:
                personas_df = personas_df.sort_values('seguidores', ascending=False).head(5)
                fig = px.bar(
                    personas_df,
                    x='nombre',
                    y='seguidores',
                    color='seguidores',
                    color_continuous_scale='Blues'
                )
                fig.update_layout(showlegend=False)
                st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Top empresas por seguidores
            st.subheader("🏢 Top Empresas - Seguidores")
            empresas_df = pd.DataFrame(data['seguidores']['empresas'])
            if not empresas_df.empty:
                empresas_df = empresas_df.sort_values('seguidores', ascending=False)
                fig = px.bar(
                    empresas_df,
                    x='nombre',
                    y='seguidores',
                    color='seguidores',
                    color_continuous_scale='Greens'
                )
                fig.update_layout(showlegend=False)
                st.plotly_chart(fig, use_container_width=True)

        # Distribución de hashtags
        st.subheader("📊 Distribución de Hashtags")
        hashtags_df = pd.DataFrame(data['hashtags']['conteo'])
        if not hashtags_df.empty:
            fig = px.pie(
                hashtags_df,
                values='cantidad',
                names='hashtag',
                title='Popularidad de Hashtags'
            )
            st.plotly_chart(fig, use_container_width=True)

    # Vista de Seguidores
    elif view_option == "👥 Seguidores":
        st.header("👥 Análisis de Seguidores")

        tab1, tab2 = st.tabs(["Personas", "Empresas"])

        with tab1:
            personas_df = pd.DataFrame(data['seguidores']['personas'])
            if not personas_df.empty:
                st.dataframe(
                    personas_df.sort_values('seguidores', ascending=False),
                    use_container_width=True,
                    hide_index=True
                )

                # Gráfico de barras horizontal
                fig = px.bar(
                    personas_df.sort_values('seguidores'),
                    x='seguidores',
                    y='nombre',
                    orientation='h',
                    title='Seguidores por Persona',
                    color='seguidores',
                    color_continuous_scale='Viridis'
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No hay datos de personas disponibles")

        with tab2:
            empresas_df = pd.DataFrame(data['seguidores']['empresas'])
            if not empresas_df.empty:
                st.dataframe(
                    empresas_df.sort_values('seguidores', ascending=False),
                    use_container_width=True,
                    hide_index=True
                )

                # Gráfico de barras horizontal
                fig = px.bar(
                    empresas_df.sort_values('seguidores'),
                    x='seguidores',
                    y='nombre',
                    orientation='h',
                    title='Seguidores por Empresa',
                    color='seguidores',
                    color_continuous_scale='Plasma'
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No hay datos de empresas disponibles")

    # Vista de Reacciones
    elif view_option == "❤️ Reacciones":
        st.header("❤️ Reacciones por Publicación")

        reacciones_df = pd.DataFrame(data['reacciones'])
        if not reacciones_df.empty:
            # Calcular engagement
            reacciones_df['total_reacciones'] = reacciones_df['likes'] + reacciones_df['dislikes']
            reacciones_df['ratio_positivo'] = (
                reacciones_df['likes'] / reacciones_df['total_reacciones'] * 100
            ).fillna(0).round(2)

            # Mostrar tabla
            st.dataframe(
                reacciones_df.sort_values('total_reacciones', ascending=False),
                use_container_width=True,
                hide_index=True
            )

            # Gráfico de likes vs dislikes
            fig = go.Figure()
            fig.add_trace(go.Bar(
                name='Likes',
                x=reacciones_df['post_id'],
                y=reacciones_df['likes'],
                marker_color='green'
            ))
            fig.add_trace(go.Bar(
                name='Dislikes',
                x=reacciones_df['post_id'],
                y=reacciones_df['dislikes'],
                marker_color='red'
            ))
            fig.update_layout(
                title='Likes vs Dislikes por Publicación',
                barmode='group',
                xaxis_title='Post ID',
                yaxis_title='Cantidad'
            )
            st.plotly_chart(fig, use_container_width=True)

            # Ratio positivo
            fig2 = px.line(
                reacciones_df.sort_values('post_id'),
                x='post_id',
                y='ratio_positivo',
                markers=True,
                title='Ratio de Aceptación por Publicación (%)'
            )
            fig2.update_layout(yaxis_range=[0, 100])
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("No hay datos de reacciones disponibles")

    # Vista de Top Publicaciones
    elif view_option == "🏆 Top Publicaciones":
        st.header("🏆 Top Publicaciones")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("⭐ Top 5 - MÁS Likes")
            mas_likes = data['top_posts']['mas_likes']
            if mas_likes:
                for i, post in enumerate(mas_likes, 1):
                    st.markdown(f"""
                    <div class="metric-card">
                        <strong>#{i}</strong> - {post['likes']} ❤️<br>
                        "{post['texto']}"
                    </div>
                    """, unsafe_allow_html=True)
                    st.markdown("<br>", unsafe_allow_html=True)
            else:
                st.info("No hay datos disponibles")

        with col2:
            st.subheader("📉 Top 5 - MENOS Likes")
            menos_likes = data['top_posts']['menos_likes']
            if menos_likes:
                for i, post in enumerate(menos_likes, 1):
                    st.markdown(f"""
                    <div class="metric-card">
                        <strong>#{i}</strong> - {post['likes']} ❤️<br>
                        "{post['texto']}"
                    </div>
                    """, unsafe_allow_html=True)
                    st.markdown("<br>", unsafe_allow_html=True)
            else:
                st.info("No hay datos disponibles")

    # Vista de Hashtags
    elif view_option == "#️⃣ Hashtags":
        st.header("#️⃣ Análisis de Hashtags")

        # Hashtag más usado
        if data['hashtags']['mas_usado']:
            mas_usado = data['hashtags']['mas_usado']
            st.markdown(f"""
            ### 🏆 Hashtag Más Usado
            <div style="text-align: center; font-size: 3rem; color: #1E88E5; padding: 2rem;">
                {mas_usado['hashtag']}
            </div>
            <div style="text-align: center; font-size: 1.5rem;">
                {mas_usado['cantidad']} publicaciones
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")

        # Tabla de todos los hashtags
        hashtags_df = pd.DataFrame(data['hashtags']['conteo'])
        if not hashtags_df.empty:
            col1, col2 = st.columns(2)

            with col1:
                st.dataframe(
                    hashtags_df.sort_values('cantidad', ascending=False),
                    use_container_width=True,
                    hide_index=True
                )

            with col2:
                # Gráfico de dona
                fig = px.pie(
                    hashtags_df,
                    values='cantidad',
                    names='hashtag',
                    hole=0.4,
                    title='Distribución de Hashtags'
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No hay datos de hashtags disponibles")

    # Vista de Bloqueados
    elif view_option == "🚫 Usuarios Bloqueados":
        st.header("🚫 Seguidores Bloqueados")

        bloqueados = data['bloqueados']
        if bloqueados:
            bloqueados_df = pd.DataFrame(bloqueados)

            # Contar bloqueos por usuario
            bloqueos_count = bloqueados_df.groupby('usuario').size().reset_index(name='total_bloqueados')

            col1, col2 = st.columns(2)

            with col1:
                st.subheader("📊 Total de Bloqueos por Usuario")
                fig = px.bar(
                    bloqueos_count.sort_values('total_bloqueados', ascending=False),
                    x='usuario',
                    y='total_bloqueados',
                    color='total_bloqueados',
                    color_continuous_scale='Reds'
                )
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                st.subheader("📋 Lista Detallada")
                st.dataframe(bloqueados_df, use_container_width=True, hide_index=True)
        else:
            st.info("No hay usuarios bloqueados")

    # Vista de Recomendaciones
    elif view_option == "💼 Recomendaciones Empresas":
        st.header("💼 Recomendaciones entre Empresas")

        recomendaciones = data['recomendaciones']
        if recomendaciones:
            rec_df = pd.DataFrame(recomendaciones)

            st.dataframe(rec_df, use_container_width=True, hide_index=True)

            # Gráfico de red (simplificado)
            st.subheader("🔗 Red de Recomendaciones")
            st.info("Red de empresas que se recomiendan entre sí")

            # Mostrar métricas
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total Recomendaciones", len(recomendaciones))
            with col2:
                empresas_unicas = set(rec_df['recomienda'].tolist() + rec_df['recomendada'].tolist())
                st.metric("Empresas Involucradas", len(empresas_unicas))
        else:
            st.info("No hay recomendaciones entre empresas")

    # Vista de Output Completo
    elif view_option == "📄 Output Completo":
        st.header("📄 Output Completo del Programa CUDA")

        st.code(data['output_raw'], language='text')

        # Botón de descarga
        st.download_button(
            label="💾 Descargar Output",
            data=data['output_raw'],
            file_name="cuda_social_network_output.txt",
            mime="text/plain"
        )

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666;">
    <p>🚀 Red Social CUDA - Ejecutado en Colab, Visualizado Localmente | Powered by Streamlit</p>
</div>
""", unsafe_allow_html=True)
