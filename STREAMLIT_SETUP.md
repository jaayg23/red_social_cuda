# 🚀 Interfaz Visual para Red Social CUDA

## Descripción

Esta interfaz visual tipo **Streamlit** permite visualizar y analizar de forma interactiva los resultados de tu red social implementada con CUDA C++.

## ✨ Características

- 📊 **Dashboard interactivo** con métricas en tiempo real
- 📈 **Gráficos dinámicos** con Plotly (barras, líneas, pie charts)
- 👥 **Análisis de seguidores** por personas y empresas
- ❤️ **Visualización de reacciones** (likes/dislikes)
- #️⃣ **Análisis de hashtags** más populares
- 🚫 **Monitor de bloqueos** entre usuarios
- 💼 **Red de recomendaciones** empresariales
- 🎨 **Interfaz moderna y responsiva**

## 📋 Requisitos Previos

### 1. CUDA Toolkit
Necesitas tener instalado **CUDA Toolkit** y el compilador `nvcc`:

- **Windows**: [CUDA Toolkit Download](https://developer.nvidia.com/cuda-downloads)
- **Linux**: `sudo apt install nvidia-cuda-toolkit`

Verifica la instalación:
```bash
nvcc --version
```

### 2. Python 3.8+
Asegúrate de tener Python instalado:
```bash
python --version
```

## 🛠️ Instalación

### Paso 1: Instalar dependencias de Python

```bash
cd OneDrive/Escritorio/proyectos/red_social_cuda
pip install -r requirements.txt
```

### Paso 2: Verificar archivos

Asegúrate de tener estos archivos en la carpeta:
```
red_social_cuda/
├── social_network.cu       # Código CUDA (ya existe)
├── app.py                  # Interfaz Streamlit (nuevo)
├── cuda_wrapper.py         # Wrapper Python (nuevo)
├── requirements.txt        # Dependencias (nuevo)
└── README.md              # Documentación original
```

## 🚀 Uso

### Iniciar la aplicación

```bash
streamlit run app.py
```

La aplicación se abrirá automáticamente en tu navegador en `http://localhost:8501`

### Primeros pasos

1. **Compilar el código CUDA** (primera vez):
   - En la barra lateral, haz clic en "🔨 Compilar Código CUDA"
   - Espera el mensaje de confirmación

2. **Ejecutar el análisis**:
   - Haz clic en "▶️ Ejecutar Análisis"
   - El programa ejecutará tu código CUDA y procesará los datos

3. **Explorar las visualizaciones**:
   - Selecciona diferentes vistas en la barra lateral:
     - 📈 Dashboard General
     - 👥 Seguidores
     - ❤️ Reacciones
     - 🏆 Top Publicaciones
     - #️⃣ Hashtags
     - 🚫 Usuarios Bloqueados
     - 💼 Recomendaciones Empresas
     - 📄 Output Completo

## 📊 Vistas Disponibles

### Dashboard General
- Métricas principales (personas, empresas, publicaciones)
- Top 5 personas por seguidores
- Top empresas por seguidores
- Distribución de hashtags

### Seguidores
- Tabla detallada de seguidores por persona/empresa
- Gráficos de barras horizontales
- Ordenamiento por cantidad

### Reacciones
- Likes y dislikes por publicación
- Ratio de aceptación
- Gráficos comparativos
- Análisis de engagement

### Top Publicaciones
- Top 5 publicaciones con MÁS likes
- Top 5 publicaciones con MENOS likes
- Texto completo de cada publicación

### Hashtags
- Hashtag más usado (destacado)
- Distribución de todos los hashtags
- Gráfico de dona interactivo

### Usuarios Bloqueados
- Lista de bloqueos por usuario
- Total de bloqueos
- Gráfico de barras

### Recomendaciones Empresas
- Red de recomendaciones entre empresas
- Tabla detallada de relaciones

### Output Completo
- Salida raw del programa CUDA
- Opción de descarga

## 🎨 Capturas de Pantalla

*(La interfaz incluye)*
- Header principal con logo
- Sidebar con controles
- Métricas con cards visuales
- Gráficos interactivos (zoom, hover, descarga)
- Tablas ordenables

## ⚙️ Configuración Avanzada

### Modificar el puerto de Streamlit

```bash
streamlit run app.py --server.port 8080
```

### Ejecutar en modo de desarrollo

```bash
streamlit run app.py --server.runOnSave true
```

### Desactivar el modo watch

```bash
streamlit run app.py --server.fileWatcherType none
```

## 🔧 Troubleshooting

### Error: "nvcc no encontrado"
**Solución**: Agrega CUDA al PATH:
```bash
# Windows
set PATH=%PATH%;C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.x\bin

# Linux
export PATH=/usr/local/cuda/bin:$PATH
```

### Error: "No module named 'streamlit'"
**Solución**: Instala las dependencias:
```bash
pip install -r requirements.txt
```

### Error al compilar CUDA
**Solución**: Verifica que:
- Tienes una GPU NVIDIA compatible
- CUDA Toolkit está instalado correctamente
- El archivo `social_network.cu` existe en el directorio

### La app no muestra datos
**Solución**:
1. Compila el código primero (botón en sidebar)
2. Ejecuta el análisis (botón verde)
3. Verifica que no haya errores en el output

## 📦 Estructura del Código

### `app.py`
Interfaz principal de Streamlit con:
- Configuración de página
- Sidebar con controles
- 7 vistas diferentes
- Gráficos con Plotly
- Manejo de estado con `st.session_state`

### `cuda_wrapper.py`
Wrapper de Python que:
- Compila código CUDA con `nvcc`
- Ejecuta el binario compilado
- Parsea la salida con regex
- Estructura los datos en diccionarios

## 🚀 Próximas Mejoras

Posibles extensiones para la interfaz:

- [ ] Cargar datos desde CSV/JSON
- [ ] Exportar resultados a Excel
- [ ] Gráficos de red interactivos (NetworkX)
- [ ] Comparación temporal de métricas
- [ ] Filtros avanzados por fecha/usuario
- [ ] Dark mode
- [ ] Multiidioma (ES/EN)
- [ ] Carga de datos en tiempo real

## 🤝 Contribuir

Si deseas mejorar la interfaz:
1. Modifica `app.py` para agregar nuevas vistas
2. Actualiza `cuda_wrapper.py` para parsear más datos
3. Agrega nuevas librerías en `requirements.txt`

## 📝 Notas

- La primera compilación puede tardar unos segundos
- Los datos se cachean automáticamente para mejor rendimiento
- Puedes recargar la página para reiniciar el estado
- Los gráficos son interactivos (zoom, pan, hover)

## 📄 Licencia

Proyecto educacional - Libre uso

---

**Creado con ❤️ usando Streamlit + CUDA**
