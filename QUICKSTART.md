# ⚡ Inicio Rápido - Red Social CUDA con Streamlit

## 🚀 Instalación en 3 pasos

### 1️⃣ Instalar dependencias
```bash
pip install -r requirements.txt
```

### 2️⃣ Iniciar la aplicación
**Opción A - Usando el script:**
```bash
# Windows
run.bat

# Linux/Mac
chmod +x run.sh
./run.sh
```

**Opción B - Manual:**
```bash
streamlit run app.py
```

### 3️⃣ Usar la aplicación
1. La app se abrirá en tu navegador automáticamente (`http://localhost:8501`)
2. En la **barra lateral**:
   - Clic en **"🔨 Compilar Código CUDA"** (solo la primera vez)
   - Clic en **"▶️ Ejecutar Análisis"**
3. ¡Explora las visualizaciones! 📊

---

## 📊 Vistas Disponibles

| Vista | Descripción |
|-------|-------------|
| 📈 Dashboard General | Métricas principales y gráficos resumen |
| 👥 Seguidores | Análisis de seguidores por persona/empresa |
| ❤️ Reacciones | Likes y dislikes por publicación |
| 🏆 Top Publicaciones | Publicaciones más y menos populares |
| #️⃣ Hashtags | Análisis de hashtags más usados |
| 🚫 Usuarios Bloqueados | Monitor de bloqueos entre usuarios |
| 💼 Recomendaciones | Red de recomendaciones empresariales |
| 📄 Output Completo | Salida raw del programa CUDA |

---

## ⚠️ Requisitos

- ✅ **Python 3.8+**
- ✅ **CUDA Toolkit** (con `nvcc`)
- ✅ **GPU NVIDIA** (para compilar y ejecutar)

### Verificar instalación de CUDA:
```bash
nvcc --version
```

Si no tienes GPU NVIDIA, puedes usar **Google Colab** con el notebook incluido: `Red_Social_CUDA.ipynb`

---

## 🎯 Ejemplo de Uso

```bash
# 1. Navegar al proyecto
cd OneDrive/Escritorio/proyectos/red_social_cuda

# 2. Instalar dependencias
pip install streamlit pandas plotly

# 3. Ejecutar
streamlit run app.py
```

**¡Listo!** La interfaz se abrirá en tu navegador 🎉

---

## 🆘 Problemas Comunes

### "nvcc no encontrado"
**Solución**: Instala CUDA Toolkit desde [nvidia.com/cuda-downloads](https://developer.nvidia.com/cuda-downloads)

### "ModuleNotFoundError: No module named 'streamlit'"
**Solución**:
```bash
pip install -r requirements.txt
```

### "No se puede compilar el código"
**Solución**: Verifica que tienes una GPU NVIDIA y CUDA instalado correctamente

---

## 📁 Estructura de Archivos

```
red_social_cuda/
├── 🎨 app.py                  # Interfaz Streamlit (NUEVO)
├── 🔧 cuda_wrapper.py         # Wrapper Python (NUEVO)
├── 📋 requirements.txt        # Dependencias (NUEVO)
├── 🚀 run.bat / run.sh        # Scripts de inicio (NUEVO)
├── 💻 social_network.cu       # Código CUDA original
├── 📓 Red_Social_CUDA.ipynb  # Notebook para Colab
├── 📖 README.md              # Documentación original
└── 📘 STREAMLIT_SETUP.md     # Guía completa de Streamlit
```

---

## 💡 Tips

- **Recarga automática**: Streamlit recarga la app automáticamente al guardar cambios
- **Modo oscuro**: Clic en el menú (⋮) → Settings → Theme
- **Descarga de gráficos**: Hover sobre cualquier gráfico → ícono de cámara
- **Fullscreen**: Clic en el ícono de expansión en tablas y gráficos

---

**¿Necesitas más ayuda?** Consulta [STREAMLIT_SETUP.md](STREAMLIT_SETUP.md) para la guía completa.
