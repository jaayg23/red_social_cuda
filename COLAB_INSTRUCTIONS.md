# 🚀 Cómo usar tu Red Social CUDA en Google Colab

## 📋 Pasos rápidos para empezar

### 1. Subir el notebook a Google Colab

1. Ve a [Google Colab](https://colab.research.google.com/)
2. Click en `File → Upload notebook`
3. Sube el archivo `red_social_cuda_colab.ipynb`

### 2. Habilitar GPU (MUY IMPORTANTE)

⚠️ **CRÍTICO**: Sin este paso no funcionará CUDA

1. En Colab, ve a `Runtime → Change runtime type`
2. En "Hardware accelerator" selecciona **GPU**
3. Click en **Save**

### 3. Subir tus archivos

Tienes 3 opciones:

#### Opción A: Upload directo (más rápido para probar)
1. En el panel izquierdo de Colab, click en el ícono de carpeta 📁
2. Click en el botón de upload ⬆️
3. Sube estos archivos:
   - `social_network.cu`
   - `cuda_wrapper.py`

#### Opción B: Google Drive (recomendado para uso frecuente)
1. Sube tus archivos a Google Drive
2. En el notebook, ejecuta la celda de "Montar Drive"
3. Autoriza el acceso
4. Copia los archivos desde Drive a Colab

#### Opción C: GitHub (si tienes repo)
```bash
!git clone https://github.com/tu-usuario/tu-repo.git
cd tu-repo
```

### 4. Ejecutar el notebook

1. Ejecuta las celdas en orden (de arriba hacia abajo)
2. Usa `Shift + Enter` para ejecutar cada celda
3. O usa `Runtime → Run all` para ejecutar todo

## 📊 Visualizaciones disponibles

El notebook incluye:

✅ Ejecución directa del programa CUDA
✅ Visualizaciones con Pandas DataFrames
✅ Gráficos interactivos con Plotly:
- Top personas por seguidores
- Empresas por seguidores
- Distribución de hashtags
- Likes vs Dislikes por publicación

## 🎯 Estructura del notebook

```
1. ✅ Verificar GPU
2. 📂 Subir archivos
3. 🔧 Instalar dependencias
4. 🛠️ Compilar CUDA
5. ▶️ Ejecutar programa
6. 🐍 Usar wrapper Python
7. 📊 Ver datos en tablas
8. 📈 Crear gráficos
9. 🌐 [Opcional] Streamlit
10. 💾 Descargar resultados
```

## ⚡ Comandos útiles en Colab

```bash
# Ver GPUs disponibles
!nvidia-smi

# Ver archivos en directorio actual
!ls -la

# Ver contenido de un archivo
!cat social_network.cu

# Compilar CUDA
!nvcc -o social_network social_network.cu -std=c++11

# Ejecutar programa
!./social_network

# Ver uso de memoria
!free -h
```

## 🔧 Solución de problemas

### Error: "nvcc not found"
- **Causa**: No seleccionaste GPU en Runtime
- **Solución**: `Runtime → Change runtime type → GPU → Save`

### Error: "No module named 'cuda_wrapper'"
- **Causa**: No subiste el archivo `cuda_wrapper.py`
- **Solución**: Sube el archivo usando el panel de archivos

### Error: "social_network.cu not found"
- **Causa**: No subiste el archivo CUDA
- **Solución**: Sube `social_network.cu` al directorio raíz

### Sesión desconectada
- **Causa**: Google Colab tiene límite de tiempo de inactividad
- **Solución**: Vuelve a ejecutar desde el inicio

### Out of memory
- **Causa**: GPU sin memoria suficiente
- **Solución**: `Runtime → Factory reset runtime`

## 💡 Tips y trucos

### Mantener la sesión activa
```python
# Ejecuta esto en una celda para evitar desconexión
import time
from IPython.display import clear_output

while True:
    time.sleep(300)  # Cada 5 minutos
    clear_output()
    print("Sesión activa ✓")
```

### Guardar en Drive automáticamente
```python
from google.colab import drive
drive.mount('/content/drive')

# Guardar resultados
!cp resultados.json /content/drive/MyDrive/
```

### Verificar tipo de GPU asignada
```python
!nvidia-smi --query-gpu=gpu_name --format=csv,noheader
```

Típicamente obtendrás:
- **T4** (16GB) - Más común en versión gratuita
- **K80** (12GB) - Ocasionalmente
- **P100** (16GB) - Menos común pero más rápida

## 📦 Archivos necesarios

Asegúrate de tener:

```
✅ red_social_cuda_colab.ipynb  (El notebook)
✅ social_network.cu            (Tu código CUDA)
✅ cuda_wrapper.py              (Wrapper Python)
✅ app.py                       (Solo si quieres usar Streamlit)
```

## 🌐 Ejecutar Streamlit en Colab (Avanzado)

Si quieres usar la interfaz Streamlit completa:

1. Sube también `app.py`
2. Ejecuta las celdas del Paso 9 en el notebook
3. Usa ngrok o localtunnel para exponer el servidor
4. Accede mediante la URL generada

**Alternativa más simple**: Usa solo las visualizaciones de Plotly en el notebook (ya incluidas)

## 📚 Recursos adicionales

- [Documentación CUDA](https://docs.nvidia.com/cuda/)
- [Google Colab FAQ](https://research.google.com/colaboratory/faq.html)
- [Plotly Python](https://plotly.com/python/)

## ⏱️ Límites de Google Colab

### Versión gratuita:
- ⏰ ~12 horas de sesión continua
- 💾 ~12 GB RAM
- 🎮 GPU T4 (16GB VRAM)
- 💽 ~100 GB almacenamiento temporal

### Colab Pro ($10/mes):
- ⏰ ~24 horas de sesión
- 💾 ~26 GB RAM
- 🎮 GPUs más potentes (V100, A100)
- 💽 ~200 GB almacenamiento

## 🎓 Próximos pasos

1. ✅ Ejecuta el notebook básico
2. 📊 Experimenta con las visualizaciones
3. 🔧 Modifica `social_network.cu` según necesites
4. 📈 Crea tus propios gráficos personalizados
5. 🚀 Si necesitas más poder, considera Colab Pro

---

¿Problemas? Revisa la sección de solución de problemas o las celdas de documentación en el notebook.
