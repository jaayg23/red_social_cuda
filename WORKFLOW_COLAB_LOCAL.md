# 🔄 Workflow: Ejecutar en Colab, Visualizar en Local

Este es el método **RECOMENDADO** si no tienes GPU NVIDIA en tu PC.

## 📋 Resumen del flujo de trabajo

```
┌─────────────────┐
│  Google Colab   │  ← Ejecuta CUDA (GPU gratis)
│   (En la nube)  │
└────────┬────────┘
         │
         ▼
    resultados.json  ← Descarga este archivo
         │
         ▼
┌─────────────────┐
│   Tu PC Local   │  ← Visualiza con Streamlit
│  (Sin necesidad │
│   de GPU/CUDA)  │
└─────────────────┘
```

## 🚀 Paso 1: Ejecutar en Google Colab

### 1. Abre el notebook en Colab
- Sube `red_social_cuda_colab.ipynb` a Google Colab
- Habilita GPU: `Runtime → Change runtime type → GPU`

### 2. Ejecuta el código CUDA
Sigue el notebook hasta obtener la variable `data` con los resultados.

### 3. Guarda y descarga resultados
Al final del notebook, ejecuta:

```python
import json
from google.colab import files

# Guardar datos en JSON
with open('resultados.json', 'w', encoding='utf-8') as f:
    json.dump(data, f, indent=2, ensure_ascii=False)

print("✓ Archivo guardado")

# Descargar automáticamente
files.download('resultados.json')
```

## 💻 Paso 2: Visualizar en tu PC Local

### 1. Asegúrate de tener las dependencias instaladas

```bash
pip install streamlit pandas plotly
```

### 2. Coloca el archivo descargado
Pon `resultados.json` en la misma carpeta que `app_sin_cuda.py`

```
proyectos/red_social_cuda/
├── app_sin_cuda.py       ← Nueva app sin CUDA
├── resultados.json       ← Descargado de Colab
├── requirements.txt
└── ...
```

### 3. Ejecuta Streamlit localmente

```bash
streamlit run app_sin_cuda.py
```

### 4. Abre tu navegador
Se abrirá automáticamente en `http://localhost:8501`

## 🎯 Ventajas de este método

✅ **No necesitas GPU** en tu PC local
✅ **No necesitas instalar CUDA Toolkit** (¡nada de nvcc!)
✅ **Rápido**: Solo ejecutas visualizaciones, no compilación
✅ **Reutilizable**: Puedes guardar múltiples `resultados.json`
✅ **Sin túneles**: Todo corre localmente sin problemas de passwords

## 📊 Características de app_sin_cuda.py

La nueva aplicación incluye:

- 📤 **Upload de archivos**: Arrastra y suelta el JSON
- 📁 **Carga desde ruta**: O especifica la ruta local
- 📈 **Todas las visualizaciones**: Igual que la app original
- 💾 **Sin dependencias CUDA**: Solo Python puro

## 🔄 Flujo completo ejemplo

```bash
# EN GOOGLE COLAB:
1. Subir social_network.cu y cuda_wrapper.py
2. Compilar: !nvcc -o social_network social_network.cu -std=c++11
3. Ejecutar: network = CUDASocialNetwork(...)
4. Guardar: json.dump(data, f)
5. Descargar: files.download('resultados.json')

# EN TU PC:
6. Mover resultados.json a la carpeta del proyecto
7. Ejecutar: streamlit run app_sin_cuda.py
8. Abrir navegador en localhost:8501
9. ¡Listo! 🎉
```

## 🆚 Comparación de métodos

| Método | GPU Necesaria | Instalación CUDA | Complejidad | Velocidad |
|--------|---------------|------------------|-------------|-----------|
| **Colab → Local** ⭐ | No (en tu PC) | No | Baja | Rápida |
| Todo en Colab + túnel | No | No | Media | Media |
| Todo local | Sí | Sí | Alta | Muy rápida |

## 📝 Notas adicionales

### Generar múltiples resultados
Puedes guardar diferentes ejecuciones:

```python
# En Colab, guarda con nombres diferentes
json.dump(data, open('resultados_v1.json', 'w'))
json.dump(data, open('resultados_v2.json', 'w'))
```

Luego en local, carga cualquiera en la app.

### Automatizar el proceso
Puedes crear un script para ejecutar automáticamente:

**run_local.bat** (Windows):
```batch
@echo off
echo Iniciando visualización local...
streamlit run app_sin_cuda.py
```

**run_local.sh** (Linux/Mac):
```bash
#!/bin/bash
echo "Iniciando visualización local..."
streamlit run app_sin_cuda.py
```

### Compartir resultados
El archivo `resultados.json` es portable. Puedes:
- Enviarlo por email
- Subirlo a Drive
- Compartirlo con otros para que visualicen

## 🐛 Solución de problemas

### "File not found: resultados.json"
- Asegúrate de que el archivo está en la misma carpeta
- O especifica la ruta completa en la app

### "Module not found: streamlit"
```bash
pip install streamlit pandas plotly
```

### La app no se abre
- Verifica que el puerto 8501 no esté en uso
- O especifica otro puerto: `streamlit run app_sin_cuda.py --server.port 8502`

## 🎓 Próximos pasos

1. ✅ Ejecuta tu código en Colab
2. ✅ Descarga el JSON
3. ✅ Usa `app_sin_cuda.py` localmente
4. 🎨 Personaliza las visualizaciones según tus necesidades
5. 📊 Experimenta con diferentes conjuntos de datos

---

**¿Preguntas?** Este método es ideal para desarrollo iterativo: modificas el código CUDA en Colab, descargas resultados, y visualizas localmente sin fricciones.
