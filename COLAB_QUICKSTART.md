# Guía Rápida para Google Colab

## Pasos para ejecutar el proyecto en Google Colab

### 1. Abrir Google Colab

Ve a [https://colab.research.google.com/](https://colab.research.google.com/)

### 2. Subir el Notebook

- Opción A: Arrastra `Red_Social_CUDA.ipynb` a la ventana de Colab
- Opción B: `File > Upload notebook` y selecciona `Red_Social_CUDA.ipynb`

### 3. Habilitar GPU

**MUY IMPORTANTE**: CUDA solo funciona con GPU NVIDIA

1. Click en `Runtime` (o `Entorno de ejecución`)
2. Click en `Change runtime type` (o `Cambiar tipo de entorno de ejecución`)
3. En `Hardware accelerator`, selecciona **GPU**
4. Click `Save`

### 4. Subir el archivo de código

En el panel lateral izquierdo (icono de carpeta):

1. Click en el icono de carpeta 📁
2. Click en el icono de upload (flecha hacia arriba)
3. Selecciona el archivo `social_network.cu` desde tu computadora

**IMPORTANTE**: Los archivos subidos se borran cuando termina la sesión de Colab. Deberás subirlos cada vez.

### 5. Ejecutar las celdas

Ejecuta las celdas en orden (Ctrl+Enter o Shift+Enter):

#### Celda 1: Verificar GPU
```python
!nvidia-smi
```

Deberías ver información de la GPU (típicamente Tesla T4).

#### Celda 2: Compilar
```bash
!nvcc -o social_network social_network.cu -std=c++11
```

Si ves errores, verifica que el archivo `social_network.cu` esté en el directorio.

#### Celda 3: Ejecutar
```bash
!./social_network
```

Verás todas las salidas de las queries.

## Alternativa: Código en una sola celda

Si prefieres no subir archivos, puedes copiar TODO el contenido de `social_network.cu` en una celda con el comando mágico `%%writefile`:

```python
%%writefile social_network.cu

# Aquí pegas todo el contenido de social_network.cu
```

Luego compila y ejecuta normalmente.

## Troubleshooting

### Error: "nvcc: command not found"

**Causa**: No tienes GPU habilitada o Colab no detecta CUDA.

**Solución**:
1. Ve a `Runtime > Change runtime type`
2. Asegúrate de seleccionar **GPU**
3. Reinicia el runtime: `Runtime > Restart runtime`

### Error: "No such file or directory: social_network.cu"

**Causa**: No subiste el archivo o está en otro directorio.

**Solución**:
1. Verifica en el panel de archivos que `social_network.cu` exista
2. Si no está, súbelo arrastrándolo al panel
3. Alternativamente, usa la opción `%%writefile` mencionada arriba

### Error de compilación: "error: identifier 'X' is undefined"

**Causa**: Puede haber un problema con el código o versión de CUDA.

**Solución**:
1. Verifica que el archivo esté completo
2. Prueba agregar flag: `!nvcc -o social_network social_network.cu -std=c++14`

### GPU limitada o no disponible

**Causa**: Colab tiene límites de uso gratuito de GPU.

**Solución**:
- Espera unas horas y vuelve a intentar
- Considera Colab Pro si necesitas más tiempo de GPU
- El código también compila sin GPU pero no ejecutará kernels

## Recursos

- [Documentación de Colab](https://colab.research.google.com/notebooks/intro.ipynb)
- [FAQ de Colab GPU](https://research.google.com/colaboratory/faq.html#gpu-availability)
- [CUDA Toolkit Docs](https://docs.nvidia.com/cuda/)

## Consejos

1. **Guarda tu trabajo**: Colab borra archivos al cerrar sesión. Descarga resultados importantes.
2. **Sesiones limitadas**: Colab tiene límite de tiempo (12h máximo). Guarda frecuentemente.
3. **Experimenta**: Modifica los datos hardcodeados en `initialize_sample_data()` para probar diferentes escenarios.
4. **Mide tiempos**: Usa `%%time` en celdas para medir rendimiento.

## Siguiente Paso: Modificar Datos

Para cambiar los datos de prueba, edita la función `initialize_sample_data()` en `social_network.cu`:

- Cambia nombres de personas/empresas
- Agrega más relaciones de seguimiento
- Crea nuevas publicaciones con diferentes hashtags
- Modifica interacciones (likes/dislikes)

Luego recompila y ejecuta.
