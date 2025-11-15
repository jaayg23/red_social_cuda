# Red Social con CUDA C++

Una implementación de red social simple usando CUDA C++ para operaciones paralelas en GPU.

## Descripción

Este proyecto implementa una red social con personas, empresas y publicaciones, utilizando CUDA para acelerar operaciones como conteo de seguidores, análisis de interacciones y cálculo de redes de influencia.

## Características Implementadas

### Entidades

- **Personas**: Usuarios individuales que pueden publicar, seguir y reaccionar
- **Empresas**: Organizaciones que pueden publicar y ser seguidas
- **Publicaciones**: Posts con texto y hashtags

### Relaciones

- Persona → Persona: Seguimiento, amistad, bloqueo
- Persona → Empresa: Seguimiento, cliente, empleado
- Empresa → Empresa: Seguimiento, recomendación
- Usuario → Publicación: Like, Dislike
- Publicación → Publicación: Republicación, recomendación

### Queries Implementadas

1. **Cantidad de seguidores** por persona y empresa
2. **Reacciones por publicación** (likes y dislikes)
3. **Top 5 publicaciones** con más y menos likes
4. **Seguidores bloqueados** por cada usuario
5. **Recomendaciones de empresas** (quién recomienda a quién)
6. **Empresas con más recomendaciones** (ordenadas descendentemente)
7. **Análisis de hashtags** (más usado, publicaciones por hashtag)
8. **Usuarios por hashtag** (personas y empresas)
9. **Mejores clientes** (clientes que más gustan de publicaciones de empresa)
10. **Empresas con más/menos likes**
11. **Visibilidad de publicaciones** (quién puede ver cada post)
12. **Red de influencia** (seguidores hasta grado N)

## Requisitos

- **GPU NVIDIA** con soporte CUDA (localmente)
- **Google Colab** (alternativa gratuita con GPU)
- **CUDA Toolkit** (si compilas localmente)
- Compilador compatible con C++11

## Estructura del Proyecto

```
red_social_cuda/
├── social_network.cu          # Código principal CUDA
├── Red_Social_CUDA.ipynb     # Notebook para Google Colab
└── README.md                 # Esta documentación
```

## Cómo Usar

### Opción 1: Google Colab (Recomendado)

1. Abre `Red_Social_CUDA.ipynb` en Google Colab
2. Ve a `Runtime > Change runtime type` y selecciona GPU
3. Sube el archivo `social_network.cu` al entorno de Colab
4. Ejecuta las celdas en orden:
   - Verificar GPU disponible
   - Compilar código
   - Ejecutar programa

### Opción 2: Compilación Local (requiere GPU NVIDIA)

```bash
# Compilar
nvcc -o social_network social_network.cu -std=c++11

# Ejecutar
./social_network
```

## Implementación Técnica

### Estructuras de Datos

- **SoA (Structure of Arrays)**: Optimizado para acceso coalescente en GPU
- **Matrices de adyacencia**: Para relaciones entre usuarios (MAX_USERS × MAX_USERS)
- **Arrays paralelos**: Para publicaciones e interacciones

### Kernels CUDA Implementados

1. **count_person_followers_kernel**: Cuenta seguidores de personas usando reducción paralela
2. **count_company_followers_kernel**: Cuenta seguidores de empresas (personas + empresas)
3. **count_post_likes_kernel**: Cuenta likes/dislikes con reducción paralela
4. **find_posts_by_hashtag_kernel**: Búsqueda paralela por hashtag
5. **check_visibility_kernel**: Determina visibilidad de publicaciones (seguidores + seguidores de seguidores)

### Optimizaciones

- **Reducción paralela** en shared memory para conteos
- **Atomic operations** para sincronización de resultados
- **Coalescencia de memoria** mediante SoA
- **Ocupación optimizada** con THREADS_PER_BLOCK = 256

### Reglas de Visibilidad

- **Publicaciones de empresa**: Visibles para todos
- **Publicaciones de persona**: Visibles para:
  - El autor
  - Seguidores no bloqueados
  - Seguidores de seguidores (grado 2)

## Datos de Prueba

El programa incluye datos hardcodeados:

- **6 personas**: Alice, Bob, Charlie, Diana, Eve, Frank
- **3 empresas**: TechCorp, SocialHub, DataInc
- **10 publicaciones** con varios hashtags (#tech, #coding, #life, #social, #data)
- Relaciones de seguimiento, bloqueo, clientes, empleados
- Interacciones de like/dislike

## Ejemplo de Salida

```
========================================
  RED SOCIAL CON CUDA
========================================

Datos cargados:
  - 6 personas
  - 3 empresas
  - 10 publicaciones

========== CANTIDAD DE SEGUIDORES ==========

--- Personas ---
Alice: 2 seguidores
Bob: 2 seguidores
Charlie: 1 seguidores
...

========== TOP 5 PUBLICACIONES ==========

--- Top 5 con MAS likes ---
1. "Nuevos productos disponibles #tech" - 3 likes
2. "Me encanta programar #coding" - 2 likes
...
```

## Limitaciones Actuales

- `MAX_USERS = 1000`: Máximo de usuarios (personas + empresas)
- `MAX_POSTS = 2000`: Máximo de publicaciones
- `MAX_TEXT_LEN = 256`: Longitud máxima de texto por publicación
- `MAX_HASHTAG_LEN = 32`: Longitud máxima de hashtag
- Datos hardcodeados (no hay persistencia)
- Red de influencia calculada en CPU (BFS iterativo)

## Posibles Extensiones

1. **Carga de datos desde archivo** (CSV, JSON)
2. **BFS paralelo en GPU** para red de influencia
3. **Ranking de publicaciones** con algoritmos más complejos
4. **Detección de comunidades** en el grafo social
5. **Análisis de sentimiento** en publicaciones
6. **Recomendación de contenido** basada en intereses
7. **Sistema de mensajería** entre usuarios
8. **Timeline personalizado** por usuario

## Referencias

- [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [CUDA Best Practices](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)
- [Graph Algorithms on GPU](https://developer.nvidia.com/gpugems/gpugems3/part-vi-gpu-computing/chapter-39-parallel-prefix-sum-scan-cuda)

## Autor

Proyecto creado con Claude Code

## Licencia

Este es un proyecto educacional de demostración.
