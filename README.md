# VISION_PARALELA_1


# **Alcances y restricciones del proyecto COIL-100 Lab**  
# ### Alcances (qué cubre el proyecto)
# 
# 1. **Comparación de dos enfoques de clasificación de imágenes**
#    - **Visión clásica**: Random Forest (100 árboles) sobre features de HOG (9 orientaciones, celdas 16×16) + LBP uniforme (10) + histogramas HSV (96) → 1.870 features totales.
#    - **Deep Learning**: MobileNetV2 preentrenado en ImageNet (transfer learning), última capa reemplazada por 5 salidas. Entrenado 8 épocas con Adam (lr=1e-4). ~2.23 M de parámetros (~8.5 MB).
# 
# 2. **Dataset y configuración experimental**
#    - Subconjunto de **5 objetos** del dataset COIL-100:
#      - obj043 → Taza con sandías
#      - obj063 → Pimiento rojo
#      - obj068 → Grapadora
#      - obj074 → Patito de goma
#      - obj075 → Fresa
#    - Resolución de entrada: principalmente 128×128 (también evaluado 64×64 en CNN).
#    - **Escenarios de partición**:
#      - Escenario A (fácil): bloques intercalados.
#      - Escenario B (difícil / principal): train 0–175° · test 180–355° (ángulos nunca vistos en entrenamiento).
# 
# 3. **Métricas y evaluación completas**
#    - Accuracy, F1 macro, latencia media, p95, throughput (img/s), RAM pico, VRAM y tiempo de entrenamiento.
#    - Matrices de confusión interactivas.
#    - Análisis de errores específicos (especialmente fresa vs pimiento).
# 
# 4. **Análisis de rendimiento y paralelización**
#    - Extracción de features con `multiprocessing.Pool` (1, 2 y 4 núcleos).
#    - Speedup real vs ideal y eficiencia.
#    - Comparación CPU vs GPU.
# 
# 5. **Resultados destacados**
#    - Clásico (escenario B): **93.9 %** accuracy (falla principalmente en la fresa).
#    - MobileNetV2 128×128: **100 %** accuracy (CPU y GPU).
#    - Latencia GPU: **7.8 ms** (~128 img/s).
#    - Speedup de extracción de features con 4 núcleos: **1.99×**.
# 
# 6. **Funcionalidades de la aplicación web**
#    - Resumen, métricas, paralelización, matrices de confusión, galería y **clasificador en vivo**.
# 
# ---
# 
# ### Restricciones (limitaciones del proyecto)
# 
# 1. **Alcance del dataset**
#    - Solo **5 clases** de las 100 disponibles en COIL-100.
#    - Solo variación de **ángulo de vista** (rotación en el plano horizontal). No incluye cambios de iluminación, oclusión, escala, fondo o deformaciones reales.
# 
# 2. **Limitaciones del enfoque clásico**
#    - Muy sensible a cambios de punto de vista.
#    - Confunde sistemáticamente fresa y pimiento rojo (ambos rojos, redondeados y con zona verde superior) cuando se ven desde ángulos no entrenados.
#    - Depende fuertemente de features hechas a mano (HOG + LBP + HSV).
# 
# 3. **Limitaciones de recursos y entorno**
#    - Experimentos realizados principalmente en entorno **Kaggle** (4 vCPU, probablemente 2 núcleos físicos + hyper-threading).
#    - Con 4 procesos la eficiencia cae al ~50 % (overhead de procesos + serialización + disco compartido).
#    - El modelo clásico usa ~300 MB de RAM; MobileNetV2 sube a ~1.5 GB de RAM (aunque solo ~60 MB de VRAM en GPU).
# 
# 4. **Restricciones del deep learning**
#    - Depende de un modelo preentrenado en ImageNet (no se entrenó desde cero).
#    - Resolución importa: a 64×64 aún aparece un error; a 128×128 alcanza 100 %.
#    - Más pesado en memoria que el enfoque clásico.
# 
# 5. **Alcance general del experimento**
#    - No es un sistema de reconocimiento de objetos genérico ni en tiempo real en entorno abierto.
#    - No incluye detección (solo clasificación de imágenes ya recortadas/centradas).
#    - No evalúa robustez ante ruido, blur, cambios de iluminación o fondos complejos.
#    - La aplicación web es principalmente un **dashboard de resultados + demo**, no un producto de producción.
# 
# ---
# 
# **Resumen ejecutivo**  
# El proyecto demuestra de forma clara y cuantitativa que, en el problema de generalización a ángulos de vista nunca vistos (escenario B), **MobileNetV2 supera claramente al Random Forest clásico** (100 % vs 93.9 %), especialmente resolviendo la confusión fresa-pimiento, a costa de mayor consumo de memoria. El enfoque clásico sigue siendo más ligero y suficiente para el escenario fácil.
