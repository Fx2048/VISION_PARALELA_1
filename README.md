# VISION_PARALELA_1

# %% [markdown]
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

# %% [code] {"execution":{"iopub.status.busy":"2026-10-21T00:28:03.709204Z","iopub.execute_input":"2026-10-21T00:28:03.710064Z","iopub.status.idle":"2026-10-21T00:28:04.834632Z","shell.execute_reply.started":"2026-10-21T00:28:03.710030Z","shell.execute_reply":"2026-10-21T00:28:04.833798Z"}}
from pathlib import Path
import os

print("Directorio actual:", os.getcwd())

# Busca cualquier PNG/JPG bajo data/ (o bajo todo el directorio actual si no existe)
base = Path("/kaggle/input/datasets/jessicali9530/coil100/coil-100") if Path("/kaggle/input/datasets/jessicali9530/coil100/coil-100").exists() else Path(".")
found = list(base.rglob("*.png")) + list(base.rglob("*.jpg"))
print("Total de imágenes encontradas:", len(found))
for p in found[:10]:
    print(p)

# %% [code] {"execution":{"iopub.status.busy":"2026-10-21T00:28:11.132836Z","iopub.execute_input":"2026-10-21T00:28:11.133439Z","iopub.status.idle":"2026-10-21T00:28:11.139244Z","shell.execute_reply.started":"2026-10-21T00:28:11.133411Z","shell.execute_reply":"2026-10-21T00:28:11.138290Z"}}
for root, dirs, files in os.walk("."):
    if root.count(os.sep) <= 2:
        print(root, "->", dirs[:5], files[:3])

# %% [code] {"execution":{"iopub.status.busy":"2026-10-21T00:28:15.236451Z","iopub.execute_input":"2026-10-21T00:28:15.237084Z","iopub.status.idle":"2026-10-21T00:28:39.145169Z","shell.execute_reply.started":"2026-10-21T00:28:15.237056Z","shell.execute_reply":"2026-10-21T00:28:39.144405Z"}}
import cv2, numpy as np, re
from pathlib import Path

RAW = Path("/kaggle/input/datasets/jessicali9530/coil100/coil-100")   # ajusta a la carpeta que te mostró el diagnóstico

# Indexa: objeto -> archivo de la vista más cercana a 0°
views = {}
for p in list(RAW.rglob("*.png")) + list(RAW.rglob("*.jpg")):
    m = re.match(r"obj(\d+)__(\d+)", p.stem)
    if m:
        obj, ang = int(m.group(1)), int(m.group(2))
        if obj not in views or ang < views[obj][0]:
            views[obj] = (ang, p)

print("Objetos encontrados:", len(views))

tiles = []
for i in range(1, 101):
    if i not in views:
        tiles.append(np.zeros((96, 96, 3), np.uint8))
        print("Falta objeto", i)
        continue
    img = cv2.imread(str(views[i][1]))
    img = cv2.resize(img, (96, 96))
    cv2.putText(img, str(i), (3, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
    tiles.append(img)

rows = [np.hstack(tiles[r*10:(r+1)*10]) for r in range(10)]
sheet = np.vstack(rows)

Path("figures").mkdir(exist_ok=True)
cv2.imwrite("figures/coil100_overview.png", sheet)

# %% [code] {"execution":{"iopub.status.busy":"2026-10-21T00:28:40.329359Z","iopub.execute_input":"2026-10-21T00:28:40.330299Z","iopub.status.idle":"2026-10-21T00:28:40.861714Z","shell.execute_reply.started":"2026-10-21T00:28:40.330254Z","shell.execute_reply":"2026-10-21T00:28:40.860276Z"}}
import matplotlib.pyplot as plt
plt.figure(figsize=(12, 12))
plt.imshow(cv2.cvtColor(sheet, cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.show()

# %% [code] {"execution":{"iopub.status.busy":"2026-10-21T00:28:46.027419Z","iopub.execute_input":"2026-10-21T00:28:46.027931Z","iopub.status.idle":"2026-10-21T00:29:19.920836Z","shell.execute_reply.started":"2026-10-21T00:28:46.027881Z","shell.execute_reply":"2026-10-21T00:29:19.919884Z"}}
import os, re, shutil
from pathlib import Path

os.chdir("/kaggle/working")

SELECTED_OBJECTS = [43, 63, 68, 74, 75]
RAW_DIR = Path("/kaggle/input")        # rglob encuentra coil-100 dentro
OUTPUT_DIR = Path("data/processed")

if OUTPUT_DIR.exists():
    shutil.rmtree(OUTPUT_DIR)

counts = {}
for p in RAW_DIR.rglob("*.png"):
    m = re.match(r"obj(\d+)__(\d+)$", p.stem)
    if not m:
        continue
    obj, angle = int(m.group(1)), int(m.group(2))
    if obj not in SELECTED_OBJECTS:
        continue
    block = (angle // 5) // 4
    split = "test" if block % 5 == 4 else "train"
    dest = OUTPUT_DIR / split / f"obj{obj:03d}"
    dest.mkdir(parents=True, exist_ok=True)
    shutil.copy(p, dest / p.name)
    counts[(split, obj)] = counts.get((split, obj), 0) + 1

for k in sorted(counts):
    print(k, counts[k])

# %% [code] {"execution":{"iopub.status.busy":"2026-10-21T00:29:31.516421Z","iopub.execute_input":"2026-10-21T00:29:31.517369Z","iopub.status.idle":"2026-10-21T00:29:31.525489Z","shell.execute_reply.started":"2026-10-21T00:29:31.517341Z","shell.execute_reply":"2026-10-21T00:29:31.524574Z"}}
import cv2
import numpy as np
from skimage.feature import hog, local_binary_pattern

IMG_SIZE = 128

def extract_features(img_path):
    img = cv2.imread(str(img_path))
    if img is None:
        raise FileNotFoundError(f"No se pudo leer: {img_path}")
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hist_h = cv2.calcHist([hsv], [0], None, [32], [0, 180])
    hist_s = cv2.calcHist([hsv], [1], None, [32], [0, 256])
    hist_v = cv2.calcHist([hsv], [2], None, [32], [0, 256])
    color_feat = np.concatenate([hist_h, hist_s, hist_v]).flatten()
    color_feat = color_feat / (color_feat.sum() + 1e-6)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hog_feat = hog(gray, orientations=9, pixels_per_cell=(16, 16),
                   cells_per_block=(2, 2), visualize=False)

    lbp = local_binary_pattern(gray, P=8, R=1, method="uniform")
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 11), density=True)

    return np.concatenate([color_feat, hog_feat, lbp_hist]).astype(np.float32)

# %% [code] {"execution":{"iopub.status.busy":"2026-10-21T00:29:36.234476Z","iopub.execute_input":"2026-10-21T00:29:36.234943Z","iopub.status.idle":"2026-10-21T00:29:42.359394Z","shell.execute_reply.started":"2026-10-21T00:29:36.234908Z","shell.execute_reply":"2026-10-21T00:29:42.358510Z"}}
import json
import time
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import psutil
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, precision_recall_fscore_support)
from tqdm import tqdm


def list_split(split):
    """Devuelve rutas y etiquetas, sin extraer características todavía."""
    paths, labels = [], []
    for class_dir in sorted(Path(f"data/processed/{split}").iterdir()):
        for p in sorted(class_dir.glob("*.png")):
            paths.append(p)
            labels.append(class_dir.name)
    return paths, np.array(labels)


def extract_all(paths, desc):
    return np.array([extract_features(p) for p in tqdm(paths, desc=desc)])


def rss_mb():
    return psutil.Process().memory_info().rss / 1024**2


def main():
    train_paths, y_train = list_split("train")
    test_paths, y_test = list_split("test")
    print(f"Train: {len(train_paths)} | Test: {len(test_paths)}")

    mem_start = rss_mb()

    # --- Extracción de características (train) ---
    t0 = time.perf_counter()
    X_train = extract_all(train_paths, "Features train")
    t_extract_train = time.perf_counter() - t0

    # --- Entrenamiento (secuencial: n_jobs=1) ---
    clf = RandomForestClassifier(n_estimators=100, n_jobs=1, random_state=42)
    t0 = time.perf_counter()
    clf.fit(X_train, y_train)
    t_fit = time.perf_counter() - t0

    # --- Inferencia imagen por imagen: extracción + predict ---
    latencies, y_pred = [], []
    for p in tqdm(test_paths, desc="Inferencia"):
        t0 = time.perf_counter()
        feat = extract_features(p)
        y_pred.append(clf.predict(feat.reshape(1, -1))[0])
        latencies.append(time.perf_counter() - t0)
    y_pred = np.array(y_pred)
    lat_ms = np.array(latencies) * 1000

    mem_peak = rss_mb()

    # --- Métricas ---
    acc = accuracy_score(y_test, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="macro", zero_division=0)
    labels = sorted(set(y_test))
    cm = confusion_matrix(y_test, y_pred, labels=labels)

    results = {
        "scenario": "clasico_secuencial",
        "accuracy": acc,
        "precision_macro": prec,
        "recall_macro": rec,
        "f1_macro": f1,
        "t_extract_train_s": t_extract_train,
        "t_fit_s": t_fit,
        "t_train_total_s": t_extract_train + t_fit,
        "latency_mean_ms": float(lat_ms.mean()),
        "latency_p95_ms": float(np.percentile(lat_ms, 95)),
        "throughput_img_s": float(len(test_paths) / np.sum(latencies)),
        "mem_start_mb": mem_start,
        "mem_peak_mb": mem_peak,
        "n_features": int(X_train.shape[1]),
    }

    print(classification_report(y_test, y_pred, zero_division=0))
    print(json.dumps(results, indent=2))

    # --- Guardado ---
    Path("results").mkdir(exist_ok=True)
    Path("figures").mkdir(exist_ok=True)
    with open("results/classic_sequential.json", "w") as f:
        json.dump(results, f, indent=2)
    joblib.dump(clf, "results/classic_model.joblib")

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicho")
    plt.ylabel("Real")
    plt.title("Matriz de confusión - Clásico secuencial")
    plt.tight_layout()
    plt.savefig("figures/cm_classic_sequential.png", dpi=150)
    plt.show()

    return results


results = main()

# %% [code] {"execution":{"iopub.status.busy":"2026-10-21T00:29:48.264202Z","iopub.execute_input":"2026-10-21T00:29:48.265099Z","iopub.status.idle":"2026-10-21T00:29:48.269918Z","shell.execute_reply.started":"2026-10-21T00:29:48.265069Z","shell.execute_reply":"2026-10-21T00:29:48.268841Z"}}
import os
os.rename("results/classic_sequential.json", "results/classic_sequential_A.json")
os.rename("figures/cm_classic_sequential.png", "figures/cm_classic_sequential_A.png")

# %% [markdown]
# ---------------------ESCENARIO FACIL CULMINADO---------------------------------------

# %% [code] {"execution":{"iopub.status.busy":"2026-10-21T00:29:54.806102Z","iopub.execute_input":"2026-10-21T00:29:54.806558Z","iopub.status.idle":"2026-10-21T00:29:55.499900Z","shell.execute_reply.started":"2026-10-21T00:29:54.806530Z","shell.execute_reply":"2026-10-21T00:29:55.499063Z"}}
import os, re, shutil
from pathlib import Path

os.chdir("/kaggle/working")

SELECTED_OBJECTS = [43, 63, 68, 74, 75]
RAW_DIR = Path("/kaggle/input")
OUTPUT_DIR = Path("data/processed")

if OUTPUT_DIR.exists():
    shutil.rmtree(OUTPUT_DIR)

counts = {}
for p in RAW_DIR.rglob("*.png"):
    m = re.match(r"obj(\d+)__(\d+)$", p.stem)
    if not m:
        continue
    obj, angle = int(m.group(1)), int(m.group(2))
    if obj not in SELECTED_OBJECTS:
        continue
    split = "train" if angle < 180 else "test"
    dest = OUTPUT_DIR / split / f"obj{obj:03d}"
    dest.mkdir(parents=True, exist_ok=True)
    shutil.copy(p, dest / p.name)
    counts[(split, obj)] = counts.get((split, obj), 0) + 1

for k in sorted(counts):
    print(k, counts[k])

# %% [code] {"execution":{"iopub.status.busy":"2026-10-21T00:29:58.978563Z","iopub.execute_input":"2026-10-21T00:29:58.979051Z","iopub.status.idle":"2026-10-21T00:30:04.025514Z","shell.execute_reply.started":"2026-10-21T00:29:58.979024Z","shell.execute_reply":"2026-10-21T00:30:04.024497Z"}}
import json
import time
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import psutil
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, precision_recall_fscore_support)
from tqdm import tqdm


def list_split(split):
    paths, labels = [], []
    for class_dir in sorted(Path(f"data/processed/{split}").iterdir()):
        for p in sorted(class_dir.glob("*.png")):
            paths.append(p)
            labels.append(class_dir.name)
    return paths, np.array(labels)


def extract_all(paths, desc):
    return np.array([extract_features(p) for p in tqdm(paths, desc=desc)])


def rss_mb():
    return psutil.Process().memory_info().rss / 1024**2


def main():
    train_paths, y_train = list_split("train")
    test_paths, y_test = list_split("test")
    print(f"Train: {len(train_paths)} | Test: {len(test_paths)}")

    mem_start = rss_mb()

    t0 = time.perf_counter()
    X_train = extract_all(train_paths, "Features train")
    t_extract_train = time.perf_counter() - t0

    clf = RandomForestClassifier(n_estimators=100, n_jobs=1, random_state=42)
    t0 = time.perf_counter()
    clf.fit(X_train, y_train)
    t_fit = time.perf_counter() - t0

    latencies, y_pred = [], []
    for p in tqdm(test_paths, desc="Inferencia"):
        t0 = time.perf_counter()
        feat = extract_features(p)
        y_pred.append(clf.predict(feat.reshape(1, -1))[0])
        latencies.append(time.perf_counter() - t0)
    y_pred = np.array(y_pred)
    lat_ms = np.array(latencies) * 1000

    mem_peak = rss_mb()

    acc = accuracy_score(y_test, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="macro", zero_division=0)
    labels = sorted(set(y_test))
    cm = confusion_matrix(y_test, y_pred, labels=labels)

    results = {
        "scenario": "clasico_secuencial_B",
        "accuracy": acc,
        "precision_macro": prec,
        "recall_macro": rec,
        "f1_macro": f1,
        "t_extract_train_s": t_extract_train,
        "t_fit_s": t_fit,
        "t_train_total_s": t_extract_train + t_fit,
        "latency_mean_ms": float(lat_ms.mean()),
        "latency_p95_ms": float(np.percentile(lat_ms, 95)),
        "throughput_img_s": float(len(test_paths) / np.sum(latencies)),
        "mem_start_mb": mem_start,
        "mem_peak_mb": mem_peak,
        "n_features": int(X_train.shape[1]),
    }

    print(classification_report(y_test, y_pred, zero_division=0))
    print(json.dumps(results, indent=2))

    Path("results").mkdir(exist_ok=True)
    Path("figures").mkdir(exist_ok=True)
    with open("results/classic_sequential_B.json", "w") as f:
        json.dump(results, f, indent=2)
    joblib.dump(clf, "results/classic_model_B.joblib")

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicho")
    plt.ylabel("Real")
    plt.title("Matriz de confusión - Clásico secuencial B")
    plt.tight_layout()
    plt.savefig("figures/cm_classic_sequential_B.png", dpi=150)
    plt.show()

    return results


results_B = main()

# %% [markdown]
# ----------------- ESCENARIO INTERMEDIO CULMINADO ----------------------

# %% [code] {"execution":{"iopub.status.busy":"2026-10-21T00:32:12.070280Z","iopub.execute_input":"2026-10-21T00:32:12.070816Z","iopub.status.idle":"2026-10-21T00:32:18.095053Z","shell.execute_reply.started":"2026-10-21T00:32:12.070776Z","shell.execute_reply":"2026-10-21T00:32:18.093907Z"}}
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import psutil
import seaborn as sns
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, precision_recall_fscore_support)
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

torch.manual_seed(42)
np.random.seed(42)

train_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Entrenando en:", train_device)

IMG_SIZE = 128
tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

train_ds = datasets.ImageFolder("data/processed/train", transform=tf)
test_ds = datasets.ImageFolder("data/processed/test", transform=tf)
print("Clases:", train_ds.classes)
print(f"Train: {len(train_ds)} | Test: {len(test_ds)}")

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=2)

# Modelo compacto con transfer learning
model = models.mobilenet_v2(weights="DEFAULT")
model.classifier[1] = nn.Linear(model.last_channel, len(train_ds.classes))
model = model.to(train_device)

n_params = sum(p.numel() for p in model.parameters())
print(f"Parámetros: {n_params:,} (~{n_params * 4 / 1024**2:.1f} MB en float32)")

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

EPOCHS = 8
t0 = time.perf_counter()
model.train()
for epoch in range(EPOCHS):
    running_loss, correct, total = 0.0, 0, 0
    for images, labels in train_loader:
        images, labels = images.to(train_device), labels.to(train_device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * labels.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += labels.size(0)
    print(f"Epoch {epoch+1}/{EPOCHS} | loss {running_loss/total:.4f} | "
          f"train acc {100*correct/total:.1f}%")
t_train_total = time.perf_counter() - t0
print(f"Tiempo de entrenamiento: {t_train_total:.1f} s")

Path("results").mkdir(exist_ok=True)
Path("figures").mkdir(exist_ok=True)
torch.save(model.state_dict(), "results/cnn_mobilenet_B.pth")

# %% [code] {"execution":{"iopub.status.busy":"2026-10-21T00:32:28.632713Z","iopub.execute_input":"2026-10-21T00:32:28.633477Z","iopub.status.idle":"2026-10-21T00:32:32.820695Z","shell.execute_reply.started":"2026-10-21T00:32:28.633446Z","shell.execute_reply":"2026-10-21T00:32:32.819770Z"}}
def rss_mb():
    return psutil.Process().memory_info().rss / 1024**2


def evaluate_per_image(model, device, scenario):
    model = model.to(device).eval()
    classes = test_ds.classes

    # Calentamiento (la primera inferencia siempre es más lenta)
    dummy = torch.randn(1, 3, IMG_SIZE, IMG_SIZE).to(device)
    with torch.no_grad():
        for _ in range(5):
            model(dummy)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    mem_start = rss_mb()

    latencies, y_true, y_pred = [], [], []
    for path, label in test_ds.samples:
        t0 = time.perf_counter()
        img = Image.open(path).convert("RGB")
        x = tf(img).unsqueeze(0).to(device)
        with torch.no_grad():
            pred = model(x).argmax(1).item()   # .item() sincroniza con la GPU
        latencies.append(time.perf_counter() - t0)
        y_true.append(classes[label])
        y_pred.append(classes[pred])

    mem_peak = rss_mb()
    lat_ms = np.array(latencies) * 1000
    y_true, y_pred = np.array(y_true), np.array(y_pred)

    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=classes)

    results = {
        "scenario": scenario,
        "accuracy": acc,
        "precision_macro": prec,
        "recall_macro": rec,
        "f1_macro": f1,
        "t_train_total_s": t_train_total,
        "train_device": str(train_device),
        "latency_mean_ms": float(lat_ms.mean()),
        "latency_p95_ms": float(np.percentile(lat_ms, 95)),
        "throughput_img_s": float(len(latencies) / np.sum(latencies)),
        "mem_start_mb": mem_start,
        "mem_peak_mb": mem_peak,
        "n_params": int(n_params),
    }
    if device.type == "cuda":
        results["gpu_mem_peak_mb"] = torch.cuda.max_memory_allocated() / 1024**2

    print(classification_report(y_true, y_pred, zero_division=0))
    print(json.dumps(results, indent=2))

    with open(f"results/{scenario}.json", "w") as f:
        json.dump(results, f, indent=2)

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=classes, yticklabels=classes)
    plt.xlabel("Predicho")
    plt.ylabel("Real")
    plt.title(f"Matriz de confusión - {scenario}")
    plt.tight_layout()
    plt.savefig(f"figures/cm_{scenario}.png", dpi=150)
    plt.show()
    return results


results_cnn_cpu = evaluate_per_image(model, torch.device("cpu"), "cnn_mobilenet_cpu_B")

# %% [markdown]
# La CNN resolvió justo lo que le costó al clásico: el pimiento (63) y la fresa (75), que el clásico confundía, ahora salen perfectos.

# %% [code] {"execution":{"iopub.status.busy":"2026-10-21T00:38:23.767346Z","iopub.execute_input":"2026-10-21T00:38:23.767784Z","iopub.status.idle":"2026-10-21T00:38:25.714207Z","shell.execute_reply.started":"2026-10-21T00:38:23.767756Z","shell.execute_reply":"2026-10-21T00:38:25.713341Z"}}
#celda 9:
if torch.cuda.is_available():
    results_cnn_gpu = evaluate_per_image(model, torch.device("cuda"), "cnn_mobilenet_gpu_B")
else:
    print("No hay GPU disponible: se omite este escenario.")

# %% [code] {"execution":{"iopub.status.busy":"2026-10-21T00:38:35.295342Z","iopub.execute_input":"2026-10-21T00:38:35.295665Z","iopub.status.idle":"2026-10-21T00:38:55.648013Z","shell.execute_reply.started":"2026-10-21T00:38:35.295642Z","shell.execute_reply":"2026-10-21T00:38:55.646960Z"}}
import os, time, json
import multiprocessing as mp
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

cv2.setNumThreads(1)   # evita que OpenCV use hilos extra dentro de cada proceso

all_paths = sorted(Path("data/processed").rglob("*.png"))
n_cpu = os.cpu_count() or 1
print(f"Imágenes: {len(all_paths)} | CPUs disponibles: {n_cpu}")


def _worker(p):
    return extract_features(p)


def run_sequential():
    return [_worker(p) for p in all_paths]


def run_parallel(n_proc):
    ctx = mp.get_context("fork")
    with ctx.Pool(processes=n_proc) as pool:
        return pool.map(_worker, all_paths, chunksize=8)


REPS = 3
proc_list = [p for p in [1, 2, 4, 8] if p <= n_cpu]

run_sequential()   # calentamiento (caché de disco)

# Baseline secuencial: bucle simple, sin Pool
t_seq_runs = []
for _ in range(REPS):
    t0 = time.perf_counter()
    feats_seq = run_sequential()
    t_seq_runs.append(time.perf_counter() - t0)
t_seq = float(np.median(t_seq_runs))
print(f"Secuencial: {t_seq:.3f} s")

rows = [{"processes": 1, "time_s": t_seq, "speedup": 1.0, "efficiency": 1.0,
         "throughput_img_s": len(all_paths) / t_seq}]

for n_proc in [p for p in proc_list if p > 1]:
    runs = []
    for _ in range(REPS):
        t0 = time.perf_counter()
        feats_par = run_parallel(n_proc)
        runs.append(time.perf_counter() - t0)
    t_par = float(np.median(runs))
    speedup = t_seq / t_par
    rows.append({"processes": n_proc, "time_s": t_par, "speedup": speedup,
                 "efficiency": speedup / n_proc,
                 "throughput_img_s": len(all_paths) / t_par})
    ok = np.allclose(np.array(feats_seq), np.array(feats_par))
    print(f"{n_proc} procesos: {t_par:.3f} s | speedup {speedup:.2f}x | "
          f"eficiencia {speedup/n_proc:.0%} | mismos resultados: {ok}")

Path("results").mkdir(exist_ok=True)
Path("figures").mkdir(exist_ok=True)
with open("results/classic_parallel_B.json", "w") as f:
    json.dump({"scenario": "clasico_paralelo_B", "n_images": len(all_paths),
               "n_cpu": n_cpu, "reps": REPS, "runs": rows}, f, indent=2)

procs = [r["processes"] for r in rows]
plt.figure(figsize=(5, 4))
plt.plot(procs, [r["speedup"] for r in rows], "o-", label="Medido")
plt.plot(procs, procs, "--", label="Ideal")
plt.xlabel("Número de procesos")
plt.ylabel("Speedup")
plt.title("Speedup - extracción de características")
plt.legend()
plt.tight_layout()
plt.savefig("figures/speedup_classic_parallel_B.png", dpi=150)
plt.show()

# %% [code] {"execution":{"iopub.status.busy":"2026-10-21T00:38:55.649999Z","iopub.execute_input":"2026-10-21T00:38:55.650267Z","iopub.status.idle":"2026-10-21T00:39:01.896246Z","shell.execute_reply.started":"2026-10-21T00:38:55.650244Z","shell.execute_reply":"2026-10-21T00:39:01.895346Z"}}
#celda 11
# Guardar las variables del escenario 128 para restaurarlas al final
_bak = dict(IMG_SIZE=IMG_SIZE, tf=tf, test_ds=test_ds,
            t_train_total=t_train_total, n_params=n_params)

IMG64 = 64
tf64 = transforms.Compose([
    transforms.Resize((IMG64, IMG64)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
train_ds64 = datasets.ImageFolder("data/processed/train", transform=tf64)
test_ds64 = datasets.ImageFolder("data/processed/test", transform=tf64)
train_loader64 = DataLoader(train_ds64, batch_size=32, shuffle=True, num_workers=2)

torch.manual_seed(42)
model64 = models.mobilenet_v2(weights="DEFAULT")
model64.classifier[1] = nn.Linear(model64.last_channel, len(train_ds64.classes))
model64 = model64.to(train_device)
n_params64 = sum(p.numel() for p in model64.parameters())

criterion64 = nn.CrossEntropyLoss()
optimizer64 = torch.optim.Adam(model64.parameters(), lr=1e-4)

t0 = time.perf_counter()
model64.train()
for epoch in range(EPOCHS):
    running_loss, correct, total = 0.0, 0, 0
    for images, labels in train_loader64:
        images, labels = images.to(train_device), labels.to(train_device)
        optimizer64.zero_grad()
        outputs = model64(images)
        loss = criterion64(outputs, labels)
        loss.backward()
        optimizer64.step()
        running_loss += loss.item() * labels.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += labels.size(0)
    print(f"Epoch {epoch+1}/{EPOCHS} | loss {running_loss/total:.4f} | "
          f"train acc {100*correct/total:.1f}%")
t_train64 = time.perf_counter() - t0
print(f"Tiempo de entrenamiento: {t_train64:.1f} s")

torch.save(model64.state_dict(), "results/cnn_mobilenet_64_B.pth")

# Evaluar con la misma función de la Celda 8, apuntando a las variables de 64x64
IMG_SIZE, tf, test_ds = IMG64, tf64, test_ds64
t_train_total, n_params = t_train64, n_params64
results_cnn64 = evaluate_per_image(model64, torch.device("cpu"), "cnn_mobilenet_cpu_64_B")

# Restaurar las variables del escenario 128
globals().update(_bak)

# %% [code] {"execution":{"iopub.status.busy":"2026-10-21T00:40:21.226002Z","iopub.execute_input":"2026-10-21T00:40:21.226526Z","iopub.status.idle":"2026-10-21T00:40:21.232715Z","shell.execute_reply.started":"2026-10-21T00:40:21.226494Z","shell.execute_reply":"2026-10-21T00:40:21.231921Z"}}
#antes de seleccionar 12, no correr
import json
p = "results/classic_sequential_A.json"
d = json.load(open(p))
d["scenario"] = "clasico_secuencial_A"
json.dump(d, open(p, "w"), indent=2)

# %% [code] {"execution":{"iopub.status.busy":"2026-10-21T00:40:32.049104Z","iopub.execute_input":"2026-10-21T00:40:32.049602Z","iopub.status.idle":"2026-10-21T00:40:32.073686Z","shell.execute_reply.started":"2026-10-21T00:40:32.049572Z","shell.execute_reply":"2026-10-21T00:40:32.072827Z"}}
#celda 12:
import json
import pandas as pd
from pathlib import Path

rows = []
for f in sorted(Path("results").glob("*.json")):
    d = json.load(open(f))
    if isinstance(d, dict) and "accuracy" in d:   # salta el JSON del paralelo
        rows.append(d)

df = pd.DataFrame(rows)
cols = ["scenario", "accuracy", "f1_macro", "latency_mean_ms", "latency_p95_ms",
        "throughput_img_s", "mem_peak_mb", "t_train_total_s"]
df = df[[c for c in cols if c in df.columns]].round(3)
df.to_csv("results/comparacion.csv", index=False)
df

# %% [code] {"execution":{"iopub.status.busy":"2026-10-21T00:55:02.853270Z","iopub.execute_input":"2026-10-21T00:55:02.853679Z","iopub.status.idle":"2026-10-21T00:55:02.860249Z","shell.execute_reply.started":"2026-10-21T00:55:02.853654Z","shell.execute_reply":"2026-10-21T00:55:02.859204Z"}}
import json
print(json.dumps(json.load(open("results/classic_parallel_B.json")), indent=2))

# %% [code] {"execution":{"iopub.status.busy":"2026-10-21T01:44:39.184456Z","iopub.execute_input":"2026-10-21T01:44:39.185233Z","iopub.status.idle":"2026-10-21T01:44:45.772086Z","shell.execute_reply.started":"2026-10-21T01:44:39.185203Z","shell.execute_reply":"2026-10-21T01:44:45.771122Z"}}
!pip install -q onnx onnxscript

# %% [code] {"execution":{"iopub.status.busy":"2026-10-21T01:44:47.797355Z","iopub.execute_input":"2026-10-21T01:44:47.797703Z","iopub.status.idle":"2026-10-21T01:45:00.953699Z","shell.execute_reply.started":"2026-10-21T01:44:47.797676Z","shell.execute_reply":"2026-10-21T01:45:00.952790Z"}}
import shutil, os, torch
from torchvision import models
import torch.nn as nn

# 1. Exportar MobileNetV2 entrenado a ONNX (ideal para correr directo en la web)
model_web = models.mobilenet_v2(weights=None)
model_web.classifier[1] = nn.Linear(model_web.last_channel, 5)
model_web.load_state_dict(torch.load("/kaggle/working/results/cnn_mobilenet_B.pth", map_location="cpu"))
model_web.eval()

dummy_input = torch.randn(1, 3, 128, 128)
torch.onnx.export(
    model_web,
    dummy_input,
    "/kaggle/working/results/cnn_mobilenet.onnx",
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}}
)

# 2. Crear una carpeta con ejemplos de prueba para la web
os.makedirs("export_web/samples", exist_ok=True)
shutil.copytree("results", "export_web/results", dirs_exist_ok=True)
if os.path.exists("figures"):
    shutil.copytree("figures", "export_web/figures", dirs_exist_ok=True)

# Copiar 2-3 imágenes de muestra por cada objeto para probar en la página
for obj in ["obj043", "obj063", "obj068", "obj074", "obj075"]:
    src_dir = f"data/processed/test/{obj}"
    dst_dir = f"export_web/samples/{obj}"
    os.makedirs(dst_dir, exist_ok=True)
    if os.path.exists(src_dir):
        for img in sorted(os.listdir(src_dir))[:4]:
            shutil.copy(os.path.join(src_dir, img), os.path.join(dst_dir, img))

# 3. Empaquetar todo en un ZIP
shutil.make_archive("experimento_web", "zip", "export_web")
print("¡Listo! En el panel derecho de Kaggle (Output), descarga 'experimento_web.zip'")


# %% [code]
