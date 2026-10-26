import os
import time
import cv2
import numpy as np
from numba import njit, prange
import matplotlib

# Configuración automática de gráficos para WSL
for backend in ['TkAgg', 'QtAgg', 'WebAgg']:
    try:
        matplotlib.use(backend, force=True)
        import matplotlib.pyplot as plt
        break
    except Exception:
        continue

# =========================================================================
# FUNCIONES COMPILADAS EN CÓDIGO MÁQUINA CON NUMBA (@njit)
# =========================================================================

@njit(parallel=True, fastmath=True)
def calcular_distancias_minimas_sq(X, centroids_actuales):
    n_samples = X.shape[0]
    k_actual = centroids_actuales.shape[0]
    dist_sq = np.empty(n_samples, dtype=np.float32)
    
    for i in prange(n_samples):
        min_d = 1e30
        r, g, b = X[i, 0], X[i, 1], X[i, 2]
        for j in range(k_actual):
            dr = r - centroids_actuales[j, 0]
            dg = g - centroids_actuales[j, 1]
            db = b - centroids_actuales[j, 2]
            d = dr*dr + dg*dg + db*db
            if d < min_d:
                min_d = d
        dist_sq[i] = min_d
    return dist_sq

def inicializar_kmeans_plusplus_numba(X, K, seed=42):
    np.random.seed(seed)
    n_samples = X.shape[0]
    centroids = np.empty((K, 3), dtype=np.float32)
    
    primer_indice = np.random.randint(0, n_samples)
    centroids[0] = X[primer_indice]
    
    for k in range(1, K):
        dist_sq = calcular_distancias_minimas_sq(X, centroids[:k])
        probabilidades = dist_sq / np.sum(dist_sq)
        siguiente_indice = np.random.choice(n_samples, p=probabilidades)
        centroids[k] = X[siguiente_indice]
        
    return centroids

@njit(parallel=True, fastmath=True)
def asignar_clusters_numba(X, centroids):
    n_samples = X.shape[0]
    K = centroids.shape[0]
    labels = np.empty(n_samples, dtype=np.int64)
    
    for i in prange(n_samples):
        min_dist = 1e30
        best_cluster = 0
        r, g, b = X[i, 0], X[i, 1], X[i, 2]
        
        for j in range(K):
            dr = r - centroids[j, 0]
            dg = g - centroids[j, 1]
            db = b - centroids[j, 2]
            dist = dr*dr + dg*dg + db*db
            if dist < min_dist:
                min_dist = dist
                best_cluster = j
                
        labels[i] = best_cluster
    return labels

@njit(fastmath=True)
def actualizar_centroides_numba(X, labels, K, old_centroids):
    new_centroids = np.zeros((K, 3), dtype=np.float32)
    counts = np.zeros(K, dtype=np.int64)
    
    for i in range(X.shape[0]):
        c = labels[i]
        new_centroids[c, 0] += X[i, 0]
        new_centroids[c, 1] += X[i, 1]
        new_centroids[c, 2] += X[i, 2]
        counts[c] += 1
        
    for j in range(K):
        if counts[j] > 0:
            new_centroids[j, 0] /= counts[j]
            new_centroids[j, 1] /= counts[j]
            new_centroids[j, 2] /= counts[j]
        else:
            new_centroids[j] = old_centroids[j]
            
    return new_centroids

@njit(parallel=True, fastmath=True)
def generar_vista_iluminada_numba(img_cuantizada, labels_2d, cluster_idx, color_centroide):
    """
    Pinta la zona seleccionada con el color exacto del centroide y oscurece
    el fondo un 50% más (al 22% de luz) para lograr un contraste perfecto.
    """
    alto, ancho = labels_2d.shape
    salida = np.empty((alto, ancho, 3), dtype=np.uint8)
    r_cent, g_cent, b_cent = color_centroide[0], color_centroide[1], color_centroide[2]
    
    for y in prange(alto):
        for x in range(ancho):
            if labels_2d[y, x] == cluster_idx:
                # Color sólido y exacto del centroide
                salida[y, x, 0] = r_cent
                salida[y, x, 1] = g_cent
                salida[y, x, 2] = b_cent
            else:
                # Fondo cuantizado oscurecido un 50% extra (al 22% de brillo)
                salida[y, x, 0] = int(img_cuantizada[y, x, 0] * 0.22)
                salida[y, x, 1] = int(img_cuantizada[y, x, 1] * 0.22)
                salida[y, x, 2] = int(img_cuantizada[y, x, 2] * 0.22)
                
    return salida

def actualizar_scatter_3d(scatter_obj, facecolors=None, edgecolors=None, sizes=None):
    if facecolors is not None:
        scatter_obj.set_facecolors(facecolors)
        scatter_obj._facecolor3d = facecolors
    if edgecolors is not None:
        scatter_obj.set_edgecolors(edgecolors)
        scatter_obj._edgecolor3d = edgecolors
    if sizes is not None:
        scatter_obj.set_sizes(sizes)
        scatter_obj._sizes3d = sizes

# =========================================================================
# FUNCIÓN PRINCIPAL CON INTERACTIVIDAD EN TIEMPO REAL
# =========================================================================

def graficar_imagen_y_rgb():
    print("🚀 Preparando compilador JIT de Numba...")

    directorio_actual = os.path.dirname(os.path.abspath(__file__))
    ruta_imagen = os.path.join(directorio_actual, 'prueba.jpg')
    
    img_bgr = cv2.imread(ruta_imagen)
    if img_bgr is None:
        print(f"❌ Error: No se encontró la imagen en: {ruta_imagen}")
        return
        
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # Redimensionar para rendimiento fluido a 60 FPS en WSLg
    max_dim = 700
    alto_orig, ancho_orig = img_rgb.shape[:2]
    if max(alto_orig, ancho_orig) > max_dim:
        escala = max_dim / max(alto_orig, ancho_orig)
        img_rgb = cv2.resize(img_rgb, (int(ancho_orig * escala), int(alto_orig * escala)), interpolation=cv2.INTER_AREA)
    
    img_pequena = cv2.resize(img_rgb, (250, 250), interpolation=cv2.INTER_NEAREST)
    X = np.ascontiguousarray(img_pequena.reshape(-1, 3), dtype=np.float32)
    
    K = 15
    epochs = 100

    # Calentamiento JIT
    _ = asignar_clusters_numba(X[:10], X[:2])
    _ = actualizar_centroides_numba(X[:10], np.zeros(10, dtype=np.int64), 2, X[:2])
    dummy_img = np.zeros((4, 4, 3), dtype=np.uint8)
    dummy_lbl = np.zeros((4, 4), dtype=np.int64)
    _ = generar_vista_iluminada_numba(dummy_img, dummy_lbl, 0, np.array([255, 255, 255], dtype=np.uint8))

    inicio_tiempo = time.time()
    centroids = inicializar_kmeans_plusplus_numba(X, K, seed=42)
    
    for epoch in range(epochs):
        labels = asignar_clusters_numba(X, centroids)
        new_centroids = actualizar_centroides_numba(X, labels, K, centroids)
        if np.allclose(centroids, new_centroids, atol=1e-4):
            centroids = new_centroids
            break
        centroids = new_centroids

    print(f"⚡ Entrenamiento completado en {time.time() - inicio_tiempo:.4f} s.")
    
    X_orig = np.ascontiguousarray(img_rgb.reshape(-1, 3), dtype=np.float32)
    labels_original = asignar_clusters_numba(X_orig, centroids)
    img_cuantizada = np.ascontiguousarray(np.round(centroids[labels_original]).astype(np.uint8).reshape(img_rgb.shape))
    labels_2d = np.ascontiguousarray(labels_original.reshape(img_rgb.shape[0], img_rgb.shape[1]))

    colores_centroides_int = np.round(centroids).astype(np.uint8)

    conteos_clusters = np.bincount(labels_original, minlength=K)
    porcentajes_clusters = (conteos_clusters / labels_original.size) * 100.0

    indices_grafica = np.random.choice(X.shape[0], size=min(2500, X.shape[0]), replace=False)
    R = X[indices_grafica, 0]
    G = X[indices_grafica, 1]
    B = X[indices_grafica, 2]
    labels_grafica = labels[indices_grafica]
    colores_kmeans = centroids[labels_grafica] / 255.0
    colores_centroides_norm = centroids / 255.0

    rgba_puntos_normal = np.column_stack([colores_kmeans, np.full(len(colores_kmeans), 0.35)])
    rgba_puntos_activo = rgba_puntos_normal.copy()
    rgba_cent_normal = np.column_stack([colores_centroides_norm, np.ones(K)])
    bordes_cent_normal = np.tile([0.0, 0.0, 0.0, 1.0], (K, 1))
    tamanos_cent_normal = np.full(K, 140)

    # 4. Crear la ventana de visualización
    fig = plt.figure(figsize=(16, 5))
    
    ax1 = fig.add_subplot(1, 3, 1)
    ax1.imshow(img_rgb)
    ax1.set_title("Imagen Original")
    ax1.axis('off')
    
    ax2 = fig.add_subplot(1, 3, 2)
    im_cuantizada_plot = ax2.imshow(img_cuantizada)
    titulo_cuant_default = f"Imagen Cuantizada (K={K})\n[Haz clic en un centroide 3D ->]"
    ax2.set_title(titulo_cuant_default)
    ax2.axis('off')
    
    ax3 = fig.add_subplot(1, 3, 3, projection='3d')
    ax3.computed_zorder = False
    
    scatter_puntos = ax3.scatter(
        R, G, B, c=rgba_puntos_normal, marker='.', depthshade=False, zorder=1
    )
    
    scatter_centroides = ax3.scatter(
        centroids[:, 0], centroids[:, 1], centroids[:, 2],
        c=rgba_cent_normal, s=tamanos_cent_normal, marker='o', edgecolors=bordes_cent_normal, 
        linewidths=1.8, label='Centroides (Clic aqui)',
        picker=True, pickradius=8, depthshade=False, zorder=10
    )
    
    ax3.set_xlabel('Canal Rojo (R)', color='red', fontweight='bold')
    ax3.set_ylabel('Canal Verde (G)', color='green', fontweight='bold')
    ax3.set_zlabel('Canal Azul (B)', color='blue', fontweight='bold')
    ax3.set_xlim([0, 255])
    ax3.set_ylim([0, 255])
    ax3.set_zlim([0, 255])
    ax3.set_title("Espacio RGB 3D (Interactivo)")
    ax3.legend()

    # =========================================================================
    # LÓGICA DE EVENTOS
    # =========================================================================
    estado = {'cluster_activo': None}

    def restaurar_vista():
        estado['cluster_activo'] = None
        im_cuantizada_plot.set_data(img_cuantizada)
        ax2.set_title(titulo_cuant_default)
        
        actualizar_scatter_3d(scatter_puntos, facecolors=rgba_puntos_normal, edgecolors=rgba_puntos_normal)
        actualizar_scatter_3d(
            scatter_centroides,
            facecolors=rgba_cent_normal,
            edgecolors=bordes_cent_normal,
            sizes=tamanos_cent_normal
        )
        fig.canvas.draw_idle()

    def al_hacer_clic_centroide(event):
        if event.artist != scatter_centroides:
            return
            
        cluster_idx = int(event.ind[0])
        
        if estado['cluster_activo'] == cluster_idx:
            restaurar_vista()
            return
            
        estado['cluster_activo'] = cluster_idx
        
        r_val, g_val, b_val = [int(x) for x in colores_centroides_int[cluster_idx]]
        porcentaje = porcentajes_clusters[cluster_idx]
        
        # Generar vista con el color sólido exacto y fondo más oscuro
        cuant_iluminada = generar_vista_iluminada_numba(
            img_cuantizada, labels_2d, cluster_idx, colores_centroides_int[cluster_idx]
        )
        
        im_cuantizada_plot.set_data(cuant_iluminada)
        ax2.set_title(
            f"Cluster #{cluster_idx} | RGB: ({r_val}, {g_val}, {b_val}) ({porcentaje:.1f}%)\n"
            f"[Clic en la imagen o en el centroide para restaurar]"
        )
        
        rgba_puntos_activo[:, 3] = np.where(labels_grafica == cluster_idx, 0.95, 0.02)
        actualizar_scatter_3d(scatter_puntos, facecolors=rgba_puntos_activo, edgecolors=rgba_puntos_activo)
        
        tamanos = np.full(K, 55)
        tamanos[cluster_idx] = 320

        colores_cent = rgba_cent_normal.copy()
        colores_cent[:, 3] = 0.35
        colores_cent[cluster_idx, 3] = 1.0

        bordes = np.tile([0.0, 0.0, 0.0, 0.3], (K, 1))
        bordes[cluster_idx] = [0.0, 1.0, 1.0, 1.0]

        actualizar_scatter_3d(
            scatter_centroides,
            facecolors=colores_cent,
            edgecolors=bordes,
            sizes=tamanos
        )
        
        print(f"💡 Cluster #{cluster_idx} -> RGB: ({r_val}, {g_val}, {b_val}) | Área: {porcentaje:.2f}%")
        fig.canvas.draw_idle()

    def al_hacer_clic_ventana(event):
        if event.inaxes == ax2 and estado['cluster_activo'] is not None:
            restaurar_vista()

    fig.canvas.mpl_connect('pick_event', al_hacer_clic_centroide)
    fig.canvas.mpl_connect('button_press_event', al_hacer_clic_ventana)

    plt.tight_layout()
    print(f"✨ ¡Modo interactivo en tiempo real listo! Backend: {matplotlib.get_backend()}")
    plt.show()

if __name__ == "__main__":
    graficar_imagen_y_rgb()