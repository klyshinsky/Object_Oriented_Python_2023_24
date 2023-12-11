import numpy as np

dist_matrix = 0
positions = 0
point_count = 0
step_no = 0
ax = None
out = None

def init_globals(dist_matrix_, positions_, point_count_, step_no_, ax_, out_):
    global dist_matrix, positions, point_count, step_no, ax, out
    dist_matrix = dist_matrix_
    positions = positions_
    point_count = point_count_
    step_no = step_no_
    ax = ax_
    out = out_

# Генерация матрица расстояний для точек. 
# По умолчанию точки расположены на расстоянии 1 друг от друга.
def generate(arg):
    global dist_matrix, positions, point_count, step_no
    
    step_no = 1
    dist_matrix = np.ones((point_count, point_count))
    for i in range(point_count):
        dist_matrix[i,i] = 0 
        
    # Для части точек делаем случайное отклонение [-0.5; +0.5] от 1
    dif_pos = np.random.rand(3 * point_count, 2) * point_count
    dif_pos = dif_pos.astype(np.int32)
    for dp in dif_pos:
        n = np.random.rand(1) - 0.5
        dist_matrix[dp[0], dp[1]] += n
        dist_matrix[dp[1], dp[0]] += n
        
    # Сделаем три компоненты сильной связности, чтобы было видно как они стягиваются друг к другу.
    for k in [[0,5], [10,15], [20,25]]:
        for i in range(k[0], k[1]):
            for j in range(k[0], k[1]):
                if i == j:
                    continue
                n = np.random.rand(1) * 0.2
                dist_matrix[i, j] = n
                dist_matrix[j, i] = n
        
    # Для еще части увеличиваем расстояния.
    dif_pos = np.random.rand(3 * point_count, 2) * point_count
    dif_pos = dif_pos.astype(np.int32)
    for dp in dif_pos:
        n = np.random.standard_normal(1)[0] * 2
        dist_matrix[dp[0], dp[1]] += n
        dist_matrix[dp[1], dp[0]] += n

    # Могли получиться отрицательные расстояния.
    dist_matrix = np.absolute(dist_matrix)
    positions = np.random.rand(point_count, 2)
    # Рисуем начальное расположение точек.
    moveAndDrawPoints(None)

# Расчет силы, действующей на две точки.
def calcForce(pos1, pos2, dist):
    v1 = pos1 - pos2
    rel = np.linalg.norm(v1) / dist
    # Если (расстояние на плоскости) / (расстояние между точками в исходном пространстве) > 1
    # то притягиваются, в противном случае - отталкиваются.
    if rel > 1:
        return v1 * rel
    else:
        return -2 * v1 / rel

# Рассчитываем вектора смещений для точек в зависимости от суммы сил,
# действующих на каждую точку.
def calcVectors(dist_matrix, positions):
    vectors = np.zeros((point_count, 2))
    for i in range(point_count):
        v = np.zeros(2)
        for j in range(point_count):
            if i == j:
                continue
            v += calcForce(positions[j], positions[i], dist_matrix[i, j])
                
        vectors[i] += v
    # Шаг будем сокращать с течением времени.
    vectors = np.array(vectors) * 0.01 / np.cbrt(step_no+10)
    vectors[vectors>1] = 1
    return vectors

# Здесь имитируем UMAP, добавляя силу, которая пытаетсяя разместить их на круге.
# Не очень хорошо видна какая-либо разница.
def calcVectors2(dist_matrix, positions):
    vectors = np.zeros((point_count, 2))
    # это расчет силы, отталкивающей точки от их центра масс.
    cx = sum([p[0] for p in positions]) / point_count
    cy = sum([p[1] for p in positions]) / point_count
    center = np.array([cx, cy])
    d = positions - center
    vectors = 100 * d / np.linalg.norm(d) ** 2

    # Это расчет вектора смещений точек.
    for j in range(point_count):
        n1 = positions - positions[j]
        n2 = np.linalg.norm(n1, axis=1) / dist_matrix[j]
        n3 = np.zeros((point_count, 2))
        for i, nn1 in enumerate(n1):
            if n2[i] > 1:
                n3[i] += nn1 * n2[i]
            elif n2[i] <= 1 and n2[i] != 0:
                n3[i] -= 2 * nn1 / (n2[i]+1e-6)
        vectors[j] += sum(n3)
    vectors *= 0.01 / np.cbrt(step_no+10)
    vectors[vectors>0.5] = 0.5
    return vectors

# Эта функция не возвращает вектор смещений, а просто смещает точки.
# Плюс, здесь сделан аналог далекий SGD - точка перемещается сразу после расчета вектора.
def calcPositions(dist_matrix, positions):
    vectors = np.zeros((point_count, 2))
    cx = sum([p[0] for p in positions]) / point_count
    cy = sum([p[1] for p in positions]) / point_count
    center = np.array([cx, cy])
    d = positions - center
    for i, p in enumerate(d):
        vectors[i] = 0.1 * p / np.linalg.norm(p) ** 2
        if any(vectors[i]>0.1):
            vectors[i] /= (max(vectors[i]) * 10)
#     positions += vectors

    for j in range(point_count):
        for m in range(5):
            n1 = positions - positions[j]
            n2 = np.linalg.norm(n1, axis=1) / dist_matrix[j]
            n3 = np.zeros((point_count, 2))
            for i, nn1 in enumerate(n1):
                if n2[i] > 1:
                    n3[i] += nn1 * n2[i]
                elif n2[i] <= 1 and n2[i] != 0:
                    n3[i] -= 2 * nn1 / (n2[i]+1e-6)
            v = sum(n3) * 0.01 / np.cbrt(step_no+10) + vectors[j]
            if any(v>0.5):
                v /= max(v) * 2
            positions[j] += v
    return positions

# Рассчитывает нове положения точек и отрисовывает.
def moveAndDrawPoints(arg):
    global dist_matrix, positions, step_no, ax, out
    if arg != None or step_no != 1:
        positions += calcVectors(dist_matrix, positions)
    max_coords = np.unravel_index(dist_matrix.argmax(), dist_matrix.shape)
    small_pos1, small_pos2 = np.where((dist_matrix<0.2) & (dist_matrix!=0))
    ax.clear()
    ax.scatter(positions[:,0], positions[:,1], s=50)
    ax.scatter(positions[max_coords[0],0], positions[max_coords[0],1], s=50, c='b')
    ax.scatter(positions[max_coords[1],0], positions[max_coords[1],1], s=50, c='b')

    for pos in zip(small_pos1, small_pos2):
        ax.plot([positions[pos[0],0],positions[pos[1],0]], [positions[pos[0],1],positions[pos[1],1]])

    with out:
        out.clear_output(wait=True)
        display(ax.figure)
    step_no += 1
        
# Рассчитывает нове положения точек и отрисовывает (для имитации UMAP).
def moveAndDrawPoints2(arg):
    global dist_matrix, positions, step_no, ax, out
    cx = sum([p[0] for p in positions]) / point_count
    cy = sum([p[1] for p in positions]) / point_count
    if arg != None or step_no != 1:
#         positions += calcVectors2(dist_matrix, positions)
        positions = calcPositions(dist_matrix, positions)
    
    max_coords = np.unravel_index(dist_matrix.argmax(), dist_matrix.shape)
    small_pos1, small_pos2 = np.where((dist_matrix<0.2) & (dist_matrix!=0))
    ax.clear()
    ax.scatter([cx], [cy], s=50, c='g')
    ax.scatter(positions[:,0], positions[:,1], s=50)
    ax.scatter(positions[max_coords[0],0], positions[max_coords[0],1], s=50, c='b')
    ax.scatter(positions[max_coords[1],0], positions[max_coords[1],1], s=50, c='b')

    for pos in zip(small_pos1, small_pos2):
        ax.plot([positions[pos[0],0],positions[pos[1],0]], [positions[pos[0],1],positions[pos[1],1]])

    with out:
        out.clear_output(wait=True)
        display(ax.figure)
    step_no += 1
