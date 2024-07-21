import numpy as np

# Datos de productos
productos = {
    'Pozol': {'costo': 7, 'precio': 15, 'popularidad': 0.56},
    'Coca-Cola': {'costo': 18, 'precio': 25, 'popularidad': 0.44},
    'Quesadilla': {'costo': 12, 'precio': 22, 'popularidad': 0.30},
    'Gordita': {'costo': 10, 'precio': 18, 'popularidad': 0.30},
    'Taco': {'costo': 9, 'precio': 14, 'popularidad': 0.16},
    'Empanada': {'costo': 8, 'precio': 15, 'popularidad': 0.24},
    'Turrón': {'costo': 4, 'precio': 10, 'popularidad': 0.48},
    'Nuégado': {'costo': 5, 'precio': 12, 'popularidad': 0.52}
}

# Parámetros del algoritmo genético
N_PRODUCTOS = len(productos)
TAMANO_POBLACION = 20
NUMERO_GENERACIONES = 10
PROBABILIDAD_MUTACION = 0.1
NUMERO_PROD_POR_COMBO = 3  # Número mínimo de productos en un combo
BUDGET = 40  # Presupuesto máximo del combo
DESCUENTO_MAXIMO = 0.2  # Descuento máximo en el precio del combo

def crear_descuento():
    return np.random.uniform(0.01, DESCUENTO_MAXIMO)  # Empieza en 1% como mínimo

def calcular_rentabilidad(combo, descuento):
    costo_total = 0
    precio_total = 0
    popularidad_total = 0
    num_productos = 0
    for i, incluir in enumerate(combo):
        if incluir:
            producto = list(productos.keys())[i]
            costo_total += productos[producto]['costo']
            precio_total += productos[producto]['precio']
            popularidad_total += productos[producto]['popularidad']
            num_productos += 1
    
    if costo_total > BUDGET or num_productos < NUMERO_PROD_POR_COMBO:
        return 0, costo_total, precio_total, precio_total, popularidad_total, num_productos, descuento

    precio_total_combo = precio_total * (1 - descuento)
    
    if descuento == 0:
        return 0, costo_total, precio_total_combo, precio_total, popularidad_total, num_productos, descuento
    
    rentabilidad = (precio_total_combo - costo_total) * popularidad_total

    return rentabilidad, costo_total, precio_total_combo, precio_total, popularidad_total, num_productos, descuento

def crear_individuo():
    return np.random.randint(2, size=N_PRODUCTOS)

def crossover(padre1, padre2):
    punto_corte = np.random.randint(1, N_PRODUCTOS-1)
    hijo1 = np.concatenate((padre1[:punto_corte], padre2[punto_corte:]))
    hijo2 = np.concatenate((padre2[:punto_corte], padre1[punto_corte:]))
    return hijo1, hijo2

def mutar(individuo):
    idx = np.random.randint(0, N_PRODUCTOS)
    individuo[idx] = 1 - individuo[idx]  # Cambiar 0 a 1 o 1 a 0

def mutar_descuento(descuento):
    return np.random.uniform(0.01, DESCUENTO_MAXIMO)  # Empieza en 1% como mínimo

def algoritmo_genetico():
    poblacion = [crear_individuo() for _ in range(TAMANO_POBLACION)]
    descuentos = [crear_descuento() for _ in range(TAMANO_POBLACION)]
    
    for generacion in range(NUMERO_GENERACIONES):
        evaluaciones = [calcular_rentabilidad(ind, desc) for ind, desc in zip(poblacion, descuentos)]
        fitness_values, costos, precios_con_descuento, precios_sin_descuento, popularidades, num_productos, descuentos_aplicados = zip(*evaluaciones)
        
        # Filtrar descuentos válidos (no 0%)
        indices_validos = [i for i in range(len(descuentos_aplicados)) if descuentos_aplicados[i] > 0]
        fitness_validos = [fitness_values[i] for i in indices_validos]
        poblacion_validas = [poblacion[i] for i in indices_validos]
        descuentos_validos = [descuentos_aplicados[i] for i in indices_validos]
        
        if fitness_validos:
            mejor_indice = np.argmax(fitness_validos)
            mejor_individuo = poblacion_validas[mejor_indice]
            mejor_descuento = descuentos_validos[mejor_indice]
            mejor_rentabilidad = fitness_validos[mejor_indice]
            
            print(f"Generación {generacion+1}:")
            print(f"  Rentabilidad mejor combo = {mejor_rentabilidad}")
            print(f"  Costo Total = {costos[mejor_indice]}")
            print(f"  Precio Total con Descuento = {precios_con_descuento[mejor_indice]}")
            print(f"  Precio Total sin Descuento = {precios_sin_descuento[mejor_indice]}")
            print(f"  Popularidad Total = {popularidades[mejor_indice]}")
            print(f"  Número de Productos = {num_productos[mejor_indice]}")
            print(f"  Descuento Aplicado = {mejor_descuento*100:.2f}%")
            print("")

            nueva_poblacion = []
            nueva_descuentos = []
            for _ in range(TAMANO_POBLACION // 2):
                padres_indices = np.random.choice(range(len(poblacion_validas)), size=2, replace=False)
                padre1, padre2 = poblacion_validas[padres_indices[0]], poblacion_validas[padres_indices[1]]
                hijo1, hijo2 = crossover(padre1, padre2)
                
                if np.random.rand() < PROBABILIDAD_MUTACION:
                    mutar(hijo1)
                if np.random.rand() < PROBABILIDAD_MUTACION:
                    mutar(hijo2)
                
                nueva_poblacion.extend([hijo1, hijo2])
                nueva_descuentos.extend([mutar_descuento(descuentos_validos[padres_indices[0]]), mutar_descuento(descuentos_validos[padres_indices[1]])])
            
            poblacion = nueva_poblacion
            descuentos = nueva_descuentos

    return mejor_individuo, mejor_descuento

# Ejecución del algoritmo genético
mejor_combo, mejor_descuento = algoritmo_genetico()

# Obtener detalles del mejor combo
productos_combo = [list(productos.keys())[i] for i in range(N_PRODUCTOS) if mejor_combo[i]]
costo_total_mejor_combo = sum(productos[producto]['costo'] for producto in productos_combo)
precio_total_sin_descuento = sum(productos[producto]['precio'] for producto in productos_combo)
precio_total_mejor_combo = precio_total_sin_descuento * (1 - mejor_descuento)
beneficio_neto_mejor_combo = precio_total_mejor_combo - costo_total_mejor_combo

print(f"Mejor combo encontrado:")
print(f"  Productos: {productos_combo}")
print(f"  Precio Total del Combo (con descuento) = {precio_total_mejor_combo:.2f} MXN")
print(f"  Precio Total del Combo (sin descuento) = {precio_total_sin_descuento:.2f} MXN")
print(f"  Costo de Producción Total del Combo = {costo_total_mejor_combo:.2f} MXN")
print(f"  Beneficio Neto del Combo = {beneficio_neto_mejor_combo:.2f} MXN")
print(f"  Porcentaje de Descuento Aplicado = {mejor_descuento*100:.2f}%")
