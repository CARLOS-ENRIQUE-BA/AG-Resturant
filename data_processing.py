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

def calcular_rentabilidad(combo):
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
    
    # Penalizar si el costo total supera el presupuesto o si el combo tiene menos de un número mínimo de productos
    if costo_total > BUDGET or num_productos < NUMERO_PROD_POR_COMBO:
        return 0, costo_total, precio_total, precio_total, popularidad_total, num_productos

    # Aplicar descuento al precio del combo
    precio_total_combo = precio_total * (1 - DESCUENTO_MAXIMO)
    
    # Rentabilidad ponderada por popularidad
    return (precio_total_combo - costo_total) * popularidad_total, costo_total, precio_total_combo, precio_total, popularidad_total, num_productos

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

def algoritmo_genetico():
    poblacion = [crear_individuo() for _ in range(TAMANO_POBLACION)]
    mejor_individuo = None
    mejor_rentabilidad = 0
    generacion_mejor_combo = -1

    for generacion in range(NUMERO_GENERACIONES):
        evaluaciones = [calcular_rentabilidad(ind) for ind in poblacion]
        fitness_values, costos, precios_con_descuento, precios_sin_descuento, popularidades, num_productos = zip(*evaluaciones)
        mejor_indice = np.argmax(fitness_values)
        
        if fitness_values[mejor_indice] > mejor_rentabilidad:
            mejor_individuo = poblacion[mejor_indice]
            mejor_rentabilidad = fitness_values[mejor_indice]
            generacion_mejor_combo = generacion
        
        print(f"Generación {generacion+1}:")
        print(f"  Rentabilidad mejor combo = {mejor_rentabilidad}")
        print(f"  Costo Total = {costos[mejor_indice]}")
        print(f"  Precio Total con Descuento = {precios_con_descuento[mejor_indice]}")
        print(f"  Precio Total sin Descuento = {precios_sin_descuento[mejor_indice]}")
        print(f"  Popularidad Total = {popularidades[mejor_indice]}")
        print(f"  Número de Productos = {num_productos[mejor_indice]}")
        print("")

        nueva_poblacion = []
        for _ in range(TAMANO_POBLACION // 2):
            padres_indices = np.random.choice(range(TAMANO_POBLACION), size=2, replace=False)
            padre1, padre2 = poblacion[padres_indices[0]], poblacion[padres_indices[1]]
            hijo1, hijo2 = crossover(padre1, padre2)
            if np.random.rand() < PROBABILIDAD_MUTACION:
                mutar(hijo1)
            if np.random.rand() < PROBABILIDAD_MUTACION:
                mutar(hijo2)
            nueva_poblacion.extend([hijo1, hijo2])
        poblacion = nueva_poblacion

    return mejor_individuo, generacion_mejor_combo

# Ejecución del algoritmo genético
mejor_combo, generacion_mejor_combo = algoritmo_genetico()

# Obtener detalles del mejor combo
productos_combo = [list(productos.keys())[i] for i in range(N_PRODUCTOS) if mejor_combo[i]]
costo_total_mejor_combo = sum(productos[producto]['costo'] for producto in productos_combo)
precio_total_sin_descuento = sum(productos[producto]['precio'] for producto in productos_combo)
precio_total_mejor_combo = precio_total_sin_descuento * (1 - DESCUENTO_MAXIMO)
beneficio_neto_mejor_combo = precio_total_mejor_combo - costo_total_mejor_combo

print(f"Mejor combo encontrado en la generación {generacion_mejor_combo + 1}: {productos_combo}")
print(f"  Precio Total del Combo (con descuento) = {precio_total_mejor_combo:.2f} MXN")
print(f"  Precio Total del Combo (sin descuento) = {precio_total_sin_descuento:.2f} MXN")
print(f"  Costo de Producción Total del Combo = {costo_total_mejor_combo:.2f} MXN")
print(f"  Beneficio Neto del Combo = {beneficio_neto_mejor_combo:.2f} MXN")
