import json

INPUT = r"C:\Users\rogst\Downloads\fundamentosIA\Q_table_Nuevo.json"
OUTPUT = r"C:\Users\rogst\Downloads\fundamentosIA\Q_table_NuevoCleaned.json"

# Umbral mínimo para conservar una entrada
MIN_N = 2   # Numero que dicta lo que quiero limpiar, eliminas las entradas un numero igual o menor al definido, para quitar entradas basuras

with open(INPUT, "r") as f:
    data = json.load(f)

Q_old = data["Q"]
N_old = data["N"]

Q_new = {}
N_new = {}

for key, n in N_old.items():
    if n >= MIN_N:
        Q_new[key] = Q_old[key]
        N_new[key] = n

print("Entradas originales:", len(Q_old))
print("Entradas conservadas:", len(Q_new))
print("Entradas eliminadas:", len(Q_old) - len(Q_new))

with open(OUTPUT, "w") as f:
    json.dump({"Q": Q_new, "N": N_new}, f)

print(f"Archivo limpio guardado en:\n{OUTPUT}")

