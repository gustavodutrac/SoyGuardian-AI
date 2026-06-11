import os

base_path = "soybean.leaf.dataset"

for pasta in os.listdir(base_path):
    caminho = os.path.join(base_path, pasta)

    if os.path.isdir(caminho):
        quantidade = len(os.listdir(caminho))

        print(f"{pasta}: {quantidade} imagens")