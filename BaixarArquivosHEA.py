import os
import requests
from bs4 import BeautifulSoup

BASE_URL = "https://physionet.org/files/hd-semg/1.0.0/pr_dataset/"
OUTPUT_DIR = "pr_dataset_files"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def get_links(url):
    """Retorna todos os links de uma página PhysioNet."""
    resp = requests.get(url)
    if resp.status_code != 200:
        raise Exception(f"Erro ao acessar {url}: {resp.status_code}")
    soup = BeautifulSoup(resp.text, "html.parser")
    return [a.get("href") for a in soup.find_all("a") if a.get("href")]

# 1. Listar subpastas (subjects) dentro de pr_dataset
subjects = [link for link in get_links(BASE_URL) if link.endswith("/")]

for subj in subjects:
    subj_url = BASE_URL + subj
    subj_links = get_links(subj_url)

    # 2. Filtrar arquivos que começam com "maintenance_preprocess_" e terminam em .hea ou .dat
    wanted_files = [
        link for link in subj_links
        if (link.startswith("maintenance_preprocess_") and (link.endswith(".hea") or link.endswith(".dat")))
    ]

    if wanted_files:
        # Criar subpasta local para cada subject
        subj_dir = os.path.join(OUTPUT_DIR, subj.strip("/"))
        os.makedirs(subj_dir, exist_ok=True)

        # 3. Baixar cada arquivo
        for fname in wanted_files:
            file_url = subj_url + fname
            local_path = os.path.join(subj_dir, fname)

            print(f"Baixando {file_url} -> {local_path}")
            r = requests.get(file_url)
            if r.status_code == 200:
                with open(local_path, "wb") as f:
                    f.write(r.content)
            else:
                print(f"Falha ao baixar {file_url}")
