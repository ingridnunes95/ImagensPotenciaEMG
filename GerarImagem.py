import wfdb
import os
import numpy as np
import matplotlib.pyplot as plt

# Caminho da pasta onde você salvou os arquivos baixados
base_path = os.path.expanduser(r"C:\Users\Ingrid Nunes\Downloads\Potencia_img\pr_dataset_files\subject02_session2")

# Lista de arquivos que você quer processar
arquivos_para_processar = [
    "maintenance_preprocess_sample1",
    "maintenance_preprocess_sample2", 
    "maintenance_preprocess_sample11",
    "maintenance_preprocess_sample12",
    "maintenance_preprocess_sample13",
    "maintenance_preprocess_sample14",
    "maintenance_preprocess_sample19",
    "maintenance_preprocess_sample20"
    # Adicione mais arquivos conforme necessário
]

# === Função para processar um arquivo ===
def processar_arquivo(record_name, base_path):
    """Processa um arquivo e gera 4 imagens separadas + 1 imagem combinada"""
    
    # Caminho completo do arquivo
    record_path = os.path.join(base_path, record_name)
    
    print(f"\n{'='*50}")
    print(f"Processando: {record_name}")
    print(f"{'='*50}")
    
    try:
        # === 1. Ler o sinal com wfdb ===
        rec = wfdb.rdrecord(record_path)
        signal = rec.p_signal    # formato (N_amostras, 256 canais)
        fs = rec.fs              # taxa de amostragem

        print("Taxa de amostragem:", fs)
        print("Formato do sinal:", signal.shape)

        # === 2. Extrair informações para o nome do arquivo ===
        # Remove "maintenance_preprocess_" do nome
        nome_simplificado = record_name.replace("maintenance_preprocess_", "")
        # Adiciona o prefixo do sujeito e sessão
        nome_base = f"subject02_session2_{nome_simplificado}"

        # === 3. Janelamento ===
        win_size = int(0.5 * fs)  # 1 segundo
        step = int(0.125 * fs)    # overlap 0,125 s
        N, C = signal.shape

        windows = []
        for start in range(0, N - win_size + 1, step):
            end = start + win_size
            segment = signal[start:end, :]
            windows.append(segment)

        print("Total de janelas extraídas:", len(windows))

        # === 4. Potência por canal ===
        # Potência = média do quadrado do sinal por canal
        powers = [np.mean(w**2, axis=0) for w in windows]
        powers = np.array(powers)  # formato (n_janelas, 256)

        print("Formato do array de potências:", powers.shape)

        # === 5. Selecionar 4 momentos diferentes ao longo do sinal ===
        total_windows = len(windows)
        selected_indices = []

        if total_windows >= 4:
            # Seleciona 4 índices igualmente espaçados
            step_indices = max(1, total_windows // 4)
            selected_indices = [i * step_indices for i in range(4)]
            selected_indices = selected_indices[:4]
        else:
            # Se tiver menos de 4 janelas, usa todas disponíveis
            selected_indices = list(range(total_windows))

        print("Índices das janelas selecionadas:", selected_indices)

        # Lista para armazenar nomes de todas as imagens geradas
        todas_imagens_geradas = []

        # === 6. Gerar 4 imagens SEPARADAS ===
        print("\nGerando imagens individuais...")
        imagens_individuais = []
        
        for i, idx in enumerate(selected_indices):
            if idx < len(powers):
                # Reorganizar em grade 16×16
                img = powers[idx].reshape(16, 16)
                
                # Calcular tempo inicial da janela em segundos
                start_time = idx * step / fs
                
                # Criar nome do arquivo para imagem individual
                image_filename = f"{nome_base}_janela{idx+1}.png"
                
                # Criar figura individual para cada imagem
                plt.figure(figsize=(8, 6))
                plt.imshow(img, cmap="hot", interpolation="nearest")
                plt.colorbar(label="Potência")
                plt.title(f"Arquivo: {nome_base}\n"
                         f"Janela {idx+1} - Tempo: {start_time:.2f}s - {start_time + 1.0:.2f}s\n"
                         f"Potência por canal (16x16)")
                plt.xlabel("Coluna Eletrodos")
                plt.ylabel("Linha Eletrodos")
                plt.tight_layout()
                
                # Salvar imagem individual
                plt.savefig(image_filename, dpi=300)
                plt.close()
                
                imagens_individuais.append(image_filename)
                todas_imagens_geradas.append(image_filename)
                print(f"  Imagem individual {i+1} salva: {image_filename}")

        # === 7. Gerar 1 imagem COMBINADA com as 4 janelas ===
        if len(selected_indices) >= 4:
            print("\nGerando imagem combinada...")
            
            # Criar figura com subplots 2x2
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            axes = axes.flatten()
            
            # Encontrar valores mínimo e máximo para escala de cores consistente
            all_power_values = []
            for idx in selected_indices:
                if idx < len(powers):
                    img_data = powers[idx].reshape(16, 16)
                    all_power_values.extend(img_data.flatten())
            
            vmin = np.min(all_power_values)
            vmax = np.max(all_power_values)
            
            # Plotar cada janela em um subplot
            for i, idx in enumerate(selected_indices):
                if idx < len(powers) and i < 4:
                    # Reorganizar em grade 16×16
                    img = powers[idx].reshape(16, 16)
                    
                    # Calcular tempo inicial da janela em segundos
                    start_time = idx * step / fs
                    
                    # Plotar no subplot correspondente
                    im = axes[i].imshow(img, cmap="hot", interpolation="nearest", 
                                      vmin=vmin, vmax=vmax)
                    axes[i].set_title(f"Janela {idx+1}\nTempo: {start_time:.2f}s - {start_time + 1.0:.2f}s", 
                                    fontsize=12, fontweight='bold')
                    axes[i].set_xlabel("Coluna Eletrodos")
                    axes[i].set_ylabel("Linha Eletrodos")
                    
                    # Adicionar colorbar para cada subplot
                    plt.colorbar(im, ax=axes[i], label="Potência")
            
            # Ajustar layout e adicionar título principal
            plt.suptitle(f"Arquivo: {nome_base}\nMapas de Potência - 4 Janelas Temporais", 
                        fontsize=16, fontweight='bold', y=0.95)
            plt.tight_layout()
            
            # Nome do arquivo para a imagem combinada
            combined_filename = f"{nome_base}_4janelas_combinadas.png"
            
            # Salvar imagem combinada
            plt.savefig(combined_filename, dpi=300, bbox_inches='tight')
            plt.close()
            
            todas_imagens_geradas.append(combined_filename)
            print(f"  Imagem combinada salva: {combined_filename}")
        
        return todas_imagens_geradas

    except Exception as e:
        print(f"Erro ao processar {record_name}: {e}")
        return []

# === Processar todos os arquivos ===
print("Iniciando processamento de múltiplos arquivos...")
print(f"Arquivos a serem processados: {arquivos_para_processar}")

todas_imagens_geradas_por_arquivo = {}

for arquivo in arquivos_para_processar:
    imagens = processar_arquivo(arquivo, base_path)
    todas_imagens_geradas_por_arquivo[arquivo] = imagens

# === Resumo final ===
print(f"\n{'='*60}")
print("RESUMO DO PROCESSAMENTO")
print(f"{'='*60}")

total_imagens = 0
total_arquivos = 0

for arquivo, imagens in todas_imagens_geradas_por_arquivo.items():
    print(f"\n{arquivo}:")
    print(f"  {len(imagens)} imagem(s) gerada(s)")
    
    # Separar imagens individuais e combinadas
    individuais = [img for img in imagens if "combinadas" not in img]
    combinadas = [img for img in imagens if "combinadas" in img]
    
    if individuais:
        print("  - Individuais:")
        for img in individuais:
            print(f"      {img}")
    if combinadas:
        print("  - Combinadas:")
        for img in combinadas:
            print(f"      {img}")
    
    total_imagens += len(imagens)
    total_arquivos += 1

print(f"\nTotal de arquivos processados: {total_arquivos}")
print(f"Total de imagens geradas: {total_imagens}")
print(f"  - Imagens individuais: {total_arquivos * 4}")
print(f"  - Imagens combinadas: {total_arquivos}")
print(f"\nTodas as imagens foram salvas no diretório: {os.getcwd()}")