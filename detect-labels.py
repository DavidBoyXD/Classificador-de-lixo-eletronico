import os

# Caminho para as labels do seu dataset
LABELS_DIR = "dataset/MobilePhone/test/labels"

corrigidos = 0
erros = 0

for root, _, files in os.walk(LABELS_DIR):
    for file in files:
        if not file.endswith(".txt"):
            continue

        path = os.path.join(root, file)

        try:
            with open(path, "r", encoding="utf-8") as f:
                lines = f.readlines()

            novas_linhas = []
            alterado = False

            for line in lines:
                line = line.strip()
                if not line:
                    continue

                partes = line.split()
                # Corrige apenas se a primeira parte (classe) for "1"
                if partes[0] == "1":
                    partes[0] = "0"
                    alterado = True

                novas_linhas.append(" ".join(partes) + "\n")

            if alterado:
                corrigidos += 1
                with open(path, "w", encoding="utf-8") as f:
                    f.writelines(novas_linhas)

        except Exception as e:
            erros += 1
            print(f"⚠️ Erro ao processar {path}: {e}")

print(f"\n✅ Correção concluída! {corrigidos} arquivos ajustados com sucesso.")
if erros:
    print(f"⚠️ {erros} arquivos não puderam ser processados.")
