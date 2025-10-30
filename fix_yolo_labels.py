#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fix_yolo_labels.py
Corrige automaticamente arquivos .txt no formato YOLO que tenham linhas
corrompidas (ex.: várias caixas concatenadas numa linha, tokens extras, truncamentos).
Cria backup dos originais e salva arquivos corrigidos em labels_fixed/ (modo seguro).
"""

import os
import re
import shutil
from pathlib import Path

# ---------- CONFIGURAÇÃO -----------
# Caminho para a pasta do dataset (alterar se necessário)
# Ex: "dataset/nomeDoDataset"
DATASET_DIR = "dataset/MobilePhone"

# Se True: salva arquivos corrigidos em pasta labels_fixed (não sobrescreve originais)
# Se False: sobrescreve os arquivos originais (após criar backup em labels_backup/)
DRY_RUN = True

# Tolerância para floats válidos (permitir pequenos erros numéricos fora do intervalo [0,1])
EPS = 1e-6

# Se True: remove caixas duplicadas (próximas) em cada arquivo
REMOVE_DUPLICATES = True
IOU_DUP_TOL = 1e-6  # se boxes idênticas dentro dessa tolerância, considera duplicada
# ------------------------------------

# diretórios de subsets padrão (ajuste se diferente)
SUBSETS = ["train", "valid", "test"]

# regex para extrair números (inteiros e floats, incluindo notação exponencial)
NUM_RE = re.compile(r'[-+]?\d*\.\d+|\d+')

def parse_numbers_from_text(text):
    """Retorna lista de tokens numéricos como strings na ordem em que aparecem."""
    return NUM_RE.findall(text)

def is_int_token(tok):
    try:
        int(tok)
        return True
    except:
        return False

def is_float_token(tok):
    try:
        float(tok)
        return True
    except:
        return False

def valid_box_tokens(class_tok, coords):
    """Verifica se token de classe e 4 coords são válidos (classe inteiro >=0, coords ~[0,1])."""
    if not is_int_token(class_tok):
        return False
    # parse class
    cls = int(class_tok)
    if cls < 0:
        return False
    # coords
    for c in coords:
        if not is_float_token(c):
            return False
        val = float(c)
        if val < -EPS or val > 1.0 + EPS:  # permite leve tolerância
            return False
    return True

def dedupe_boxes(boxes):
    """Remove caixas duplicadas idênticas (com tolerância). Boxes: list of tuples (cls,x,y,w,h)."""
    uniq = []
    for b in boxes:
        found = False
        for u in uniq:
            if b[0] == u[0] and all(abs(b[i]-u[i]) <= IOU_DUP_TOL for i in range(1,5)):
                found = True
                break
        if not found:
            uniq.append(b)
    return uniq

def fix_label_file(path_in, path_out):
    """
    Lê arquivo corrompido, extrai números, tenta reconstituir caixas válidas.
    Retorna (kept_boxes, discarded_count)
    """
    text = path_in.read_text(encoding='utf-8', errors='ignore')
    tokens = parse_numbers_from_text(text)
    if not tokens:
        return [], 0

    boxes = []
    i = 0
    discarded = 0
    n = len(tokens)
    # estratégia: procurar padrões [int][float float float float]
    while i <= n - 5:
        # try tokens[i] as class and next 4 as coords
        cls_tok = tokens[i]
        coords = tokens[i+1:i+5]
        if valid_box_tokens(cls_tok, coords):
            cls = int(cls_tok)
            vals = [float(c) for c in coords]
            boxes.append((cls, vals[0], vals[1], vals[2], vals[3]))
            i += 5
        else:
            # se não válidos, pular 1 token adiante (isso corrige concatenações)
            i += 1
            discarded += 1

    # Caso tenha encontrado zero caixas, tenta heurística alternativa:
    if not boxes:
        # às vezes cada linha é boa exceto por espaços estranhos; tentar por linhas
        for ln in text.splitlines():
            toks = parse_numbers_from_text(ln)
            if len(toks) >= 5:
                # tentar agrupar do começo em grupos de 5
                j = 0
                while j <= len(toks) - 5:
                    if valid_box_tokens(toks[j], toks[j+1:j+5]):
                        cls = int(toks[j])
                        vals = [float(c) for c in toks[j+1:j+5]]
                        boxes.append((cls, vals[0], vals[1], vals[2], vals[3]))
                        j += 5
                    else:
                        j += 1
            # se linha curta / inválida => ignorar
        # note: não tromos com discarded contagem precisa aqui

    # opcional: remover duplicatas próximas
    if REMOVE_DUPLICATES and boxes:
        boxes = dedupe_boxes(boxes)

    # salvar no arquivo de saída (em formato YOLO: uma caixa por linha)
    lines = []
    for b in boxes:
        lines.append(f"{int(b[0])} {b[1]:.12f} {b[2]:.12f} {b[3]:.12f} {b[4]:.12f}")

    path_out.write_text("\n".join(lines), encoding='utf-8')
    discarded_total = discarded
    return boxes, discarded_total

def process_dataset_labels(dataset_dir):
    dataset_dir = Path(dataset_dir)
    if not dataset_dir.exists():
        print(f"ERRO: dataset não encontrado: {dataset_dir}")
        return

    summary = {
        "files_total": 0,
        "files_fixed": 0,
        "files_skipped_empty": 0,
        "boxes_total_before": 0,
        "boxes_total_after": 0,
    }

    for split in SUBSETS:
        labels_dir = dataset_dir / split / "labels"
        if not labels_dir.exists():
            print(f"Aviso: labels não encontrado: {labels_dir} (pulando)")
            continue

        fixed_out_dir = dataset_dir / split / "labels_fixed"
        fixed_out_dir.mkdir(parents=True, exist_ok=True)

        backup_dir = dataset_dir / split / "labels_backup"
        backup_dir.mkdir(parents=True, exist_ok=True)

        for txt_file in labels_dir.glob("*.txt"):
            summary["files_total"] += 1
            # ler original
            original_text = txt_file.read_text(encoding='utf-8', errors='ignore').strip()
            # contar boxes aproximado antes (linhas válidas com 5 tokens)
            lines = [ln for ln in original_text.splitlines() if ln.strip()]
            boxes_before = 0
            for ln in lines:
                toks = parse_numbers_from_text(ln)
                if len(toks) >= 5 and is_int_token(toks[0]):
                    boxes_before += 1
            summary["boxes_total_before"] += boxes_before

            # se arquivo vazio -> copiar para backup e criar arquivo fix vazio
            if not original_text:
                summary["files_skipped_empty"] += 1
                (backup_dir / txt_file.name).write_text(original_text, encoding='utf-8')
                (fixed_out_dir / txt_file.name).write_text("", encoding='utf-8')
                continue

            # cria backup do original
            shutil.copy2(txt_file, backup_dir / txt_file.name)

            # processa e grava fixado (modo dry-run grava em labels_fixed)
            boxes_after, discarded = fix_label_file(txt_file, fixed_out_dir / txt_file.name)

            if boxes_after:
                summary["files_fixed"] += 1
                summary["boxes_total_after"] += len(boxes_after)
            else:
                # nenhum box recuperado -> avisar; cria arquivo vazio
                (fixed_out_dir / txt_file.name).write_text("", encoding='utf-8')

    # print resumo
    print("\n--- RESUMO DA CORREÇÃO ---")
    print(f"Arquivos processados: {summary['files_total']}")
    print(f"Arquivos corrigidos (com ao menos 1 caixa): {summary['files_fixed']}")
    print(f"Arquivos vazios pulados: {summary['files_skipped_empty']}")
    print(f"Caixas totais ANTES (aprox): {summary['boxes_total_before']}")
    print(f"Caixas totais DEPOIS: {summary['boxes_total_after']}")
    print(f"Arquivos corrigidos gravados em 'labels_fixed' (backup dos originais em 'labels_backup') por subset.")

if __name__ == "__main__":
    process_dataset_labels(DATASET_DIR)
    print("\nConcluído. Verifique as pastas labels_fixed e labels_backup em cada subset.")
