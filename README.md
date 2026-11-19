<p align="center">
  <img width="100" height="100" alt="Ícone do projeto" src="./assets/icon.png">
</p>

<h1 align="center">Árvores Binárias como Reservatório Quântico usando Quantum Walks</h1>

<p align="center">
  <img alt="Disciplina" src="https://img.shields.io/badge/IMD0039-Estruturas%20de%20Dados%20B%C3%A1sicas%20II-0b5fa5?style=for-the-badge&labelColor=000">
  <img alt="UFRN" src="https://img.shields.io/badge/UFRN-IMD-0b5fa5?style=for-the-badge&labelColor=000">
  <img alt="Python" src="https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&labelColor=000&logo=python&logoColor=fff">
</p>

Código e resultados do projeto **"Árvores Binárias Completas como Reservatório Quântico de Quantum Walks"**, desenvolvido na disciplina **IMD0039 – Estruturas de Dados Básicas II** (Instituto Metrópole Digital, UFRN).

<br />

## 📖 Sobre

Em _quantum reservoir computing_ (QRC), apenas uma leitura linear é treinada sobre um sistema quântico fixo. Aqui, um **quantum walk de tempo contínuo (CTQW)** sobre uma **árvore binária completa** faz o papel de reservatório, e a árvore é armazenada de forma implícita, com **indexação de heap**:

| Operação              | Fórmula                                 |
| --------------------- | --------------------------------------- |
| filhos de `v`         | `2v + 1` e `2v + 2`                     |
| pai de `v`            | `⌊(v − 1) / 2⌋`                         |
| nível de `v`          | `⌊log₂(v + 1)⌋`                         |
| vértices do nível `k` | intervalo contíguo `[2ᵏ − 1, 2ᵏ⁺¹ − 2]` |

### Contribuições

- **C1** – descrição da árvore binária como estrutura do reservatório;
- **C2** – duas proposições: os estados uniformes por nível formam um subespaço invariante de dimensão `d + 1`, e o espectro se decompõe em cadeias com autovalores distintos;
- **C3** – comparação com um ciclo e com uma ESN do mesmo tamanho, mais um benchmark de detecção de intrusão.

### Resultados em destaque

- Com pesos iguais por nível, uma árvore de 127 vértices produz só **7 atributos independentes** (`d + 1`).
- Com pesos aleatórios e `H = −A`, a mesma árvore atinge memória linear **11,09**, contra 8,96 do ciclo e 5,02 da ESN.
- No benchmark **CSE-CIC-IDS2018**, a melhor árvore (`n = 31`, amplitude, `H = L`, `τ = 1,5`) obtém ROC-AUC **0,943**, abaixo das ESNs de 100 e 500 unidades (0,946 e 0,951).

<br />

## 🗂️ Estrutura do repositório

```
.
├── assets/
│   └── icon.png                  # ícone do projeto (a adicionar)
├── binary_tree_ids_benchmark.py  # benchmark de intrusão (CSE-CIC-IDS2018)
└── qrc_binary_tree.py
```

<br />

## 🧪 Benchmark de detecção de intrusão

O script `binary_tree_ids_benchmark.py` avalia reservatórios CTQW em árvores binárias completas sobre janelas de 20 fluxos do CSE-CIC-IDS2018:

- **Profundidades:** 2 a 4 (7, 15 e 31 vértices);
- **Codificações:** fase, qsample, hamiltoniana, amplitude, base e multibase;
- **Hamiltonianos:** laplaciano (`L`) e adjacência (`−A`);
- **Tempos de evolução:** `τ ∈ {1,5; 3}`;
- **Referências:** `ESN-8`, `ESN-100`, `ESN-500`, `RFF-256`, `PCA-linear` e `XGBoost`;
- **Protocolo:** divisão cronológica por episódio (60/20/20), leitura _ridge_ sobre o estado final e a média dos estados da janela, 5 sementes por modelo.

### ⚠️ Pré-requisitos

O script depende da biblioteca **`qrckit`** (reservatórios de quantum walk em C++ com interface Python), que **ainda não foi publicada** e **não está neste repositório**. Ele também importa o módulo auxiliar `_results.py` do `qrckit`. Para executá-lo, copie `binary_tree_ids_benchmark.py` para `python/examples/` do `qrckit`, ao lado de `_results.py`.

O dataset **CSE-CIC-IDS2018** não é redistribuído aqui; ele está disponível no site do [Canadian Institute for Cybersecurity](https://www.unb.ca/cic/datasets/ids-2018.html).

### Executando

```bash
# plano de execução, sem rodar nada
python binary_tree_ids_benchmark.py --list

# benchmark completo (um processo por CPU; retoma tarefas já concluídas)
python binary_tree_ids_benchmark.py --csv /caminho/para/data.csv

# só consolidar as tabelas
python binary_tree_ids_benchmark.py --merge

# teste rápido com dados sintéticos
python binary_tree_ids_benchmark.py --synthetic --seeds 1 --depths 2 3 --max-train-windows 300
```

O benchmark também roda em arrays do **SLURM** (`SLURM_ARRAY_TASK_ID` e `SLURM_ARRAY_TASK_COUNT` são lidos automaticamente). A execução do projeto foi feita no supercomputador do IMD/UFRN, partição `intel-128`.

<br />

## 📊 Resultados completos

Os plots estão em [`plots/`](./plots).

<br />

## 👥 Autores (Grupo 3 — T05, 2026.2)

| Nome                                  | Contribuição principal                                                     |
| ------------------------------------- | -------------------------------------------------------------------------- |
| Aristofany Herderson G. de S. Menezes | implementação do `qrckit`, scripts de experimentos e execução do benchmark |
| Acirio Gabriel Ferreira Barbosa       | revisão da literatura e trabalhos relacionados                             |
| Gabriel Augusto Gomes Pinheiro        | formulação e demonstração das proposições                                  |
| Thaís Evelin Barbosa da Silva         | modelo do experimento de detecção e apresentação                           |
| Leonardo Rosendo Faustino             | introdução, resumo, conclusão e formatação em LaTeX                        |

**Orientação da disciplina:** Profa. Ligia Borges, IMD/UFRN.
