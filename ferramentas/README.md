# Ferramentas de relatório

Pipeline usado nas entregas da disciplina de RL. Reaproveitável: para uma atividade nova,
copie `figuras_a2c.py` e troque as definições de série.

Pré-requisitos: `.env` com `WANDB_ENTITY` / `WANDB_PROJECT` / `WANDB_API_KEY`, e Google Chrome
(usado em modo headless para gerar o PDF, porque não há pandoc nem LaTeX nesta máquina).

**Atenção:** se `WANDB_PROJECT` estiver *exportada* no shell, o `load_dotenv()` do `train.py`
não a sobrescreve e a run vai para o projeto errado. Rode `unset WANDB_PROJECT` antes.

## Arquivos

| Arquivo | Papel |
|---|---|
| `wandb_hist.py` | baixa o histórico do W&B → `cache/hist_<algo>.json` |
| `grafico.py` | motor de gráficos em SVG puro (sem matplotlib) — compartilhado |
| `figuras_dqn.py` | séries da atividade de DQN |
| `figuras_a2c.py` | séries da atividade de A2C |
| `figuras_ppo.py` | séries da atividade de PPO (Partes 4 e 5) |
| `svg_para_png.py` | SVG → PNG via `qlmanage`, e apaga os SVGs |
| `md_para_html.py` | Markdown → HTML com o CSS de impressão A4 |

## 1. Baixar o histórico para cache local

```bash
python ferramentas/wandb_hist.py a2c                  # projeto padrão do algoritmo
python ferramentas/wandb_hist.py dqn ENTITY/PROJETO   # outro projeto
python ferramentas/wandb_hist.py ppo                  # PPO *e* A2C, para a comparação
```

`dqn` e `a2c` filtram por `config.algorithm` e baixam uma lista fixa de métricas. O `ppo` é
diferente em dois pontos, porque a Parte 5 compara dois algoritmos dentro do mesmo projeto:
`todos_algos=True` desliga o filtro, e `keys=None` baixa o histórico inteiro — as runs de A2C não
logam `approx_kl` nem `clipfrac`, e pedir uma chave inexistente devolveria série vazia. O
`_runtime` vem junto, que é o que permite plotar retorno × tempo de relógio.

## 2. Gerar os gráficos

```bash
python ferramentas/figuras_a2c.py
```

Escreve SVGs em `figuras/`. Recursos do motor (`grafico.py`):

- `ylog` escala log · `ysym=t` escala **simlog**: linear perto de zero, log além de `t`
  (para métricas que ficam quase sempre em zero e dão picos raros, como o `policy_loss`
  com poucos ambientes)
- `smooth=N` média móvel — métricas logadas por atualização (`value_loss`,
  `explained_variance`, `advantage_std`) são ruidosas demais para o traço cru; é o mesmo
  alisamento que o W&B aplica na interface. Diga no subtítulo que ele foi aplicado.
- `band=N` faixa de ±1 desvio-padrão móvel · `clip=(p1,p2)` domínio por percentil
- `ypos` impede o domínio de descer abaixo de zero · `zero` força o zero no domínio
- domínio e ticks são calculados dos dados quando `ydom` é omitido

Convenções: paleta **categórica** (azul, laranja, verde-água, amarelo, magenta — matizes
distintos, validados para daltonismo), **linha cheia = seed 1** e **tracejada = seed 2** no
mesmo matiz, rótulo direto na ponta de cada linha.

## 3. Converter SVG em PNG (macOS) e descartar os SVGs

```bash
python ferramentas/svg_para_png.py          # todos os SVGs de figuras/
python ferramentas/svg_para_png.py ppo_     # só os de um prefixo
```

O `qlmanage` gera a miniatura em canvas quadrado; o script corta de volta para a proporção
660×390 do SVG e remove os `.svg`. Só os PNGs vão para o repositório.

## 4. Markdown → PDF

```bash
python ferramentas/md_para_html.py relatorio.md
"/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" \
  --headless --disable-gpu --no-pdf-header-footer \
  --print-to-pdf=relatorio.pdf --virtual-time-budget=6000 relatorio.html
```

`md_para_html.py` cobre o subconjunto de Markdown usado nos relatórios — títulos, parágrafos,
`código`, **negrito**, *itálico*, links, linhas de figuras e **tabelas de pipe** — e aplica o CSS
de impressão A4. Uma linha com 1, 2, 3 ou 4 figuras vira `.cols1`…`.cols4`: três por linha é o
padrão, quatro viram 2+2. Confira a contagem de páginas no PDF antes de entregar; dá para contar
sem abrir:

```bash
python -c "import re;d=open('relatorio_ppo.pdf','rb').read();print(len(re.findall(rb'/Type\s*/Page[^s]',d)))"
```
