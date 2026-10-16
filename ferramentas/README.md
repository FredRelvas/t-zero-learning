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
| `figuras_sac.py` | séries da atividade de SAC (Partes 3, 4 e 5) |
| `svg_para_png.py` | SVG → PNG via `qlmanage`, e apaga os SVGs |
| `md_para_html.py` | Markdown → HTML com o CSS de impressão A4 |

## 1. Baixar o histórico para cache local

```bash
python ferramentas/wandb_hist.py a2c                  # projeto padrão do algoritmo
python ferramentas/wandb_hist.py dqn ENTITY/PROJETO   # outro projeto
python ferramentas/wandb_hist.py ppo                  # PPO *e* A2C, para a comparação
python ferramentas/wandb_hist.py sac                  # SAC *e* PPO contínuo, idem
```

`dqn` e `a2c` filtram por `config.algorithm` e baixam uma lista fixa de métricas. O `ppo` e o
`sac` são diferentes em dois pontos, porque a Parte 5 compara dois algoritmos dentro do mesmo projeto:
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
- o 4º item de cada série é o estilo da linha: `False` cheia, `True` tracejada, ou um padrão SVG
  próprio como `'1.5 3'` (pontilhada) — o SAC tem 3 seeds por config. Rótulo vazio (`''`) tira a
  entrada da legenda, para não repetir o mesmo config três vezes
- com `ysym`, **passe `ydom` e `yticks` explícitos**: o automático é simétrico (−lim, +lim) e
  desperdiça metade do gráfico quando os dados só vão para um lado
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

`md_para_html.py` cobre o subconjunto de Markdown usado nos relatórios — títulos (`#` a `###`), parágrafos,
`código`, **negrito**, *itálico*, links, linhas de figuras e **tabelas de pipe** — e aplica o CSS
de impressão A4. Uma linha com 1, 2, 3 ou 4 figuras vira `.cols1`…`.cols4`: três por linha é o
padrão, quatro viram 2+2; numa linha de 3, a figura órfã desce centralizada e **maior** (62%).
Numa tabela, células curtas que contenham número ganham `class="nw"` e não quebram — a regra antiga
era por posição de coluna (2ª e 3ª) e estourava a largura quando essas colunas eram texto longo. Confira a contagem de páginas no PDF antes de entregar; dá para contar
sem abrir:

```bash
python -c "import re;d=open('relatorio_ppo.pdf','rb').read();print(len(re.findall(rb'/Type\s*/Page[^s]',d)))"
```
