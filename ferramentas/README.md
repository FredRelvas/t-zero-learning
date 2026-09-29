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
| `md_para_html.py` | Markdown → HTML com o CSS de impressão A4 |

## 1. Baixar o histórico para cache local

```bash
python ferramentas/wandb_hist.py a2c                  # projeto padrão do algoritmo
python ferramentas/wandb_hist.py dqn ENTITY/PROJETO   # outro projeto
```

Filtra por `config.algorithm`, então um projeto com runs misturadas não atrapalha.

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
T=$(mktemp -d)
qlmanage -t -s 1980 -o "$T" figuras/*.svg >/dev/null 2>&1
python - "$T" <<'PY'
from PIL import Image; import glob, os, sys
T = sys.argv[1]
for f in sorted(glob.glob(T + '/*.svg.png')):
    im = Image.open(f); w, _ = im.size
    im.crop((0, 0, w, round(w * 390 / 660))).save('figuras/' + os.path.basename(f).replace('.svg.png', '.png'))
PY
rm -f figuras/*.svg
```

O `qlmanage` gera a miniatura em canvas quadrado; o corte devolve a proporção 660×390 do SVG.

## 4. Markdown → PDF

```bash
python ferramentas/md_para_html.py relatorio.md
"/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" \
  --headless --disable-gpu --no-pdf-header-footer \
  --print-to-pdf=relatorio.pdf --virtual-time-budget=6000 relatorio.html
```

`md_para_html.py` cobre o subconjunto de Markdown usado no relatório e aplica o CSS de
impressão: A4, margens de 17-18 mm, corpo 10,5 pt, figuras em duas por linha (uma sozinha
na linha ocupa a largura toda). O limite de 3 páginas da atividade foi conferido assim.
