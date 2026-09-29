"""Figuras da atividade de A2C.

    python ferramentas/wandb_hist.py a2c     # primeiro: atualiza o cache
    python ferramentas/figuras_a2c.py        # depois: gera os SVGs

Para adaptar a uma atividade nova, copie o bloco de uma questao e troque o
hiperparametro em `find(...)` e a metrica em `chart(...)`.
"""
import math, pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from grafico import RAMP2, RAMP3, RAMP5, chart, listar, load, milhar, num

D, find = load('a2c')

RET = 'charts/episodic_return_mean_last100'
PL, VL, ENT = 'losses/policy_loss', 'losses/value_loss', 'losses/entropy'
EV, AM, AS, SPS = 'losses/explained_variance', 'charts/advantage_mean', 'charts/advantage_std', 'charts/SPS'

BASE = dict(envs=8, ns=5, ent=0.01, base=True)          # a configuracao padrao
RET_AX = dict(ydom=(0, 500), yticks=[0, 100, 200, 300, 400, 500])
LN2 = math.log(2)
pow10 = lambda v: ('%g' % v) if v >= 1 else f'{v:g}'.replace('.', ',')
SEEDS = 'seed 1 (linha cheia) e seed 2 (tracejada)'


def serie(rotulo, cor, seed, **kw):
    """Um par (seed 1 cheia, seed 2 tracejada) para a mesma configuracao."""
    cfg = dict(BASE, **kw)
    out = [(rotulo, cor, find(seed=1, **cfg), False)]
    s2 = find(seed=2, **cfg)
    if s2 is not None:
        out.append((f'{rotulo} (seed 2)', cor, s2, True))
    return out


# ---------------------------------------------------------------- Q1: num_envs
q1 = [s for i, n in enumerate([1, 8, 32])
      for s in serie(f'{n} ambiente' + ('s' if n > 1 else ''), RAMP3[i], 1, envs=n)]
sub1 = f'CartPole-v1 · {SEEDS} · baseline: 8 ambientes · lote = num_envs × 5'
chart('a2c_q1_retorno.svg', 'Q1 — Retorno episódico por número de atores em paralelo', sub1,
      RET, q1, 'retorno médio (últimos 100 ep.)', endlabels=True, **RET_AX)
# so a seed 1: seis tracos sobrepostos escondem a comparacao que interessa
q1s1 = [s for s in q1 if not s[3]]
chart('a2c_q1_policyloss.svg', 'Q1 — Perda de política (escala simlog)',
      'CartPole-v1 · apenas seed 1 · desvio-padrão da série inteira: 4,45 (1 env) · '
      '1,37 (8) · 0,74 (32)',
      PL, q1s1, 'losses/policy_loss', ysym=0.05)
chart('a2c_q1_sps.svg', 'Q1 — Passos de ambiente por segundo', sub1,
      SPS, q1, 'charts/SPS', yfmt=milhar, endlabels=True)

# ------------------------------------------------------------- Q2: a2c.num_steps
q2 = [s for i, n in enumerate([1, 5, 32, 128])
      for s in serie(f'n = {n}', RAMP5[i], 1, ns=n)]
sub2 = f'CartPole-v1 · {SEEDS} · baseline: n = 5 · lote = 8 × n · média móvel de 21 pontos'
chart('a2c_q2_retorno.svg', 'Q2 — Retorno episódico por horizonte do retorno de n passos', sub2,
      RET, q2, 'retorno médio (últimos 100 ep.)', endlabels=True, **RET_AX)
chart('a2c_q2_valueloss.svg', 'Q2 — Perda do crítico (escala log)', sub2,
      VL, q2, 'losses/value_loss', ylog=True, yfmt=pow10, smooth=21, endlabels=True)
chart('a2c_q2_explvar.svg', 'Q2 — Variância explicada pelo crítico', sub2,
      EV, q2, 'losses/explained_variance', ydom=(-1, 1), smooth=21,
      yticks=[-1, -0.5, 0, 0.5, 1], refline=(0, 'crítico inútil'), endlabels=True)

# -------------------------------------------------------------- Q3: a2c.ent_coef
q3 = [s for i, e in enumerate([0, 0.01, 0.1])
      for s in serie(f'ent_coef = {num(e)}', RAMP3[i], 1, ent=e)]
sub3 = f'CartPole-v1 · {SEEDS} · baseline: ent_coef = 0,01'
chart('a2c_q3_retorno.svg', 'Q3 — Retorno episódico por coeficiente de entropia', sub3,
      RET, q3, 'retorno médio (últimos 100 ep.)', endlabels=True, **RET_AX)
chart('a2c_q3_entropia.svg', 'Q3 — Entropia da política', sub3,
      ENT, q3, 'losses/entropy', ydom=(0, 0.72), yticks=[0, 0.2, 0.4, 0.6],
      refline=(LN2, 'política uniforme: ln 2 ≈ 0,69'), endlabels=True)

# ----------------------------------------------------------- Q4: a2c.use_baseline
q4 = (serie('com baseline: R − V(s)', RAMP2[0], 1, base=True)
      + serie('sem baseline: R', RAMP2[1], 1, base=False))
sub4 = f'CartPole-v1 · {SEEDS} · peso do gradiente de política'
sub4m = sub4 + ' · média móvel de 21 pontos'
chart('a2c_q4_retorno.svg', 'Q4 — Retorno episódico com e sem baseline', sub4,
      RET, q4, 'retorno médio (últimos 100 ep.)', endlabels=True, **RET_AX)
chart('a2c_q4_advmean.svg', 'Q4 — Média dos pesos usados no gradiente (escala simlog)', sub4m,
      AM, q4, 'charts/advantage_mean', ysym=0.1, smooth=21, refline=(0, 'zero'), endlabels=True)
chart('a2c_q4_advstd.svg', 'Q4 — Desvio-padrão dos pesos usados no gradiente', sub4m,
      AS, q4, 'charts/advantage_std', smooth=21, ypos=True, endlabels=True)
chart('a2c_q4_entropia.svg', 'Q4 — Entropia da política com e sem baseline', sub4,
      ENT, q4, 'losses/entropy', ydom=(0, 0.72), yticks=[0, 0.2, 0.4, 0.6],
      refline=(LN2, 'política uniforme: ln 2 ≈ 0,69'), endlabels=True)

listar()
