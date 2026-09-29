"""Figuras da atividade de DQN (entregue em 22/09/2026).

    python ferramentas/figuras_dqn.py
"""
import pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from grafico import RAMP3, RAMP5, chart, listar, load, num

D, find = load('dqn')

RET = 'charts/episodic_return_mean_last100'
QV, TD, EPS = 'losses/q_values', 'losses/td_loss', 'charts/epsilon'
pow10 = lambda v: ('%g' % v) if v >= 1 else f'{v:g}'

# ---- Q1: target_network_frequency ----
q1 = [(f'tnf = {t}', RAMP5[i], find(tnf=t, buf=10000, ef=0.5, seed=1), False)
      for i, t in enumerate([1, 50, 500, 5000, 50000])]
q1s2 = [(f'tnf = {t} (seed 2)', RAMP5[i], find(tnf=t, buf=10000, ef=0.5, seed=2), True)
        for i, t in [(0, 1), (4, 50000)]]
sub1 = 'CartPole-v1 · seed 1 (linha cheia) e seed 2 (tracejada) · baseline: tnf = 500'
chart('q1_retorno.svg', 'Q1 — Retorno episódico por frequência de sync da target network', sub1,
      RET, q1 + q1s2, 'retorno médio (últimos 100 ep.)', ydom=(0, 500),
      yticks=[0, 100, 200, 300, 400, 500], endlabels=True)
chart('q1_qvalues.svg', 'Q1 — Q médio predito (escala log)', sub1,
      QV, q1 + q1s2, 'losses/q_values', ylog=True, ydom=(0.05, 3000),
      yticks=[0.1, 1, 10, 100, 1000], yfmt=pow10,
      refline=(100, 'teto teórico 1/(1−γ) = 100'), endlabels=True)
chart('q1_tdloss.svg', 'Q1 — Perda de TD (escala log)', sub1,
      TD, q1 + q1s2, 'losses/td_loss', ylog=True, ydom=(0.0001, 50000),
      yticks=[0.0001, 0.01, 1, 100, 10000], yfmt=pow10, endlabels=True)

# ---- Q2: buffer_size ----
q2 = [(f'buffer = {b:,}'.replace(',', '.'), RAMP5[i], find(buf=b, tnf=500, ef=0.5, seed=1), False)
      for i, b in enumerate([100, 200, 500, 10000, 100000])]
q2s2 = [('buffer = 100 (seed 2)', RAMP5[0], find(buf=100, tnf=500, ef=0.5, seed=2), True)]
sub2 = 'CartPole-v1 · seed 1 (linha cheia) e seed 2 (tracejada) · baseline: buffer = 10.000'
chart('q2_retorno.svg', 'Q2 — Retorno episódico por tamanho do replay buffer', sub2,
      RET, q2 + q2s2, 'retorno médio (últimos 100 ep.)', ydom=(0, 500),
      yticks=[0, 100, 200, 300, 400, 500], endlabels=True)
chart('q2_qvalues.svg', 'Q2 — Q médio predito por tamanho do replay buffer (escala log)', sub2,
      QV, q2 + q2s2, 'losses/q_values', ylog=True, ydom=(0.05, 3000),
      yticks=[0.1, 1, 10, 100, 1000], yfmt=pow10,
      refline=(100, 'teto teórico 1/(1−γ) = 100'), endlabels=True)

# ---- Q3: exploration_fraction ----
q3 = [(f'expl. = {num(e)}', RAMP3[i], find(ef=e, buf=10000, tnf=500, seed=1), False)
      for i, e in enumerate([0.05, 0.5, 0.95])]
q3s2 = [('expl. = 0,95 (seed 2)', RAMP3[2], find(ef=0.95, buf=10000, tnf=500, seed=2), True)]
# o cronograma de epsilon e deterministico: a seed 2 sobreporia a seed 1, entao nao entra
sub3e = 'CartPole-v1 · cronograma determinístico — idêntico nas duas seeds'
sub3r = 'CartPole-v1 · seed 1 (linha cheia) e seed 2 (tracejada) · baseline: expl. = 0,5'
chart('q3_epsilon.svg', 'Q3 — Cronograma de exploração efetivamente usado', sub3e,
      EPS, q3, 'epsilon', ydom=(0, 1), yticks=[0, 0.2, 0.4, 0.6, 0.8, 1.0], endlabels=True)
chart('q3_retorno.svg', 'Q3 — Retorno da política de coleta (epsilon-greedy)', sub3r,
      RET, q3 + q3s2, 'retorno médio (últimos 100 ep.)', ydom=(0, 500),
      yticks=[0, 100, 200, 300, 400, 500], endlabels=True)

listar()
