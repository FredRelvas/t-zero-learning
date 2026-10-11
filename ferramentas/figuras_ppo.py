"""Figuras da atividade de PPO (Partes 4 e 5).

    python ferramentas/wandb_hist.py ppo     # primeiro: atualiza o cache
    python ferramentas/figuras_ppo.py        # depois: gera os SVGs

Diferente das atividades anteriores, aqui as series nao sao os valores de um
hiperparametro e sim os *passos da busca*: cada `exp_name` e uma mudanca
acumulada em relacao ao config inicial. A Parte 5 compara PPO com A2C, que
vivem no mesmo projeto do W&B.
"""
import math, pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from grafico import RAMP2, RAMP3, RAMP5, chart, listar, load, milhar, num

D, find = load('ppo')

RET = 'charts/episodic_return_mean_last100'
LEN = 'charts/episodic_length_mean_last100'
KL, CF = 'losses/approx_kl', 'losses/clipfrac'
EV, ENT, VL = 'losses/explained_variance', 'losses/entropy', 'losses/value_loss'
NU = 'charts/num_updates'

MI = 1000000                                    # orcamento de passos de ambiente
SEEDS = 'seed 1 (linha cheia) e seed 2 (tracejada)'
S1 = 'apenas seed 1, para não sobrepor traços'
RET_AX = dict(ydom=(-250, 300), yticks=[-200, -100, 0, 100, 200, 300],
              refline=(200, 'critério: 200'), xmax=MI, endlabels=True)
LN4 = math.log(4)


def par(exp, rotulo, cor):
    """Par (seed 1 cheia, seed 2 tracejada) de um mesmo exp_name."""
    out = []
    for seed, tracejada in ((1, False), (2, True)):
        r = find(exp=exp, seed=seed)
        if r is not None:
            out.append((rotulo if not tracejada else f'{rotulo} (seed 2)', cor, r, tracejada))
    return out


def so1(exp, rotulo, cor):
    r = find(exp=exp, seed=1)
    return [(rotulo, cor, r, False)] if r is not None else []


def por_tempo(serie, key):
    """Troca o eixo x da serie: passo de ambiente -> segundos de relogio.

    O W&B guarda `_runtime` no mesmo passo de log das metricas, entao basta
    reindexar. Sem isso nao da para responder "e se o orcamento fosse o mesmo
    tempo de relogio?", que a Parte 5 pede.
    """
    out = []
    for rotulo, cor, rec, tracejada in serie:
        if rec is None:
            continue
        rt = dict(rec['series'].get('_runtime', []))
        pts = [(rt[s], v) for s, v in rec['series'].get(key, []) if s in rt]
        out.append((rotulo, cor, dict(rec, series={key: pts}), tracejada))
    return out


# ===================================================== Parte 4: resolver o env
inicial_final = (par('ppo_lunarlander', 'config inicial', RAMP2[0])
                 + par('ppo_lunarlander_final', 'config final', RAMP2[1]))
sub_if = f'LunarLander-v3 · {SEEDS} · 1M passos de ambiente (orçamento: 2M)'

chart('ppo_p4_retorno.svg', 'Parte 4 — Retorno: config inicial × config final', sub_if,
      RET, inicial_final, 'retorno médio (últimos 100 ep.)', **RET_AX)
chart('ppo_p4_eplen.svg', 'Parte 4 — Comprimento do episódio: pairar × pousar', sub_if,
      LEN, inicial_final, 'passos por episódio (últimos 100)', xmax=MI, endlabels=True,
      refline=(1000, 'truncamento do ambiente'), ydom=(0, 1150), yticks=[0, 250, 500, 750, 1000])

caminho = (so1('ppo_lunarlander', 'inicial', RAMP5[0]) + so1('ppo_ll_m1', 'M1 horizonte', RAMP5[1])
           + so1('ppo_ll_m2', 'M2 gamma', RAMP5[2]) + so1('ppo_ll_m3a', 'M3a lr', RAMP5[3])
           + so1('ppo_ll_m4a', 'M4a ent_coef', RAMP5[4]))
chart('ppo_p4_caminho.svg', 'Parte 4 — As quatro mudanças, acumuladas', f'LunarLander-v3 · {S1}',
      RET, caminho, 'retorno médio (últimos 100 ep.)', **RET_AX)

diag = (so1('ppo_lunarlander', 'inicial', RAMP5[0]) + so1('ppo_ll_m1', 'M1 horizonte', RAMP5[1])
        + so1('ppo_ll_m2', 'M2 gamma', RAMP5[2]) + so1('ppo_ll_m3a', 'M3a lr', RAMP5[3]))
sub_d = f'LunarLander-v3 · {S1} · média móvel de 21 pontos'
chart('ppo_p4_kl.svg', 'Parte 4 — approx_kl: quanto a política se move por rollout', sub_d,
      KL, diag, 'losses/approx_kl', smooth=21, xmax=MI, ypos=True)
chart('ppo_p4_clipfrac.svg', 'Parte 4 — clipfrac: fração de amostras clipadas', sub_d,
      CF, diag, 'losses/clipfrac', smooth=21, xmax=MI, ypos=True)
chart('ppo_p4_explvar.svg', 'Parte 4 — Variância explicada: efeito do horizonte do GAE',
      f'LunarLander-v3 · {SEEDS} · média móvel de 11 pontos',
      EV, par('ppo_lunarlander', 'inicial (num_steps=128)', RAMP2[0])
      + par('ppo_ll_m1', 'M1 (num_steps=1024)', RAMP2[1]),
      'losses/explained_variance', ydom=(-0.6, 1.05), yticks=[-0.5, -0.25, 0, 0.25, 0.5, 0.75, 1],
      smooth=11,
      refline=(0, 'crítico inútil'), xmax=MI, endlabels=True)

entropia = (par('ppo_ll_m2', 'M2: ent_coef=0,01', RAMP3[0])
            + par('ppo_ll_m4a', 'M4a: ent_coef=0,001', RAMP3[1]))
chart('ppo_p4_entropia.svg', 'Parte 4 — Entropia: o que fazia o agente pairar',
      f'LunarLander-v3 · {SEEDS}', ENT, entropia, 'losses/entropy',
      ydom=(0.3, 1.52), yticks=[0.4, 0.6, 0.8, 1.0, 1.2, 1.4], smooth=9,
      refline=(LN4, 'política uniforme: ln 4 ≈ 1,39'), xmax=MI, endlabels=True)

chart('ppo_p4_falhou.svg', 'Parte 4 — O ramo que não deu certo: update_epochs',
      f'LunarLander-v3 · {SEEDS} · ambos partem do M2',
      RET, par('ppo_ll_m4a', 'M4a lr=1e-3', RAMP2[0]) + par('ppo_ll_m4b', 'M4b epochs=10', RAMP2[1]),
      'retorno médio (últimos 100 ep.)', **RET_AX)

# ================================================== Parte 5: PPO contra o A2C
versus = (par('ppo_final_solo', 'PPO (config final)', RAMP2[0])
          + par('a2c_base_solo', 'A2C', RAMP2[1]))
sub_v = f'LunarLander-v3 · {SEEDS} · 1M passos · runs medidos sozinhos na máquina'

chart('ppo_p5_amostral.svg', 'Parte 5 — Eficiência amostral: retorno × passos de ambiente',
      sub_v, RET, versus, 'retorno médio (últimos 100 ep.)', **RET_AX)

if versus:
    tempos = [s for _r, _c, rec, _d in versus for s, _v in rec['series'].get('_runtime', [])
              for s in [_v]]
    xt = int(math.ceil(max(tempos) / 50) * 50) if tempos else 250
    chart('ppo_p5_tempo.svg', 'Parte 5 — Tempo de relógio: retorno × segundos', sub_v,
          RET, por_tempo(versus, RET), 'retorno médio (últimos 100 ep.)',
          ydom=(-250, 300), yticks=[-200, -100, 0, 100, 200, 300], refline=(200, 'critério: 200'),
          endlabels=True, xmax=xt, xlab='tempo de relógio (s)', xfmt=lambda s: f'{s} s')

a2cs = (par('a2c_ll_base', 'A2C como está', RAMP3[0])
        + par('a2c_ll_g999', 'A2C + gamma=0,999', RAMP3[1])
        + par('a2c_ll_tuned', 'A2C + lote 512', RAMP3[2]))
chart('ppo_p5_a2c.svg', 'Parte 5 — As três tentativas de ajustar o A2C',
      f'LunarLander-v3 · {SEEDS} · o que funciona no PPO não transfere',
      RET, a2cs, 'retorno médio (últimos 100 ep.)', **RET_AX)

listar()
