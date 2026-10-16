"""Figuras da atividade de SAC (Partes 3, 4 e 5).

    python ferramentas/wandb_hist.py sac     # primeiro: atualiza o cache
    python ferramentas/figuras_sac.py        # depois: gera os SVGs

Duas diferencas em relacao as atividades anteriores. A Parte 3 pede 3 seeds por
config (e nao 2), entao cada config vira um trio cheia/tracejada/pontilhada no
mesmo matiz. E as metricas de crítico da ablacao de um Q so estouram duas ordens
de grandeza acima do normal, o que exige escala simlog (`ysym`) para caber o
config final e a ablacao no mesmo eixo.
"""
import math, pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from grafico import RAMP2, RAMP3, chart, listar, load, num

D, find = load('sac')

RET = 'charts/episodic_return_mean_last100'
ALPHA, ENT = 'losses/alpha', 'losses/entropy'
QMR, QV = 'charts/q_minus_return', 'losses/qf1_values'

K = 100000                                      # orcamento de passos de ambiente
S3 = 'seeds 1 (cheia), 2 (tracejada) e 3 (pontilhada)'
S2 = 'seeds 1 (cheia) e 2 (tracejada)'
PONT = '1.5 3'
RET_AX = dict(ydom=(0, 1900), yticks=[0, 500, 1000, 1500], xmax=K, endlabels=True,
              xlab='passo de ambiente')


def serie(exp, rotulo, cor, seeds=(1, 2, 3), rotulo_em=1):
    """Um config = ate 3 seeds no mesmo matiz, estilos de linha diferentes.

    rotulo_em: qual seed carrega o rotulo de fim de linha. O padrao e a seed 1
    (cheia); a ablacao do alpha aponta para a 2 (tracejada), que e onde o efeito
    aparece -- quem le segue o rotulo, entao ele tem que estar na linha certa.
    """
    estilo = {1: False, 2: True, 3: PONT}
    out = []
    for sd in seeds:
        r = find(exp=exp, seed=sd)
        if r is not None:
            out.append((rotulo if sd == rotulo_em else '', cor, r, estilo[sd]))
    return out


def so1(exp, rotulo, cor):
    r = find(exp=exp, seed=1)
    return [(rotulo, cor, r, False)] if r is not None else []


def por_tempo(series, key):
    """Troca o eixo x: passo de ambiente -> segundos de relogio (`_runtime`)."""
    out = []
    for rotulo, cor, rec, tracejada in series:
        if rec is None:
            continue
        rt = dict(rec['series'].get('_runtime', []))
        pts = [(rt[s], v) for s, v in rec['series'].get(key, []) if s in rt]
        out.append((rotulo, cor, dict(rec, series={key: pts}), tracejada))
    return out


# =============================================== Parte 3: eficiencia amostral
inicial_final = (serie('sac_hopper', 'inicial (ups=1)', RAMP2[0])
                 + serie('sac_h_m4', 'final (ups=4)', RAMP2[1]))
sub3 = f'Hopper-v5 · {S3} · ups = updates_per_step'

chart('sac_p3_final.svg', 'Parte 3 — Retorno: config inicial × config final', sub3,
      RET, inicial_final, 'retorno médio (últimos 100 ep.)',
      refline=(1233, 'retorno final do inicial'), **RET_AX)

# ====================================================== Parte 4, ablacao 1: alpha
# O rotulo (e o destaque) vai na seed 2: e nela que a entropia sobe a -1,81 e o
# retorno trava em 841. Na seed 1 a ablacao quase nao doi.
abl1 = (serie('sac_h_m4', 'final (autotune)', RAMP2[0], seeds=(1, 2), rotulo_em=2)
        + serie('sac_abl_alpha', 'α fixo = 0,1', RAMP2[1], seeds=(1, 2), rotulo_em=2))
sub41 = 'Hopper-v5 · seeds 1 (clara) e 2 (tracejada, em destaque) · autotune ligado × desligado'

chart('sac_p4a_alpha.svg', 'Ablação 1 — A temperatura que o autotune escolhe', sub41,
      ALPHA, abl1, 'losses/alpha', xmax=K, endlabels=True, dashlabels=True, ypos=True,
      xlab='passo de ambiente')   # sem refline: a propria serie laranja e o valor fixado
# O rotulo da refline fica ilegivel aqui: a serie do autotune passa exatamente
# em cima dela. Vai no subtitulo. O piso -6,2 e para nao cortar o mergulho
# inicial do alpha fixo, que chega a -5,6.
chart('sac_p4a_entropia.svg', 'Ablação 1 — Entropia: o autotune segura no alvo',
      f'{sub41} · cinza = alvo −3',
      ENT, abl1, 'losses/entropy', xmax=K, endlabels=True, dashlabels=True, smooth=9, xlab='passo de ambiente',
      ydom=(-6.2, -1.4), yticks=[-6, -5, -4, -3, -2], refline=(-3.0, ''))
chart('sac_p4a_retorno.svg', 'Ablação 1 — Retorno: o custo aparece depois de 60k', sub41,
      RET, abl1, 'retorno médio (últimos 100 ep.)', dashlabels=True, **RET_AX)

# ================================================ Parte 4, ablacao 2: um critico
abl2 = (serie('sac_h_m4', 'final (dois críticos)', RAMP2[0], seeds=(1, 2))
        + serie('sac_abl_1q', 'um crítico só', RAMP2[1], seeds=(1, 2)))
sub42 = f'Hopper-v5 · {S2} · escala simlog: linear até ±100, log além'

chart('sac_p4b_qminus.svg', 'Ablação 2 — Erro do crítico (Q − retorno): superestimação realimentada', sub42,
      QMR, abl2, 'charts/q_minus_return', ysym=100, xmax=K, endlabels=True, smooth=9,
      ydom=(-200, 8000), yticks=[-100, 0, 100, 1000, 5000], xlab='passo de ambiente')
chart('sac_p4b_qvals.svg', 'Ablação 2 — Nível do crítico: o Q médio do batch', sub42,
      QV, abl2, 'losses/qf1_values', ysym=300, xmax=K, endlabels=True, smooth=9,
      ydom=(0, 8000), yticks=[0, 300, 1000, 5000], xlab='passo de ambiente')
chart('sac_p4b_retorno.svg', 'Ablação 2 — Retorno: não diverge, simplesmente não aprende',
      f'Hopper-v5 · {S2}', RET, abl2, 'retorno médio (últimos 100 ep.)', **RET_AX)

# ================================================== Parte 5: SAC contra o PPO
versus = (serie('sac_final_solo', 'SAC (ups=4)', RAMP3[0], seeds=(1, 2))
          + serie('ppo_cont_solo', 'PPO contínuo', RAMP3[1], seeds=(1, 2)))
sub5 = f'Hopper-v5 · {S2} · runs sozinhos na máquina · ups = updates_per_step'

chart('sac_p5_amostral.svg', 'Parte 5 — Eficiência amostral: retorno × passos de ambiente',
      sub5, RET, versus, 'retorno médio (últimos 100 ep.)', **RET_AX)

tempo = versus + serie('sac_ups1_solo', 'SAC (ups=1)', RAMP3[2], seeds=(1,))
chart('sac_p5_tempo.svg', 'Parte 5 — Tempo de relógio: retorno × segundos',
      'Hopper-v5 · mesmos 100 mil passos para os três · cada curva acaba com o run'
      ' · ups = updates_per_step',
      RET, por_tempo(tempo, RET), 'retorno médio (últimos 100 ep.)',
      ydom=(0, 1900), yticks=[0, 500, 1000, 1500], endlabels=True,
      xmax=1250, xlab='tempo de relógio (s)', xfmt=lambda s: f'{int(s)} s')

listar()
