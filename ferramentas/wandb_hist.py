"""Baixa o historico de runs do W&B para um cache local.

    python ferramentas/wandb_hist.py a2c
    python ferramentas/wandb_hist.py dqn ENTITY/PROJETO

Grava ferramentas/cache/hist_<algo>.json. Todo o resto (grafico.py,
figuras_*.py) trabalha em cima desse cache, sem rede.
"""
import json, math
import pathlib, sys

ROOT = pathlib.Path(__file__).resolve().parent.parent   # raiz do repo
from dotenv import load_dotenv
load_dotenv(ROOT / '.env')
import wandb

COMUM = ['charts/episodic_return_mean_last100']

ALGOS = {
    'dqn': dict(
        projeto='fredrelvas-ceia/dqn-assignment',
        keys=COMUM + ['losses/td_loss', 'losses/q_values', 'charts/epsilon'],
        campos=lambda c: dict(
            tnf=c.get('dqn', {}).get('target_network_frequency'),
            buf=c.get('dqn', {}).get('buffer_size'),
            ef=c.get('dqn', {}).get('exploration_fraction'),
        ),
        ordem=lambda r: (r['ef'], r['buf'], r['tnf'], r['seed']),
        rotulo=lambda r: f"ef={r['ef']:<5} buf={r['buf']:<7} tnf={r['tnf']:<6} seed={r['seed']}",
    ),
    'a2c': dict(
        projeto='fredrelvas-ceia/a2c-assignment',
        keys=COMUM + ['losses/policy_loss', 'losses/value_loss', 'losses/entropy',
                      'losses/explained_variance', 'losses/grad_norm',
                      'charts/advantage_mean', 'charts/advantage_std', 'charts/SPS'],
        campos=lambda c: dict(
            envs=c.get('num_envs'),
            ns=c.get('a2c', {}).get('num_steps'),
            ent=c.get('a2c', {}).get('ent_coef'),
            base=c.get('a2c', {}).get('use_baseline'),
        ),
        ordem=lambda r: (not r['base'], r['ent'], r['ns'], r['envs'], r['seed']),
        rotulo=lambda r: (f"envs={r['envs']:<3} ns={r['ns']:<4} ent={r['ent']:<5} "
                          f"base={str(r['base']):<5} seed={r['seed']}"),
    ),
}

algo = sys.argv[1] if len(sys.argv) > 1 else 'a2c'
if algo not in ALGOS:
    sys.exit(f"algoritmo desconhecido: {algo} (use {' ou '.join(ALGOS)})")
spec = ALGOS[algo]
projeto = sys.argv[2] if len(sys.argv) > 2 else spec['projeto']

api = wandb.Api()
out = []
for r in api.runs(projeto):
    if r.state != 'finished':
        continue
    if r.config.get('algorithm') != algo:
        continue
    rec = dict(seed=r.config.get('seed'),
               eval_mean=r.summary.get('eval/mean_return'),
               eval_std=r.summary.get('eval/std_return'),
               url=r.url, series={})
    rec.update(spec['campos'](r.config))
    h = r.history(keys=spec['keys'], samples=2000, pandas=False)
    for k in spec['keys']:
        pts = [(x['_step'], x[k]) for x in h
               if x.get(k) is not None and not (isinstance(x[k], float) and math.isnan(x[k]))]
        rec['series'][k] = pts[::max(1, len(pts) // 600)]
    out.append(rec)

(ROOT / 'ferramentas/cache').mkdir(exist_ok=True)
destino = ROOT / f'ferramentas/cache/hist_{algo}.json'
json.dump(out, open(destino, 'w'))
print(f'{len(out)} runs em cache -> {destino.relative_to(ROOT)}')
for r in sorted(out, key=spec['ordem']):
    ev = r['eval_mean']
    print(f"  {spec['rotulo'](r)}  eval={ev if ev is None else round(ev)}")
