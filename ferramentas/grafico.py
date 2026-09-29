"""Motor de graficos em SVG puro (sem matplotlib).

Compartilhado pelas atividades: figuras_dqn.py, figuras_a2c.py.
Convencoes: paleta categorica validada para daltonismo, linha cheia = seed 1,
tracejada = seed 2, rotulo direto na ponta da linha, escala log onde a metrica
varia ordens de grandeza.
"""
import json, math, os, pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent   # raiz do repo
OUT = str(ROOT / 'figuras')
os.makedirs(OUT, exist_ok=True)

SURFACE, GRID, INK, INK2, RULE = '#fcfcfb', '#e8e7e4', '#0b0b0b', '#52514e', '#9a9992'
RAMP5 = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4']  # categorica, slots 1-5
RAMP3 = ['#2a78d6', '#eb6834', '#1baf7a']                        # categorica, slots 1-3
RAMP2 = ['#2a78d6', '#eb6834']
FONT = 'Helvetica, Arial, sans-serif'
W, H = 660, 390
ML, MR, MT, MB = 68, 132, 78, 46


def esc(s):
    return s.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')


def num(v):
    """Formata numero com virgula decimal (pt-BR)."""
    return f'{v:g}'.replace('.', ',')


def milhar(v):
    """19540 -> 19,5k ; 500 -> 500"""
    if abs(v) >= 1000:
        return num(round(v / 1000, 1)) + 'k'
    return num(v)


def load(algo):
    """Carrega o cache de um algoritmo. Devolve (dados, find)."""
    path = ROOT / f'ferramentas/cache/hist_{algo}.json'
    D = json.load(open(path))

    def find(**kw):
        for r in D:
            if all(r.get(k) == v for k, v in kw.items()):
                return r
        return None

    return D, find


def _vals(series, key):
    out = []
    for item in series:
        rec = item[2]
        if rec is None:
            continue
        for _s, v in rec['series'].get(key, []):
            if v is None or (isinstance(v, float) and math.isnan(v)):
                continue
            out.append(v)
    return out


def _nice_step(raw):
    mag = 10 ** math.floor(math.log10(raw)) if raw > 0 else 1
    for m in (1, 2, 2.5, 5, 10):
        if raw <= m * mag:
            return m * mag
    return 10 * mag


def autodom(series, key, log=False, clip=None, pad=0.06, zero=False):
    """Dominio e ticks calculados a partir dos dados.

    clip: par de percentis (ex. (1, 99)) para series muito ruidosas, de modo
    que um unico outlier nao achate o resto do grafico.
    zero: forca o zero dentro do dominio (util quando o sinal importa).
    """
    vals = sorted(v for v in _vals(series, key) if not log or v > 0)
    if not vals:
        return (0, 1), [0, 1]
    if clip:
        lo_i = int(len(vals) * clip[0] / 100)
        hi_i = max(lo_i, int(len(vals) * clip[1] / 100) - 1)
        lo, hi = vals[lo_i], vals[hi_i]
    else:
        lo, hi = vals[0], vals[-1]

    if log:
        lo = 10 ** math.floor(math.log10(lo))
        hi = 10 ** math.ceil(math.log10(hi))
        e0, e1 = int(round(math.log10(lo))), int(round(math.log10(hi)))
        passo = max(1, math.ceil((e1 - e0) / 5))
        return (lo, hi), [10.0 ** e for e in range(e0, e1 + 1, passo)]

    if zero:
        lo, hi = min(lo, 0.0), max(hi, 0.0)
    span = (hi - lo) or (abs(hi) or 1)
    lo, hi = lo - span * pad, hi + span * pad
    passo = _nice_step((hi - lo) / 5)
    t0 = math.floor(lo / passo) * passo
    ticks, v = [], t0
    while v <= hi + passo * 1e-9:
        ticks.append(round(v, 10))
        v += passo
    return (ticks[0], ticks[-1]), ticks


def simlog(t):
    """Escala simetrica-logaritmica: linear perto de zero, log alem de `t`.

    Serve para metricas que ficam quase sempre perto de zero mas dao picos
    raros de ordens de grandeza -- o policy_loss com poucos ambientes, por
    exemplo. Uma escala linear esconde a estrutura perto de zero; uma log nao
    aceita valores negativos nem o proprio zero.
    """
    f = lambda v: math.copysign(math.log10(1 + abs(v) / t), v)

    def ticks(lo, hi):
        out, e = [0.0], 0
        while True:
            v = t * 10 ** e
            if v > max(abs(lo), abs(hi)):
                break
            out += [v, -v]
            e += 1
        return sorted(v for v in out if lo <= v <= hi)

    return f, ticks


def _rolling(pts, win):
    """Envelope movel: (x, media-dp, media, media+dp) por janela de `win` pontos.

    Desvio-padrao e nao min-max: uma metrica com picos raros (o policy_loss com
    poucos ambientes tem varios) faria o envelope min-max virar um borrao que
    esconde justamente a comparacao que interessa.
    """
    out = []
    for i in range(len(pts)):
        a, b = max(0, i - win // 2), min(len(pts), i + win // 2 + 1)
        jan = [v for _s, v in pts[a:b]]
        m = sum(jan) / len(jan)
        dp = (sum((v - m) ** 2 for v in jan) / len(jan)) ** 0.5
        out.append((pts[i][0], m - dp, m, m + dp))
    return out


def chart(path, title, subtitle, key, series, ylab, ylog=False, ydom=None,
          yticks=None, yfmt=num, refline=None, endlabels=False, band=0,
          clip=None, zero=False, ysym=None, smooth=0, ypos=False, xmax=500000):
    """Escreve um SVG em figuras/.

    series: lista de (rotulo, cor, registro, tracejada).
    band: se > 0, desenha o envelope movel (min-max) dessa largura de janela
          em vez da linha crua -- serve para mostrar o *ruido* de uma metrica.
    smooth: se > 0, media movel dessa largura. Metricas logadas por atualizacao
          (value_loss, explained_variance, advantage_std) sao ruidosas demais
          para o traco cru; e o mesmo alisamento que o W&B aplica na interface.
    ypos: impede o dominio de descer abaixo de zero (metricas nao-negativas).
    """
    if smooth:
        suav = []
        for rotulo, cor, rec, tracejada in series:
            if rec is not None and rec['series'].get(key):
                pts = rec['series'][key]
                mm = []
                for i in range(len(pts)):
                    a, b = max(0, i - smooth // 2), min(len(pts), i + smooth // 2 + 1)
                    jan = [v for _s, v in pts[a:b]]
                    mm.append((pts[i][0], sum(jan) / len(jan)))
                rec = dict(rec, series=dict(rec['series'], **{key: mm}))
            suav.append((rotulo, cor, rec, tracejada))
        series = suav
    if ysym:
        vals = _vals(series, key)
        lim = max(abs(v) for v in vals) if vals else 1.0
        ydom = ydom or (-lim, lim)
        _f, _tk = simlog(ysym)
        yticks = yticks or _tk(*ydom)
    elif ydom is None:
        ydom, auto_ticks = autodom(series, key, log=ylog, clip=clip, zero=zero)
        if ypos and ydom[0] < 0:
            ydom = (0, ydom[1])
            auto_ticks = [t for t in auto_ticks if t >= 0]
        yticks = yticks or auto_ticks
    x0, x1 = ML, W - MR

    # layout da legenda primeiro: define onde a area de plot comeca
    leg, lx, ly = [], ML, 55
    for label, color, _, _ in series:
        w = 21 + 6.2 * len(label)
        if lx + w > W - 16:
            lx, ly = ML, ly + 16
        leg.append((lx, ly, label, color))
        lx += w + 16
    y0, y1 = ly + 24, H - MB

    if ysym:
        lo, hi = _f(ydom[0]), _f(ydom[1])
        sy = lambda v: y1 - (_f(min(max(v, ydom[0]), ydom[1])) - lo) / (hi - lo) * (y1 - y0)
    elif ylog:
        lo, hi = math.log10(ydom[0]), math.log10(ydom[1])
        sy = lambda v: y1 - (math.log10(min(max(v, ydom[0]), ydom[1])) - lo) / (hi - lo) * (y1 - y0)
    else:
        lo, hi = ydom
        sy = lambda v: y1 - (min(max(v, lo), hi) - lo) / (hi - lo) * (y1 - y0)
    sx = lambda s: x0 + min(s, xmax) / xmax * (x1 - x0)

    p = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" font-family="{FONT}">',
         f'<rect width="{W}" height="{H}" fill="{SURFACE}"/>',
         f'<text x="{ML}" y="26" font-size="15" font-weight="600" fill="{INK}">{esc(title)}</text>',
         f'<text x="{ML}" y="44" font-size="11.5" fill="{INK2}">{esc(subtitle)}</text>']

    # grid + eixo y
    for v in yticks:
        y = sy(v)
        p.append(f'<line x1="{x0}" y1="{y:.1f}" x2="{x1}" y2="{y:.1f}" stroke="{GRID}" stroke-width="1"/>')
        p.append(f'<text x="{x0-9}" y="{y+3.5:.1f}" font-size="10.5" fill="{INK2}" text-anchor="end">{esc(yfmt(v))}</text>')
    # eixo x
    for s in range(0, xmax + 1, xmax // 5):
        x = sx(s)
        p.append(f'<line x1="{x:.1f}" y1="{y1}" x2="{x:.1f}" y2="{y1+4}" stroke="{RULE}" stroke-width="1"/>')
        p.append(f'<text x="{x:.1f}" y="{y1+18}" font-size="10.5" fill="{INK2}" text-anchor="middle">{s//1000}k</text>')
    p.append(f'<line x1="{x0}" y1="{y1}" x2="{x1}" y2="{y1}" stroke="{RULE}" stroke-width="1"/>')
    p.append(f'<text x="{x0}" y="{H-8}" font-size="10.5" fill="{INK2}">passo de treino</text>')
    p.append(f'<text x="14" y="{(y0+y1)/2:.1f}" font-size="10.5" fill="{INK2}" text-anchor="middle" '
             f'transform="rotate(-90 14 {(y0+y1)/2:.1f})">{esc(ylab)}</text>')

    if refline is not None:
        v, lbl = refline
        y = sy(v)
        p.append(f'<line x1="{x0}" y1="{y:.1f}" x2="{x1}" y2="{y:.1f}" stroke="{RULE}" stroke-width="1.5" stroke-dasharray="6 4"/>')
        p.append(f'<text x="{x0+6}" y="{y-7:.1f}" font-size="10" fill="{INK2}">{esc(lbl)}</text>')

    # legenda
    for lx, ly_, label, color in leg:
        p.append(f'<rect x="{lx}" y="{ly_}" width="16" height="3" rx="1.5" fill="{color}"/>')
        p.append(f'<text x="{lx+21}" y="{ly_+6.5}" font-size="11" fill="{INK2}">{esc(label)}</text>')

    # series
    ends = []
    for label, color, rec, dashed in series:
        if rec is None:
            continue
        pts = [(s, v) for s, v in rec['series'].get(key, [])
               if v is not None and not (isinstance(v, float) and math.isnan(v))]
        if not pts:
            continue
        dash = ' stroke-dasharray="5 4"' if dashed else ''
        if band:
            env = _rolling(pts, band)
            topo = ' L'.join(f'{sx(s):.1f},{sy(mx):.1f}' for s, _mn, _md, mx in env)
            base = ' L'.join(f'{sx(s):.1f},{sy(mn):.1f}' for s, mn, _md, _mx in reversed(env))
            p.append(f'<path d="M{topo} L{base} Z" fill="{color}" opacity="{0.10 if dashed else 0.17}" stroke="none"/>')
            linha = [(s, md) for s, _mn, md, _mx in env]
        else:
            linha = pts
        d = 'M' + ' L'.join(f'{sx(s):.1f},{sy(v):.1f}' for s, v in linha)
        p.append(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="2" '
                 f'stroke-linejoin="round" stroke-linecap="round"{dash} opacity="{0.75 if dashed else 1}"/>')
        if endlabels and not dashed:
            s, v = linha[-1]
            ends.append([sy(v), label, color, sx(s)])

    ends.sort(key=lambda e: e[0])
    for i in range(1, len(ends)):                      # separa rotulos colados
        if ends[i][0] - ends[i-1][0] < 14:
            ends[i][0] = ends[i-1][0] + 14
    for yy, label, color, xx in ends:
        p.append(f'<text x="{xx+8:.1f}" y="{yy+3.5:.1f}" font-size="10.5" font-weight="600" fill="{color}">{esc(label)}</text>')
    p.append('</svg>')
    open(f'{OUT}/{path}', 'w').write('\n'.join(p))
    return path


def listar():
    print('gerados em figuras/:')
    for f in sorted(os.listdir(OUT)):
        print(f'  {f:24s} {os.path.getsize(OUT + "/" + f)/1024:.1f} KB')
