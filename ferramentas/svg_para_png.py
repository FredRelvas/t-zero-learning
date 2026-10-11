"""Converte os SVGs de figuras/ em PNG e descarta os SVGs (macOS).

    python ferramentas/svg_para_png.py            # todos os SVGs de figuras/
    python ferramentas/svg_para_png.py ppo_p4     # so os que comecam com isso

Usa o `qlmanage` do macOS, que renderiza o SVG numa miniatura de canvas
quadrado; o corte devolve a proporcao 660x390 do original. Nao ha cairosvg
nem inkscape nesta maquina, e o Chrome headless so exporta PDF.
"""
import glob, os, pathlib, shutil, subprocess, sys, tempfile
from PIL import Image

ROOT = pathlib.Path(__file__).resolve().parent.parent
FIG = ROOT / 'figuras'
W, H = 660, 390                      # proporcao do SVG gerado por grafico.py

prefixo = sys.argv[1] if len(sys.argv) > 1 else ''
svgs = sorted(glob.glob(str(FIG / f'{prefixo}*.svg')))
if not svgs:
    sys.exit(f'nenhum SVG em figuras/ com prefixo {prefixo!r}')

tmp = tempfile.mkdtemp()
subprocess.run(['qlmanage', '-t', '-s', '1980', '-o', tmp] + svgs,
               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

n = 0
for f in sorted(glob.glob(tmp + '/*.svg.png')):
    im = Image.open(f)
    w, _ = im.size
    destino = FIG / os.path.basename(f).replace('.svg.png', '.png')
    im.crop((0, 0, w, round(w * H / W))).save(destino)
    n += 1
shutil.rmtree(tmp)

for s in svgs:
    os.remove(s)
print(f'{n} PNGs em figuras/ ({len(svgs)} SVGs removidos)')
