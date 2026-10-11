import re, sys, html
import pathlib, sys
ROOT = pathlib.Path(__file__).resolve().parent.parent   # raiz do repo

SRC = sys.argv[1] if len(sys.argv) > 1 else 'relatorio.md'
md = open(ROOT/SRC).read()
# um titulo colado no paragrafo anterior sairia como '## ...' literal;
# garante a linha em branco antes de qualquer '#' em inicio de linha
md = re.sub(r'(?m)([^\n])\n(#{1,6} )', r'\1\n\n\2', md)
blocks = [b for b in re.split(r'\n\s*\n', md) if b.strip()]

def inline(t):
    t = html.escape(t)
    t = re.sub(r'`([^`]+)`', r'<code>\1</code>', t)
    t = re.sub(r'\*\*([^*]+)\*\*', r'<strong>\1</strong>', t)
    t = re.sub(r'(?<!\*)\*([^*]+)\*(?!\*)', r'<em>\1</em>', t)
    t = re.sub(r'(https?://[^\s<]+)', r'<a href="\1">\1</a>', t)
    return t

out = []
for b in blocks:
    b = b.strip()
    if b.startswith('# '):
        out.append(f'<h1>{inline(b[2:])}</h1>')
    elif b.startswith('## '):
        out.append(f'<h2>{inline(b[3:])}</h2>')
    elif b.startswith('|'):
        linhas = [l.strip() for l in b.split(chr(10)) if l.strip().startswith('|')]
        celulas = [[c.strip() for c in l.strip('|').split('|')] for l in linhas]
        # a segunda linha e o separador (|---|---|), nao e conteudo
        corpo = [c for c in celulas[1:] if not all(set(x) <= set('-: ') for x in c)]
        cab = ''.join(f'<th>{inline(c)}</th>' for c in celulas[0])
        tr = ''.join('<tr>' + ''.join(f'<td>{inline(c)}</td>' for c in linha) + '</tr>'
                     for linha in corpo)
        out.append(f'<table><thead><tr>{cab}</tr></thead><tbody>{tr}</tbody></table>')
    elif b.startswith('!['):
        imgs = re.findall(r'!\[([^\]]*)\]\(([^)]+)\)', b)
        n = len(imgs)
        cells = ''.join(f'<figure><img src="{s}" alt="{html.escape(a)}"></figure>' for a, s in imgs)
        out.append(f'<div class="figs cols{n}">{cells}</div>')
    else:
        out.append(f'<p>{inline(" ".join(l.strip() for l in b.split(chr(10))))}</p>')

CSS = """
@page { size: A4; margin: 14mm 15mm 12mm 15mm; }
* { box-sizing: border-box; }
body { margin:0; font: 10pt/1.38 Helvetica, Arial, sans-serif; color:#111; }
h1 { font-size: 16pt; margin: 0 0 6pt; letter-spacing:-.2pt; }
h2 { font-size: 11.5pt; margin: 10pt 0 4pt; padding-top: 5pt;
     border-top: 1px solid #d8d7d3; break-after: avoid; }
p { margin: 0 0 6pt; text-align: justify; hyphens: auto; }
code { font: 9.3pt/1 "SF Mono", Menlo, Consolas, monospace; background:#f2f1ee;
       padding: .5pt 2.5pt; border-radius: 2px; }
a { color:#1c5cab; text-decoration: none; word-break: break-all; }
.figs { display:flex; flex-wrap:wrap; gap:3mm; margin: 5pt 0 6pt;
        break-inside: avoid; align-items:flex-start; justify-content:center; }
/* Todas as figuras ocupam meia largura: 2 por linha, e a sobra desce para a
   linha seguinte no mesmo tamanho (3 viram 2+1, 4 viram 2+2). O flex-grow: 0
   e o que impede a figura orfa de esticar ate a largura toda. */
.figs figure { margin:0; flex:0 1 calc(50% - 1.5mm); min-width:0; }
.figs.cols1 figure { flex-basis: 72%; }
.figs img { width:100%; height:auto; display:block; }
strong { font-weight:650; }
table { border-collapse: collapse; width:100%; font-size: 8.8pt; margin: 4pt 0 7pt;
        break-inside: avoid; }
th { text-align:left; font-weight:650; border-bottom: 1px solid #b9b8b4; padding: 2.2pt 4pt; }
td { border-bottom: 1px solid #e8e7e4; padding: 2.2pt 4pt; vertical-align: top; }
td code, th code { font-size: 8.2pt; }
/* colunas numericas estreitas: sem isso um valor como 2,5e-4 quebra em duas linhas */
td:nth-child(2), td:nth-child(3), th:nth-child(2), th:nth-child(3) { white-space: nowrap; }
"""
titulo = next((b[2:].strip() for b in blocks if b.startswith('# ')), SRC)
doc = ('<!doctype html><html lang="pt-BR"><head><meta charset="utf-8">'
       f'<title>{html.escape(titulo)}</title><style>{CSS}</style></head><body>'
       + '\n'.join(out) + '</body></html>')
open(ROOT/SRC.replace('.md', '.html'), 'w').write(doc)
print(f'{len(blocks)} blocos -> {SRC.replace(chr(46)+chr(109)+chr(100), chr(46)+"html")}')
