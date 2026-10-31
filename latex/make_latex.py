# Converts master_draft.md to a self-contained Overleaf project (main.tex + figures/).
# Usage (from the folder that holds master_draft.md and figures/):  python make_latex.py
import re, subprocess, sys, os

SRC = "master_draft.md"
OUT = "latex/main.tex"
md = open(SRC, encoding="utf-8").read()

def pandoc_inline(chunks):
    """Convert a list of small markdown strings (one paragraph each) to LaTeX strings."""
    if not chunks:
        return []
    doc = "\n\n".join((c.strip() or "EMPTYCELL") for c in chunks)
    r = subprocess.run(["pandoc", "-f", "markdown+tex_math_dollars+autolink_bare_uris", "-t", "latex", "--wrap=none"],
                       input=doc, capture_output=True, text=True, check=True)
    parts = [p.strip() for p in re.split(r"\n\s*\n", r.stdout.strip())]
    if len(parts) != len(chunks):
        raise RuntimeError(f"cell count mismatch {len(parts)} vs {len(chunks)}")
    return ["" if p == "EMPTYCELL" else p for p in parts]

MATH = re.compile(r"(\$[^$\n]*\$)")

def textfix(s):
    """Unicode and cross-reference fixes, applied to text outside inline math."""
    out = []
    for seg in MATH.split(s):
        if seg.startswith("$") and seg.endswith("$") and len(seg) > 1:
            out.append(seg); continue
        seg = seg.replace("\u207b\u00b9", "`\\ensuremath{^{-1}}`{=latex}").replace("\u207b\u00b2", "`\\ensuremath{^{-2}}`{=latex}")
        seg = seg.replace("\u00b2", "`\\ensuremath{^{2}}`{=latex}").replace("\u00b9", "`\\ensuremath{^{1}}`{=latex}")
        seg = seg.replace("\u00d7", "`\\ensuremath{\\times}`{=latex}").replace("\u2212", "`\\ensuremath{-}`{=latex}")
        seg = re.sub(r"(?<=[\s(])-(?=\d)", lambda m: "`\\ensuremath{-}`{=latex}", seg)
        # equations
        seg = re.sub(r"Eqs\. \((\d+)\) to \((\d+)\)", lambda m: f"`Eqs.~\\eqref{{eq:{m[1]}}} to~\\eqref{{eq:{m[2]}}}`{{=latex}}", seg)
        seg = re.sub(r"Eq\. \((\d+)\)", lambda m: f"`Eq.~\\eqref{{eq:{m[1]}}}`{{=latex}}", seg)
        seg = re.sub(r"Eq\. (\d+)(?!\d)", lambda m: f"`Eq.~\\ref{{eq:{m[1]}}}`{{=latex}}", seg)
        # tables and figures
        seg = re.sub(r"\b(Table|Figure)s? (\d+)", lambda m: f"`{'Table' if m[1]=='Table' else 'Fig.'}~\\ref{{{'tab' if m[1]=='Table' else 'fig'}:{m[2]}}}`{{=latex}}", seg)
        # citations such as [12]
        seg = re.sub(r"\s\[(\d{1,2})\]", lambda m: f"`~\\cite{{ref{m[1]}}}`{{=latex}}", seg)
        out.append(seg)
    return "".join(out)

# ---- split off references
head, refs = md.split("\n## References\n")
# ---- title, author, abstract
title = re.match(r"# (.+)", head).group(1).strip()
author = re.search(r"^\*(\[.*\])\*$", head, re.M).group(1)
abs_block = re.search(r"## Abstract\n\n(.+?)\n\n## ", head, re.S).group(1).strip()
abs_text, kw_line = abs_block.split("\n\n")
kw_text = re.sub(r"^\*Index Terms:\*\s*", "", kw_line.strip())
body = head.split("\n## Introduction", 1)[1]
body = "## Introduction" + body

# ---- display equations -> placeholders
eqs = []
def eq_sub(m):
    inner = m.group(1).strip()
    tag = re.search(r"\\qquad\s*\\text\{\((\d+)\)\}", inner)
    n = tag.group(1) if tag else None
    inner = re.sub(r"\s*\\qquad\s*\\text\{\(\d+\)\}", "", inner).strip()
    if n in ("5", "11", "14", "16", "17", "19", "21", "22"):   # too wide for one column: stack the parts
        parts = [p.strip() for p in re.split(r"\\qquad|\\quad\\text\{that is\}\\quad", inner) if p.strip()]
        inner = "\\begin{gathered}\n" + " \\\\\n".join(parts) + "\n\\end{gathered}"
    env = "equation" if n else "equation*"
    lab = f"\\label{{eq:{n}}}" if n else ""
    eqs.append(f"\\begin{{{env}}}\n{inner}{lab}\n\\end{{{env}}}")
    return f"\n\n@@EQ{len(eqs)-1}@@\n\n"
body = re.sub(r"\$\$(.+?)\$\$", eq_sub, body, flags=re.S)

# ---- tables and figures
raws = []
def raw(tex):
    raws.append(tex); return f"@@RAW{len(raws)-1}@@"

lines = body.split("\n")
res, i = [], 0
cells_todo = []   # (key, markdown)
def reg(md_text):
    cells_todo.append(textfix(md_text)); return len(cells_todo) - 1

pending = []  # builders run after cell conversion
while i < len(lines):
    ln = lines[i]
    mt = re.match(r"\*\*Table (\d+)\. (.*?)\*\*\s*$", ln)
    if mt:
        num, cap = mt.group(1), mt.group(2)
        j = i + 1
        while lines[j].strip() == "": j += 1
        rows = []
        while j < len(lines) and lines[j].startswith("|"):
            rows.append([c.strip() for c in lines[j].strip().strip("|").split("|")]); j += 1
        header, align, data = rows[0], rows[1], rows[2:]
        cap_i = reg(cap)
        grid = [[reg(c) for c in r] for r in [header] + data]
        widths = []
        for c in range(len(header)):
            L = max(min(len(re.sub(r"[$\\{}]", "", r[c])), 38) for r in [header] + data)
            widths.append(max(L, 5))
        tot = sum(widths); widths = [w / tot for w in widths]
        key = len(raws); raws.append(None)
        pending.append(("table", key, num, cap_i, grid, widths))
        res.append(f"\n@@RAW{key}@@\n")
        i = j; continue
    mi = re.match(r"!\[\]\((.+?)\)\s*$", ln)
    if mi:
        j = i + 1
        while lines[j].strip() == "": j += 1
        mc = re.match(r"\*\*Figure (\d+)\.\*?\*?\s*(.*?)(\*\*)?\s*$", lines[j])
        num, cap = mc.group(1), mc.group(2)
        # caption may be fully bold "**Figure 4. text**" or "**Figure 1.** text"
        cap = re.sub(r"\*\*$", "", cap).strip()
        cap_i = reg(cap)
        key = len(raws); raws.append(None)
        pending.append(("figure", key, num, cap_i, mi.group(1), None))
        res.append(f"\n@@RAW{key}@@\n")
        i = j + 1; continue
    res.append(ln); i += 1
body = "\n".join(res)

# ---- convert registered cells in one pandoc call
converted = pandoc_inline(cells_todo)
for kind, key, num, cap_i, g, widths in pending:
    cap = converted[cap_i]
    if kind == "table":
        spec = "".join(f">{{\\raggedright\\arraybackslash}}p{{\\dimexpr {w:.3f}\\linewidth-2\\tabcolsep\\relax}}" for w in widths)
        rows_tex = []
        for ri, r in enumerate(g):
            cells = [converted[c] for c in r]
            if ri == 0: cells = [f"\\textbf{{{c}}}" for c in cells]
            rows_tex.append(" & ".join(cells) + " \\\\")
        star = "*" if len(widths) >= 4 else ""
        size = "\\small" if star else "\\footnotesize"
        tex = (f"\\begin{{table{star}}}[!t]\n\\centering{size}\n"
               f"\\caption{{{cap}}}\\label{{tab:{num}}}\n"
               f"\\begin{{tabular}}{{{spec}}}\n\\toprule\n{rows_tex[0]}\n\\midrule\n" + "\n".join(rows_tex[1:]) +
               f"\n\\bottomrule\n\\end{{tabular}}\n\\end{{table{star}}}")
    else:
        tex = (f"\\begin{{figure*}}[!t]\n\\centering\n\\includegraphics[width=0.95\\textwidth]{{{g}}}\n"
               f"\\caption{{{cap}}}\\label{{fig:{num}}}\n\\end{{figure*}}")
    raws[key] = tex

# ---- remaining text
body = textfix(body)
# paragraph after an equation that starts in lower case continues the sentence
def mark_eq(m): return m.group(0)
body = re.sub(r"(@@EQ\d+@@\n\n)([a-z])", lambda m: m.group(1) + "`\\noindent`{=latex} " + m.group(2), body)
for k, t in enumerate(raws):
    body = body.replace(f"@@RAW{k}@@", "```{=latex}\n" + t + "\n```")
for k, t in enumerate(eqs):
    body = body.replace(f"@@EQ{k}@@", "```{=latex}\n" + t + "\n```")
r = subprocess.run(["pandoc", "-f", "markdown+tex_math_dollars+autolink_bare_uris+raw_tex", "-t", "latex", "--wrap=preserve",
                    "--top-level-division=section", "--shift-heading-level-by=-1"],
                   input=body, capture_output=True, text=True, check=True)
main = r.stdout
main = re.sub(r"\\hypertarget\{[^}]*\}\{%\n(\\(?:sub)*section\{.*?\}\\label\{[^}]*\})\}", r"\1", main)
main = re.sub(r"\\label\{(?!eq:|tab:|fig:)[^}]*\}", "", main)   # drop pandoc heading labels
main = re.sub(r"\\texorpdfstring\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}\{[^{}]*\}", r"\1", main)

# ---- bibliography
bib = []
for blk in re.split(r"\n\s*\n", refs.strip()):
    m = re.match(r"\[(\d+)\]\s*(.*)", blk.strip(), re.S)
    if m: bib.append((m[1], m[2].replace("\n", " ")))
bib_tex = pandoc_inline([b[1] for b in bib])
bib_block = "\\begin{thebibliography}{99}\n" + "\n".join(f"\\bibitem{{ref{n}}} {t}" for (n, _), t in zip(bib, bib_tex)) + "\n\\end{thebibliography}"

abs_tex = pandoc_inline([textfix(abs_text)])[0]
title_tex = pandoc_inline([title])[0]
auth_tex = pandoc_inline([author.strip("[]")])[0]

PRE = r"""\documentclass[conference]{IEEEtran}
\IEEEoverridecommandlockouts
\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage{amsmath,amssymb}
\usepackage{graphicx}
\usepackage{booktabs,array}
\usepackage{textcomp}
\usepackage{xurl}
\usepackage[hidelinks]{hyperref}
\DeclareUnicodeCharacter{00B0}{\textdegree}
\DeclareUnicodeCharacter{03C6}{\ensuremath{\varphi}}
\DeclareUnicodeCharacter{03BB}{\ensuremath{\lambda}}
\renewcommand{\arraystretch}{1.2}
\providecommand{\tightlist}{\setlength{\itemsep}{0pt}\setlength{\parskip}{0pt}}
\sloppy
"""
AUTH = r"""\author{\IEEEauthorblockN{Santhosh Kumar D}
\IEEEauthorblockA{\textit{Dept. of Computer Science and Engineering with AIML}\\
\textit{SRM Institute of Science and Technology}\\
Tiruchirappalli, India \\
sd8663@srmist.edu.in}
\and
\IEEEauthorblockN{[Co-author name]}
\IEEEauthorblockA{\textit{[Department]}\\
\textit{[Institution]}\\
{[City, Country]} \\
{[email]}}
\and
\IEEEauthorblockN{[Advisor name]}
\IEEEauthorblockA{\textit{[Designation, department]}\\
\textit{[Institution]}\\
{[City, Country]} \\
{[email]}}}
"""
kw_tex = pandoc_inline([kw_text])[0]
doc = PRE + f"\\title{{{title_tex}}}\n\n" + AUTH + "\n\\begin{document}\n\\maketitle\n\n"
doc += f"\\begin{{abstract}}\n{abs_tex}\n\\end{{abstract}}\n\n\\begin{{IEEEkeywords}}\n{kw_tex}\n\\end{{IEEEkeywords}}\n\n" + main + "\n\n" + bib_block + "\n\n\\end{document}\n"
os.makedirs("latex", exist_ok=True)
open(OUT, "w", encoding="utf-8").write(doc)
print("written", OUT, len(doc))
