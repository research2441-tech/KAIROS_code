"""Builds the manuscript into the IEEE master template: native OMML equations (pandoc LaTeX->OMML),
template-style numbered display equations (4-column matrix), numeric citations, tables, figures,
algorithm boxes, empty header/footer."""
import os, re, shutil, subprocess, zipfile, html, json, tempfile
from PIL import Image

TPL = "/mnt/user-data/uploads/e6307675-Q1_SCI_IEEE_Transactions_Master_Template.docx"
CM = '<w:rPr><w:rFonts w:ascii="Cambria Math" w:hAnsi="Cambria Math"/></w:rPr>'


def esc(s):
    return html.escape(s, quote=False)


# ------------------------------------------------------------------ LaTeX -> OMML
class MathCache:
    def __init__(self, path="omml_cache.json"):
        self.path = path
        self.d = json.load(open(path)) if os.path.exists(path) else {}

    def convert(self, items):
        """items: list of (latex, display_bool). Fills cache for missing ones."""
        todo = [x for x in dict.fromkeys(items) if f"{int(x[1])}|{x[0]}" not in self.d]
        if not todo:
            return
        md = []
        for lx, disp in todo:
            md.append(("$$" + lx + "$$") if disp else ("$" + lx + "$"))
        tmp = tempfile.mkdtemp()
        open(f"{tmp}/m.md", "w").write("\n\n".join(md) + "\n")
        subprocess.run(["pandoc", f"{tmp}/m.md", "-f", "markdown+tex_math_dollars", "-o", f"{tmp}/m.docx"], check=True)
        x = zipfile.ZipFile(f"{tmp}/m.docx").read("word/document.xml").decode()
        paras = re.findall(r"<w:p>.*?</w:p>|<w:p [^>]*>.*?</w:p>", x, re.S)
        paras = [p for p in paras if "<m:oMath" in p]
        assert len(paras) == len(todo), (len(paras), len(todo))
        for (lx, disp), p in zip(todo, paras):
            m = re.search(r"<m:oMath>.*?</m:oMath>", p, re.S)
            if m is None:
                raise ValueError("no omml for " + lx)
            self.d[f"{int(disp)}|{lx}"] = m.group(0)
        json.dump(self.d, open(self.path, "w"))

    def get(self, lx, disp=False, size=None):
        o = self.d[f"{int(disp)}|{lx}"]
        # Cambria Math on every math run, as in the template (and the surrounding font size inside tables)
        cm = CM if size is None else CM.replace("</w:rPr>", f'<w:sz w:val="{size}"/><w:szCs w:val="{size}"/></w:rPr>')
        o = re.sub(r"<m:r>(?!<m:rPr>|<w:rPr>)", "<m:r>" + cm, o)
        o = re.sub(r"<m:r><m:rPr>(.*?)</m:rPr>(?!<w:rPr>)", lambda mm: "<m:r><m:rPr>" + mm.group(1) + "</m:rPr>" + cm, o)
        if size is not None:
            o = o.replace("<m:ctrlPr>" + CM, "<m:ctrlPr>" + cm)
        return o


MATHC = MathCache()


# ------------------------------------------------------------------ inline markup
TOK = re.compile(r"(\$[^$]+\$|\*\*[^*]+\*\*|\*[^*]+\*)")


def collect_math(text):
    return [(m[1:-1], False) for m in re.findall(r"\$[^$]+\$", text)]


def runs(text, size=None, color=None, bold=False, italic=False):
    out = []
    for part in TOK.split(text):
        if not part:
            continue
        if part.startswith("$"):
            out.append(MATHC.get(part[1:-1], size=size))
            continue
        b, i = bold, italic
        if part.startswith("**"):
            part, b = part[2:-2], True
        elif part.startswith("*") and part.endswith("*") and len(part) > 1:
            part, i = part[1:-1], True
        rp = ""
        if b: rp += "<w:b/>"
        if i: rp += "<w:i/>"
        if color: rp += f'<w:color w:val="{color}"/>'
        if size: rp += f'<w:sz w:val="{size}"/><w:szCs w:val="{size}"/>'
        out.append(f'<w:r>{"<w:rPr>" + rp + "</w:rPr>" if rp else ""}<w:t xml:space="preserve">{esc(part)}</w:t></w:r>')
    return "".join(out)


def para(text, style=None, ind_first=259, jc="both", size=None, spacing=None, keep=False, ind_left=None, ind_right=None, hanging=None):
    pp = ""
    if style: pp += f'<w:pStyle w:val="{style}"/>'
    if keep: pp += "<w:keepNext/>"
    if spacing: pp += spacing
    ind = ""
    if ind_first: ind += f' w:firstLine="{ind_first}"'
    if ind_left is not None: ind += f' w:left="{ind_left}"'
    if ind_right is not None: ind += f' w:right="{ind_right}"'
    if hanging is not None: ind += f' w:hanging="{hanging}"'
    if ind: pp += f"<w:ind{ind}/>"
    if jc: pp += f'<w:jc w:val="{jc}"/>'
    return f"<w:p><w:pPr>{pp}</w:pPr>{runs(text, size=size)}</w:p>"


def eq_para(latex, num):
    o = MATHC.get(latex, True)
    inner = re.sub(r"^<m:oMath>|</m:oMath>$", "", o)
    mpr = ('<m:mPr><m:plcHide m:val="1"/><m:mcs><m:mc><m:mcPr><m:count m:val="4"/><m:mcJc m:val="center"/>'
           '</m:mcPr></m:mc></m:mcs><m:ctrlPr>' + CM + '</m:ctrlPr></m:mPr>')
    numr = f'<m:r><m:rPr><m:sty m:val="p"/></m:rPr>{CM}<m:t>({num})</m:t></m:r>'
    m = f"<m:m>{mpr}<m:mr><m:e/><m:e>{inner}</m:e><m:e/><m:e>{numr}</m:e></m:mr></m:m>"
    return f'<w:p><w:pPr><w:spacing w:before="60" w:after="60"/><w:jc w:val="both"/></w:pPr><m:oMathPara><m:oMath>{m}</m:oMath></m:oMathPara></w:p>'


# ------------------------------------------------------------------ figures
class Media:
    def __init__(self):
        self.items = []

    def add(self, path):
        rid = f"rIdImg{len(self.items) + 1}"
        self.items.append((rid, path))
        return rid


def fig_para(media, path, width_in):
    rid = media.add(path)
    w, h = Image.open(path).size
    cx = int(width_in * 914400); cy = int(cx * h / w)
    n = len(media.items)
    return (f'<w:p><w:pPr><w:keepNext/><w:spacing w:before="80" w:after="40"/><w:jc w:val="center"/></w:pPr><w:r><w:drawing>'
            f'<wp:inline distT="0" distB="0" distL="0" distR="0"><wp:extent cx="{cx}" cy="{cy}"/><wp:docPr id="{100 + n}" name="Figure {n}"/>'
            f'<a:graphic xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"><a:graphicData uri="http://schemas.openxmlformats.org/drawingml/2006/picture">'
            f'<pic:pic xmlns:pic="http://schemas.openxmlformats.org/drawingml/2006/picture"><pic:nvPicPr><pic:cNvPr id="{n}" name="img{n}.png"/><pic:cNvPicPr/></pic:nvPicPr>'
            f'<pic:blipFill><a:blip r:embed="{rid}"/><a:stretch><a:fillRect/></a:stretch></pic:blipFill>'
            f'<pic:spPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="{cx}" cy="{cy}"/></a:xfrm><a:prstGeom prst="rect"><a:avLst/></a:prstGeom></pic:spPr></pic:pic>'
            f'</a:graphicData></a:graphic></wp:inline></w:drawing></w:r></w:p>')


# ------------------------------------------------------------------ tables
def cell(text, w, head=False, jc="center", span=1, size=14, vmerge=None, shade=None):
    tcp = f'<w:tcW w:w="{w}" w:type="dxa"/>'
    if span > 1: tcp += f'<w:gridSpan w:val="{span}"/>'
    if vmerge: tcp += f'<w:vMerge w:val="{vmerge}"/>' if vmerge == "restart" else "<w:vMerge/>"
    if head: tcp += '<w:shd w:val="clear" w:color="auto" w:fill="E7E6E6"/>'
    elif shade: tcp += f'<w:shd w:val="clear" w:color="auto" w:fill="{shade}"/>'
    tcp += '<w:tcMar><w:top w:w="15" w:type="dxa"/><w:left w:w="45" w:type="dxa"/><w:bottom w:w="15" w:type="dxa"/><w:right w:w="45" w:type="dxa"/></w:tcMar><w:vAlign w:val="center"/>'
    body = runs(text, size=size, bold=head)
    return f'<w:tc><w:tcPr>{tcp}</w:tcPr><w:p><w:pPr><w:spacing w:after="0" w:line="220" w:lineRule="auto"/><w:jc w:val="{jc}"/></w:pPr>{body}</w:p></w:tc>'


def table(rows, widths, head_rows=1, first_col_left=True, size=14, shade_rows=()):
    grid = "".join(f'<w:gridCol w:w="{w}"/>' for w in widths)
    out = [f'<w:tbl><w:tblPr><w:tblStyle w:val="TableGrid"/><w:tblW w:w="{sum(widths)}" w:type="dxa"/><w:jc w:val="center"/>'
           f'<w:tblLayout w:type="fixed"/><w:tblLook w:val="04A0" w:firstRow="1" w:lastRow="0" w:firstColumn="1" w:lastColumn="0" w:noHBand="0" w:noVBand="1"/></w:tblPr><w:tblGrid>{grid}</w:tblGrid>']
    for ri, r in enumerate(rows):
        head = ri < head_rows
        tr = '<w:tr><w:trPr><w:cantSplit/>' + ('<w:tblHeader/>' if head else '') + '<w:jc w:val="center"/></w:trPr>'
        ci = 0
        for c in r:
            span = 1
            if isinstance(c, tuple):
                c, span = c
            w = sum(widths[ci:ci + span])
            jc = "left" if (first_col_left and ci == 0 and not head) else "center"
            tr += cell(c, w, head, jc, span, size, shade="F2F2F2" if ri in shade_rows else None)
            ci += span
        out.append(tr + "</w:tr>")
    out.append("</w:tbl>")
    return "".join(out)


def alg_box(title, lines, width):
    """Algorithm as a ruled single-cell table (top/bottom rules), numbered lines, 8 pt."""
    ps = [f'<w:p><w:pPr><w:spacing w:after="20"/></w:pPr>{runs(title, size=16, bold=True)}</w:p>']
    for i, (indent, txt) in enumerate(lines, 1):
        num = f'<w:r><w:rPr><w:sz w:val="15"/></w:rPr><w:t xml:space="preserve">{i:>2}: </w:t></w:r>' if txt else ""
        ps.append(f'<w:p><w:pPr><w:spacing w:after="0" w:line="216" w:lineRule="auto"/><w:ind w:left="{indent * 220}"/></w:pPr>{num}{runs(txt, size=15)}</w:p>')
    b = ('<w:tblBorders><w:top w:val="single" w:sz="8" w:space="0" w:color="000000"/><w:bottom w:val="single" w:sz="8" w:space="0" w:color="000000"/>'
         '<w:left w:val="nil"/><w:right w:val="nil"/><w:insideH w:val="nil"/><w:insideV w:val="nil"/></w:tblBorders>')
    return (f'<w:tbl><w:tblPr><w:tblW w:w="{width}" w:type="dxa"/><w:jc w:val="center"/>{b}<w:tblLayout w:type="fixed"/></w:tblPr>'
            f'<w:tblGrid><w:gridCol w:w="{width}"/></w:tblGrid><w:tr><w:tc><w:tcPr><w:tcW w:w="{width}" w:type="dxa"/>'
            f'<w:tcMar><w:top w:w="40" w:type="dxa"/><w:left w:w="60" w:type="dxa"/><w:bottom w:w="40" w:type="dxa"/><w:right w:w="60" w:type="dxa"/></w:tcMar></w:tcPr>'
            + "".join(ps) + "</w:tc></w:tr></w:tbl>")


# ------------------------------------------------------------------ sections
PG = ('<w:pgSz w:w="12240" w:h="15840"/><w:pgMar w:top="1008" w:right="936" w:bottom="1080" w:left="936" '
      'w:header="720" w:footer="720" w:gutter="0"/>')


def sect(cols, continuous=True):
    c = '<w:cols w:num="2" w:space="288"/>' if cols == 2 else '<w:cols w:space="288"/>'
    return f'<w:sectPr>{"<w:type w:val=\"continuous\"/>" if continuous else ""}{PG}{c}<w:docGrid w:linePitch="360"/></w:sectPr>'


def sect_para(cols, continuous=True):
    return f'<w:p><w:pPr><w:spacing w:after="0"/>{sect(cols, continuous)}</w:pPr></w:p>'


ROMAN = ["I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X", "XI", "XII", "XIII", "XIV", "XV", "XVI", "XVII", "XVIII", "XIX", "XX"]


def build(blocks, out_path, front):
    """blocks: list of tuples; front: dict(title, authors, affil, abstract, keywords)."""
    # pass 1: numbering of equations, figures, tables, citations
    eqn, fign, tabn, cites = {}, {}, {}, {}
    for b in blocks:
        if b[0] == "eq": eqn[b[2]] = len(eqn) + 1
        if b[0] == "fig": fign[b[1]] = len(fign) + 1
        if b[0] == "tab": tabn[b[1]] = len(tabn) + 1
    textual = [front["abstract"]] + [x for b in blocks for x in b[1:] if isinstance(x, str)]

    def cite_order(t):
        for grp in re.findall(r"\[@([^\]]+)\]", t):
            for k in grp.split(","):
                k = k.strip()
                if k not in cites: cites[k] = len(cites) + 1
    for b in blocks:
        if b[0] in ("p", "pl", "li"): cite_order(b[1])
        if b[0] == "tab":
            for r in b[3]:
                for c in r: cite_order(c if isinstance(c, str) else c[0])
            cite_order(b[2])
        if b[0] == "fig": cite_order(b[3])

    def fmt_cites(grp):
        nums = sorted(cites[k.strip()] for k in grp.split(","))
        assert len(nums) <= 3, grp
        if len(nums) >= 3 and nums[-1] - nums[0] == len(nums) - 1:
            return f"[{nums[0]}–{nums[-1]}]"
        return "[" + ", ".join(map(str, nums)) + "]"

    def sub(t):
        t = re.sub(r"\[@([^\]]+)\]", lambda m: fmt_cites(m.group(1)), t)
        t = re.sub(r"\{eq:([\w-]+)\}", lambda m: f"({eqn[m.group(1)]})", t)
        t = re.sub(r"\{fig:([\w-]+)\}", lambda m: f"{fign[m.group(1)]}", t)
        t = re.sub(r"\{tab:([\w-]+)\}", lambda m: f"{ROMAN[tabn[m.group(1)] - 1]}", t)
        return t

    # math collection
    allm = collect_math(front["abstract"])
    for b in blocks:
        for x in b[1:]:
            if isinstance(x, str): allm += collect_math(x)
            if isinstance(x, list):
                for r in x:
                    if isinstance(r, (list, tuple)):
                        for c in r:
                            if isinstance(c, str): allm += collect_math(c)
                            elif isinstance(c, tuple) and isinstance(c[0], str): allm += collect_math(c[0])
        if b[0] == "eq": allm.append((b[1], True))
    MATHC.convert(allm)

    media = Media(); body = []
    body.append(para(front["title"], style="PaperTitle", ind_first=0, jc="center"))
    body.append(para(front["authors"], style="Authors", ind_first=0, jc="center"))
    body.append(para(front["affil"], style="Affiliations", ind_first=0, jc="center"))
    body.append('<w:p/>')
    body.append(f'<w:p><w:pPr><w:ind w:left="360" w:right="360"/><w:jc w:val="both"/></w:pPr><w:r><w:rPr><w:b/><w:i/></w:rPr><w:t>Abstract—</w:t></w:r>{runs(sub(front["abstract"]))}</w:p>')
    body.append(f'<w:p><w:pPr><w:ind w:left="360" w:right="360"/><w:jc w:val="both"/></w:pPr><w:r><w:rPr><w:b/><w:i/></w:rPr><w:t>Index Terms—</w:t></w:r>{runs(front["keywords"])}</w:p>')
    body.append(sect_para(1, continuous=False))
    for b in blocks:
        k = b[0]
        if k == "h1":
            body.append(f'<w:p><w:pPr><w:pStyle w:val="IEEESection"/><w:keepNext/></w:pPr><w:r><w:rPr><w:smallCaps/></w:rPr><w:t>{esc(b[1])}</w:t></w:r></w:p>')
        elif k == "h2":
            body.append(f'<w:p><w:pPr><w:pStyle w:val="IEEESubsection"/><w:keepNext/></w:pPr>{runs(b[1])}</w:p>')
        elif k == "p":
            body.append(para(sub(b[1])))
        elif k == "pl":   # paragraph without first-line indent
            body.append(para(sub(b[1]), ind_first=0))
        elif k == "li":   # list item with hanging indent
            body.append(para(sub(b[1]), ind_first=None, ind_left=259, hanging=180))
        elif k == "eq":
            body.append(eq_para(b[1], eqn[b[2]]))
        elif k == "fig":
            _, key, path, cap, width = b
            body.append(fig_para(media, path, width))
            body.append(f'<w:p><w:pPr><w:pStyle w:val="FigureCaption"/><w:jc w:val="both"/></w:pPr>{runs("Fig. " + str(fign[key]) + ". " + sub(cap))}</w:p>')
        elif k == "tab":
            _, key, title, rows, widths, opts = b
            body.append(f'<w:p><w:pPr><w:pStyle w:val="TableCaption"/><w:keepNext/><w:spacing w:before="100" w:after="40"/></w:pPr><w:r><w:t>TABLE {ROMAN[tabn[key] - 1]}</w:t></w:r><w:r><w:br/></w:r>{runs(sub(title).upper() if opts.get("upper", True) else sub(title))}</w:p>')
            rows = [[(sub(c[0]), c[1]) if isinstance(c, tuple) else sub(c) for c in r] for r in rows]
            body.append(table(rows, widths, opts.get("head", 1), size=opts.get("size", 14), shade_rows=opts.get("shade", ())))
            if opts.get("note"):
                body.append(para(sub(opts["note"]), ind_first=0, size=14, spacing='<w:spacing w:before="20" w:after="100"/>'))
            else:
                body.append('<w:p><w:pPr><w:spacing w:after="60"/></w:pPr></w:p>')
        elif k == "alg":
            _, title, lines, width = b
            body.append(alg_box(sub(title), [(i, sub(t)) for i, t in lines], width))
            body.append('<w:p><w:pPr><w:spacing w:after="60"/></w:pPr></w:p>')
        elif k == "wide_start":
            body.append(sect_para(2))
        elif k == "wide_end":
            body.append(sect_para(1))
        elif k == "ref":
            body.append(para(b[1], ind_first=None, ind_left=360, hanging=360, size=15, jc="both", spacing='<w:spacing w:after="20"/>'))
        elif k == "raw":
            body.append(b[1])
    # assemble
    work = tempfile.mkdtemp(); shutil.unpack_archive(TPL, work, "zip")
    doc = open(f"{work}/word/document.xml").read()
    head = doc[:doc.index("<w:body>") + len("<w:body>")]
    if "xmlns:wp=" not in head:
        raise RuntimeError("namespace wp missing")
    xml = head + "".join(body) + sect(2, continuous=True) + "</w:body></w:document>"
    open(f"{work}/word/document.xml", "w").write(xml)
    # empty footers (template footers carried page labels) - nothing in header or footer
    for f in os.listdir(f"{work}/word"):
        if f.startswith("footer") or f.startswith("header"):
            open(f"{work}/word/{f}", "w").write(re.sub(r"<w:p[ >].*</w:p>", "<w:p/>", open(f"{work}/word/{f}").read(), flags=re.S))
    rel = open(f"{work}/word/_rels/document.xml.rels").read()
    os.makedirs(f"{work}/word/media", exist_ok=True)
    add = ""
    for rid, p in media.items:
        fn = os.path.basename(p)
        shutil.copy(p, f"{work}/word/media/{fn}")
        add += f'<Relationship Id="{rid}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image" Target="media/{fn}"/>'
    rel = rel.replace("</Relationships>", add + "</Relationships>")
    open(f"{work}/word/_rels/document.xml.rels", "w").write(rel)
    ct = open(f"{work}/[Content_Types].xml").read()
    if 'Extension="png"' not in ct:
        ct = ct.replace("<Default ", '<Default Extension="png" ContentType="image/png"/><Default ', 1)
        open(f"{work}/[Content_Types].xml", "w").write(ct)
    # core properties: title, no template author text
    if os.path.exists(f"{work}/docProps/core.xml"):
        c = open(f"{work}/docProps/core.xml").read()
        c = re.sub(r"<dc:title>.*?</dc:title>", f"<dc:title>{esc(front['title'])}</dc:title>", c)
        open(f"{work}/docProps/core.xml", "w").write(c)
    if os.path.exists(out_path): os.remove(out_path)
    zf = zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED)
    for root, _, files in os.walk(work):
        for f in files:
            full = os.path.join(root, f); arc = os.path.relpath(full, work)
            zf.write(full, arc)
    zf.close()
    return dict(eq=len(eqn), fig=len(fign), tab=len(tabn), cites=cites)
