"""Build the editable malaria forecast presentation with only Python's stdlib."""

from __future__ import annotations

import subprocess
import zipfile
from html import escape
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
ODP_PATH = DOCS / "apresentacao_resultados.odp"

W, H = 13.333, 7.5
# Impress resolves this ODP page layout to a 16:9 canvas of 11.023 × 6.2 in.
# Scale drawing coordinates to that canvas so no content falls outside the slide.
GEOMETRY_SCALE = 11.023 / W
COLORS = {
    "ink": "#153B3B",
    "deep": "#0E3034",
    "teal": "#277C74",
    "mint": "#DDEDE8",
    "paper": "#F5F7F3",
    "white": "#FFFFFF",
    "orange": "#E98B45",
    "pale_orange": "#FCE9D9",
    "muted": "#547070",
    "line": "#C8D8D3",
    "red": "#C76352",
    "pale_red": "#F4E2DE",
    "blue": "#4B83A3",
    "pale_blue": "#E3EEF3",
}

NS = {
    "office": "urn:oasis:names:tc:opendocument:xmlns:office:1.0",
    "style": "urn:oasis:names:tc:opendocument:xmlns:style:1.0",
    "text": "urn:oasis:names:tc:opendocument:xmlns:text:1.0",
    "draw": "urn:oasis:names:tc:opendocument:xmlns:drawing:1.0",
    "fo": "urn:oasis:names:tc:opendocument:xmlns:xsl-fo-compatible:1.0",
    "svg": "urn:oasis:names:tc:opendocument:xmlns:svg-compatible:1.0",
    "presentation": "urn:oasis:names:tc:opendocument:xmlns:presentation:1.0",
    "manifest": "urn:oasis:names:tc:opendocument:xmlns:manifest:1.0",
    "dc": "http://purl.org/dc/elements/1.1/",
    "meta": "urn:oasis:names:tc:opendocument:xmlns:meta:1.0",
    "config": "urn:oasis:names:tc:opendocument:xmlns:config:1.0",
    "xlink": "http://www.w3.org/1999/xlink",
}

PREFIXES = " ".join(f'xmlns:{prefix}="{uri}"' for prefix, uri in NS.items())


def inch(value: float) -> str:
    return f"{value:.3f}in"


def paragraph_style(name: str, size: int, color: str, bold: bool = False,
                    align: str = "left", margin_bottom: float = 0.06,
                    line_height: int = 112) -> str:
    weight = ' fo:font-weight="bold"' if bold else ""
    return (
        f'<style:style style:name="{name}" style:family="paragraph">'
        f'<style:paragraph-properties fo:margin-top="0in" '
        f'fo:margin-bottom="{inch(margin_bottom)}" fo:line-height="{line_height}%" '
        f'fo:text-align="{align}"/>'
        f'<style:text-properties style:font-name="Liberation Sans" '
        f'fo:font-family="Liberation Sans" fo:font-size="{size}pt" '
        f'fo:color="{color}"{weight}/></style:style>'
    )


def make_automatic_styles() -> tuple[str, dict[str, str]]:
    styles = [
        '<style:style style:name="Frame" style:family="graphic"><style:graphic-properties '
        'draw:fill="none" draw:stroke="none" fo:padding="0in" '
        'draw:textarea-vertical-align="top"/></style:style>',
        '<style:style style:name="Rect" style:family="graphic"><style:graphic-properties '
        'draw:fill="solid" draw:fill-color="#FFFFFF" draw:stroke="none"/>'
        '</style:style>',
    ]
    styles.extend([
        paragraph_style("Title", 31, COLORS["ink"], True, margin_bottom=0.08, line_height=105),
        paragraph_style("TitleWhite", 31, COLORS["white"], True, margin_bottom=0.08, line_height=105),
        paragraph_style("CoverTitle", 35, COLORS["white"], True, margin_bottom=0.08, line_height=108),
        paragraph_style("Subtitle", 17, COLORS["muted"], margin_bottom=0.05, line_height=115),
        paragraph_style("SubtitleWhite", 17, "#D6E8E3", margin_bottom=0.05, line_height=115),
        paragraph_style("Eyebrow", 10, COLORS["teal"], True, margin_bottom=0.04),
        paragraph_style("EyebrowWhite", 10, "#91C7BA", True, margin_bottom=0.04),
        paragraph_style("Body", 16, COLORS["ink"], margin_bottom=0.10, line_height=118),
        paragraph_style("BodySmall", 12, COLORS["muted"], margin_bottom=0.07, line_height=115),
        paragraph_style("BodyWhite", 16, COLORS["white"], margin_bottom=0.10, line_height=118),
        paragraph_style("BodyWhiteSmall", 12, "#D6E8E3", margin_bottom=0.07, line_height=115),
        paragraph_style("CardTitle", 17, COLORS["ink"], True, margin_bottom=0.08, line_height=108),
        paragraph_style("CardTitleWhite", 17, COLORS["white"], True, margin_bottom=0.08, line_height=108),
        paragraph_style("Metric", 27, COLORS["teal"], True, margin_bottom=0.03, line_height=100),
        paragraph_style("MetricWhite", 27, COLORS["white"], True, margin_bottom=0.03, line_height=100),
        paragraph_style("MetricLabel", 12, COLORS["muted"], margin_bottom=0.03, line_height=108),
        paragraph_style("ChartLabel", 14, COLORS["ink"], True, margin_bottom=0.01, line_height=105),
        paragraph_style("ChartNumber", 15, COLORS["ink"], True, margin_bottom=0.01, line_height=100),
        paragraph_style("Footer", 9, COLORS["muted"], margin_bottom=0, line_height=100),
        paragraph_style("FooterWhite", 9, "#B9D1CB", margin_bottom=0, line_height=100),
        paragraph_style("CenterBody", 15, COLORS["ink"], False, "center", 0.05, 112),
        paragraph_style("CenterTitle", 15, COLORS["ink"], True, "center", 0.04, 105),
        paragraph_style("Tiny", 9, COLORS["muted"], margin_bottom=0.03, line_height=110),
        paragraph_style("TinyWhite", 9, "#D6E8E3", margin_bottom=0.03, line_height=110),
    ])
    colors = set(COLORS.values())
    for i, color in enumerate(sorted(colors)):
        styles.append(
            f'<style:style style:name="G{i}" style:family="graphic">'
            f'<style:graphic-properties draw:fill="solid" draw:fill-color="{color}" '
            'draw:stroke="none"/></style:style>'
        )
    color_names = {color: f"G{i}" for i, color in enumerate(sorted(colors))}
    return "<office:automatic-styles>" + "".join(styles) + "</office:automatic-styles>", color_names


class Slide:
    def __init__(self, name: str, background: str = "paper") -> None:
        self.name = name
        self.shapes: list[str] = []
        self.rect(0, 0, W, H, COLORS[background])

    def rect(self, x: float, y: float, w: float, h: float, color: str,
             radius: bool = False) -> None:
        hex_color = COLORS.get(color, color)
        style = COLOR_NAMES[hex_color]
        kind = "ellipse" if radius else "rect"
        x, y, w, h = (v * GEOMETRY_SCALE for v in (x, y, w, h))
        self.shapes.append(
            f'<draw:{kind} draw:style-name="{style}" draw:layer="layout" '
            f'svg:x="{inch(x)}" svg:y="{inch(y)}" svg:width="{inch(w)}" '
            f'svg:height="{inch(h)}"/>'
        )

    def text(self, x: float, y: float, w: float, h: float,
             paragraphs: list[tuple[str, str]]) -> None:
        x, y, w, h = (v * GEOMETRY_SCALE for v in (x, y, w, h))
        content = "".join(
            f'<text:p text:style-name="{style}">{"<text:line-break/>".join(escape(line) for line in value.splitlines())}</text:p>'
            for style, value in paragraphs
        )
        self.shapes.append(
            f'<draw:frame draw:style-name="Frame" draw:layer="layout" '
            f'svg:x="{inch(x)}" svg:y="{inch(y)}" svg:width="{inch(w)}" '
            f'svg:height="{inch(h)}"><draw:text-box>{content}</draw:text-box></draw:frame>'
        )

    def card(self, x: float, y: float, w: float, h: float, color: str,
             title: str, lines: list[str], title_style: str = "CardTitle",
             body_style: str = "BodySmall") -> None:
        self.rect(x, y, w, h, color)
        paragraphs = [(title_style, title)] + [(body_style, line) for line in lines]
        self.text(x + 0.20, y + 0.18, w - 0.40, h - 0.30, paragraphs)

    def footer(self, page_number: int, dark: bool = False) -> None:
        style = "FooterWhite" if dark else "Footer"
        self.text(0.65, 7.17, 11.7, 0.20, [(style, f"BREVES • PREVISÃO MENSAL  |  {page_number:02d}")])


def slide_title(slide: Slide, title: str, subtitle: str | None = None,
                dark: bool = False, section: str | None = None) -> None:
    slide.text(0.68, 0.42, 11.9, 0.65, [("TitleWhite" if dark else "Title", title)])
    if section:
        slide.text(0.70, 0.18, 5.0, 0.24, [("EyebrowWhite" if dark else "Eyebrow", section.upper())])
    if subtitle:
        slide.text(0.70, 1.05, 11.9, 0.50, [("SubtitleWhite" if dark else "Subtitle", subtitle)])


def build_slides() -> list[Slide]:
    slides: list[Slide] = []

    # 1 — Cover
    s = Slide("Capa", "deep")
    s.rect(10.15, 0.0, 3.18, 7.5, "ink")
    s.rect(10.63, 1.12, 1.68, 1.68, "teal", radius=True)
    s.rect(11.12, 1.61, 0.70, 0.70, "orange", radius=True)
    s.rect(10.60, 4.70, 0.42, 1.15, "teal")
    s.rect(11.27, 4.08, 0.42, 1.77, "orange")
    s.rect(11.94, 3.38, 0.42, 2.47, "blue")
    s.text(0.82, 0.82, 8.8, 0.35, [("EyebrowWhite", "PESQUISA • BREVES, PARÁ")])
    s.text(0.82, 1.75, 8.8, 1.85, [("CoverTitle", "Prever notificações\nmensais de malária")])
    s.text(0.85, 4.00, 8.2, 0.95, [("SubtitleWhite", "O que os modelos aprendem com o histórico de casos — sem dados de clima")])
    s.text(0.85, 5.80, 8.4, 0.55, [("BodyWhiteSmall", "Resultados de 2007 a setembro de 2026\nUma explicação simples dos modelos e dos achados")])
    s.footer(1, dark=True)
    slides.append(s)

    # 2 — Question and data
    s = Slide("Pergunta")
    slide_title(s, "A pergunta da pesquisa", "Usar notificações passadas para prever o próximo mês.", section="Ponto de partida")
    s.rect(0.72, 1.80, 11.88, 1.18, "mint")
    s.text(1.02, 2.04, 11.25, 0.67, [("CardTitle", "Qual método prevê melhor a contagem mensal de notificações em Breves?")])
    stats = [("237", "meses completos"), ("30.236", "notificações somadas"), ("2007–2026", "janeiro de 2007 a setembro de 2026")]
    x_positions = [0.72, 4.82, 8.92]
    for x, (value, label) in zip(x_positions, stats):
        s.rect(x, 3.35, 3.70, 1.43, "white")
        s.text(x + 0.18, 3.60, 3.30, 0.57, [("Metric", value)])
        s.text(x + 0.20, 4.24, 3.28, 0.34, [("MetricLabel", label)])
    s.rect(0.72, 5.10, 11.88, 1.18, "pale_orange")
    s.text(0.98, 5.31, 11.32, 0.78, [("Body", "Os modelos receberam somente contagens passadas e o mês do ano. Outubro de 2026 ficou de fora por estar incompleto; chuva e temperatura não foram usadas.")])
    s.footer(2)
    slides.append(s)

    # 3 — Evaluation protocol
    s = Slide("Como foi avaliado")
    slide_title(s, "Como testamos sem olhar o futuro", "O teste imita uma previsão feita mês após mês.", section="Como funciona")
    s.rect(0.82, 2.05, 11.50, 0.18, "line")
    stages = [
        (1.00, "1", "Aprender", "Usar meses antigos para\nidentificar padrões."),
        (4.95, "2", "Prever um mês", "Estimar o mês seguinte\ncom o que já era conhecido."),
        (8.90, "3", "Atualizar", "Quando o valor real chega,\nele entra no mês seguinte."),
    ]
    for x, number, title, body in stages:
        s.rect(x, 1.80, 0.70, 0.70, "teal", radius=True)
        s.text(x + 0.16, 1.98, 0.38, 0.32, [("CenterTitle", number)])
        s.text(x - 0.04, 2.80, 3.45, 0.40, [("CardTitle", title)])
        s.text(x - 0.04, 3.33, 3.55, 1.03, [("BodySmall", body)])
    s.card(0.82, 4.75, 5.55, 1.37, "mint", "Comparação principal", ["112 meses para desenvolvimento • 113 para teste", "Teste de maio de 2017 a setembro de 2026"], body_style="BodySmall")
    s.card(6.77, 4.75, 5.55, 1.37, "pale_blue", "O que isso significa", ["Cada previsão olha só um mês adiante.", "Não é uma previsão de dez anos sem atualizar os dados."], body_style="BodySmall")
    s.footer(3)
    slides.append(s)

    # 4 — Model families
    s = Slide("Famílias de modelos")
    slide_title(s, "Os algoritmos, em palavras simples", "Cada família procura um tipo diferente de padrão.", section="Resumo dos métodos")
    s.card(0.72, 1.72, 3.83, 2.03, "mint", "Regras de referência", ["Último mês: o próximo pode parecer com o mais recente.", "Média de 3 meses: suaviza subidas e descidas."], body_style="BodySmall")
    s.card(4.75, 1.72, 3.83, 2.03, "pale_blue", "Modelos de série temporal", ["ETS, ARIMA e SARIMA acompanham tendências e mudanças.", "SARIMA também procura um padrão anual."], body_style="BodySmall")
    s.card(8.78, 1.72, 3.83, 2.03, "pale_orange", "Modelos de contagem", ["Poisson e binomial negativa usam meses passados para estimar contagens.", "Binomial negativa aceita oscilações maiores."], body_style="BodySmall")
    s.card(0.72, 4.05, 5.83, 2.05, "white", "INGARCH: contagem que depende da contagem", ["Usa o valor real do mês anterior e a previsão anterior.", "Comparamos Poisson e binomial negativa, com um ciclo anual."], body_style="BodySmall")
    s.card(6.78, 4.05, 5.83, 2.05, "white", "Aprendizado de máquina", ["Random Forest combina muitas regras pequenas; boosting corrige erros aos poucos.", "SVR encontra uma curva que acompanha os padrões, mesmo quando não são retos."], body_style="BodySmall")
    s.footer(4)
    slides.append(s)

    # 5 — Explain ML / count algorithms
    s = Slide("Poisson, NB e máquina")
    slide_title(s, "O que muda de um algoritmo para outro?", "Um exemplo simples: todos recebem o mesmo histórico, mas fazem contas diferentes.", section="Resumo dos métodos")
    s.card(0.72, 1.85, 3.83, 3.32, "pale_orange", "Poisson", ["Estima uma quantidade e espera oscilações próximas desse valor.", "Foi uma referência para prever contagens."], body_style="BodySmall")
    s.card(4.75, 1.85, 3.83, 3.32, "mint", "Binomial negativa", ["Também estima uma contagem, mas aceita oscilações maiores.", "Pode ajudar quando alguns meses ficam muito acima ou abaixo da média."], body_style="BodySmall")
    s.card(8.78, 1.85, 3.83, 3.32, "pale_blue", "SVR, Forest e Boosting", ["Aprendem relações entre meses passados e o mês previsto.", "SVR teve o menor erro na divisão principal. Forest e boosting aprendem regras de outros modos."], body_style="BodySmall")
    s.rect(0.72, 5.55, 11.88, 0.76, "deep")
    s.text(0.96, 5.72, 11.32, 0.42, [("BodyWhiteSmall", "Analogia: ferramentas diferentes resolvem a mesma tarefa com os mesmos dados.")])
    s.footer(5)
    slides.append(s)

    # 6 — Primary results
    s = Slide("Resultado 50 50")
    slide_title(s, "Na divisão principal, o SVR teve menor erro", "Menor MAE = menor diferença média entre previsão e valor observado.", section="Resultados")
    chart = [
        ("SVR", 24.71, "teal"),
        ("Último mês", 27.07, "orange"),
        ("SARIMA", 27.59, "blue"),
        ("INGARCH binomial negativa", 27.66, "muted"),
    ]
    x0, max_width, max_scale = 3.38, 5.85, 32.0
    for i, (label, value, color_key) in enumerate(chart):
        y = 1.95 + i * 0.90
        s.text(0.85, y + 0.05, 2.42, 0.35, [("ChartLabel", label)])
        s.rect(x0, y, max_width, 0.48, "mint")
        s.rect(x0, y, max_width * value / max_scale, 0.48, color_key)
        s.text(x0 + max_width + 0.17, y + 0.04, 1.05, 0.36, [("ChartNumber", f"{value:.2f}")])
    s.rect(10.17, 1.92, 2.42, 3.80, "pale_orange")
    s.text(10.38, 2.17, 2.00, 3.22, [
        ("CardTitle", "Leitura simples"),
        ("BodySmall", "O SVR errou em média cerca de 25 notificações por mês."),
        ("BodySmall", "Ele teve vantagem sobre o último mês, mas a faixa de incerteza ainda inclui empate."),
    ])
    s.text(0.88, 5.95, 11.5, 0.47, [("BodySmall", "Outra medida, o RMSE, também ficou perto: SVR 35,31 e SARIMA 35,39. O SVR subestimou em média 6,44 notificações.")])
    s.footer(6)
    slides.append(s)

    # 7 — Window sensitivity
    s = Slide("Janelas de teste")
    slide_title(s, "O ranking muda quando muda o período", "Por isso, um resultado isolado não escolhe um vencedor definitivo.", section="Resultados")
    windows = [
        (0.75, "50% treino", "SVR liderou", "MAE 24,71", "Último mês: 27,07", "2017–2026", "mint"),
        (4.77, "60% treino", "SVR liderou", "MAE 21,92", "Último mês: 23,72", "2019–2026", "pale_blue"),
        (8.79, "70% treino", "ETS liderou", "MAE 23,18", "SVR: 24,09 • último mês: 24,15", "2021–2026", "pale_orange"),
    ]
    for x, title, winner, best, baseline, period, fill in windows:
        s.rect(x, 1.95, 3.75, 3.50, fill)
        s.text(x + 0.20, 2.22, 3.35, 0.42, [("Eyebrow", title)])
        s.text(x + 0.20, 2.84, 3.35, 0.48, [("CardTitle", winner)])
        s.text(x + 0.20, 3.52, 3.35, 0.48, [("Metric", best.replace("MAE ", ""))])
        s.text(x + 0.20, 4.15, 3.35, 0.48, [("BodySmall", f"{best} • {baseline}")])
        s.text(x + 0.20, 4.83, 3.35, 0.30, [("MetricLabel", f"Teste: {period}")])
    s.rect(0.75, 5.80, 11.79, 0.77, "deep")
    s.text(0.98, 5.98, 11.29, 0.38, [("BodyWhiteSmall", "As janelas compartilham meses. São comparações de sensibilidade, não testes independentes.")])
    s.footer(7)
    slides.append(s)

    # 8 — INGARCH diagnostics
    s = Slide("Contagem e variação")
    slide_title(s, "Por que testar a binomial negativa?", "Uma checagem mostrou que o Poisson não acompanhou bem as oscilações.", section="Resultados")
    s.text(0.85, 1.72, 11.4, 0.42, [("Body", "Dispersão dos erros comparada com o que cada modelo esperava (perto de 1 é mais alinhado):")])
    rows = [("INGARCH Poisson", 12.95, "red"), ("INGARCH binomial negativa", 0.99, "teal")]
    x0, max_width, max_scale = 3.85, 5.65, 14.0
    for i, (label, value, color_key) in enumerate(rows):
        y = 2.65 + i * 1.18
        s.text(0.88, y + 0.08, 2.77, 0.48, [("ChartLabel", label)])
        s.rect(x0, y, max_width, 0.62, "mint")
        s.rect(x0, y, max(0.12, max_width * value / max_scale), 0.62, color_key)
        s.text(x0 + max_width + 0.19, y + 0.08, 1.12, 0.42, [("ChartNumber", f"{value:.2f}")])
    s.rect(0.88, 5.25, 11.62, 1.03, "pale_orange")
    s.text(1.10, 5.44, 11.18, 0.65, [("BodySmall", "A binomial negativa representou melhor a variação, mas não reduziu claramente o erro médio frente ao método simples do último mês.")])
    s.footer(8)
    slides.append(s)

    # 9 — Conclusions and limits
    s = Slide("Conclusões")
    slide_title(s, "O que podemos dizer hoje", "Resultado útil para orientar os próximos passos, ainda sem um campeão confirmado.", section="Conclusões")
    bullets = [
        ("1", "SVR foi o melhor resultado pontual em 50/50 e 60/40; em 70/30, ETS ficou na frente."),
        ("2", "As diferenças contra o método do último mês não foram conclusivas."),
        ("3", "INGARCH binomial negativa acompanhou melhor a variação dos dados que a versão Poisson."),
    ]
    for i, (n, value) in enumerate(bullets):
        y = 1.86 + i * 1.12
        s.rect(0.85, y, 0.56, 0.56, "teal", radius=True)
        s.text(1.00, y + 0.13, 0.26, 0.26, [("CenterTitle", n)])
        s.text(1.70, y - 0.02, 10.60, 0.78, [("Body", value)])
    s.rect(0.85, 5.45, 11.60, 0.86, "deep")
    s.text(1.08, 5.62, 11.12, 0.50, [("BodyWhiteSmall", "Limites: só notificações, sem clima ou população; previsão de um mês à frente; períodos de teste sobrepostos.")])
    s.footer(9)
    slides.append(s)

    # 10 — Next steps
    s = Slide("Próximos passos")
    slide_title(s, "Próximos passos", "Continuar com o que está disponível e validar antes de ampliar as conclusões.", section="Encerramento")
    next_items = [
        ("Confirmar os dados", "Entender atrasos e revisões das notificações com a fonte."),
        ("Acompanhar meses novos", "Verificar se o padrão se mantém em dados que ainda não entraram nos testes."),
        ("Ampliar os modelos de contagem", "Comparar outras formas de representar sazonalidade e dependência temporal."),
        ("Adicionar clima depois", "Só quando chuva e temperatura estiverem alinhadas ao que já seria conhecido na previsão."),
    ]
    for i, (heading, body) in enumerate(next_items):
        col, row = i % 2, i // 2
        x, y = 0.82 + col * 6.00, 1.82 + row * 1.95
        fill = "mint" if i in (0, 3) else "pale_blue"
        s.card(x, y, 5.58, 1.52, fill, heading, [body], body_style="BodySmall")
    s.text(0.85, 6.00, 11.5, 0.38, [("Tiny", "Modelos de contagem: Ferland et al. (2006); Zhu (2011). Análises e arquivos: docs/revisao_e_protocolo.md e results/.")])
    s.footer(10)
    slides.append(s)

    return slides


def write_odp(slides: list[Slide], color_names: dict[str, str]) -> None:
    page_style = (
        '<style:page-layout style:name="PM1"><style:page-layout-properties '
        'fo:page-width="13.333in" fo:page-height="7.500in" '
        'style:print-orientation="landscape"/></style:page-layout>'
    )
    auto_styles, _ = make_automatic_styles()
    pages = []
    for index, slide in enumerate(slides, 1):
        pages.append(
            f'<draw:page draw:name="Slide {index}" draw:style-name="dp1" '
            f'draw:master-page-name="Default">{"".join(slide.shapes)}</draw:page>'
        )
    content = (
        f'<?xml version="1.0" encoding="UTF-8"?>'
        f'<office:document-content {PREFIXES} office:version="1.3">'
        f'{auto_styles}<office:body><office:presentation>{"".join(pages)}'
        '</office:presentation></office:body></office:document-content>'
    )
    styles = (
        f'<?xml version="1.0" encoding="UTF-8"?>'
        f'<office:document-styles {PREFIXES} office:version="1.3">'
        '<office:font-face-decls><style:font-face style:name="Liberation Sans" '
        'svg:font-family="Liberation Sans"/></office:font-face-decls>'
        f'<office:automatic-styles>{page_style}</office:automatic-styles>'
        '<office:master-styles><style:master-page style:name="Default" '
        'style:page-layout-name="PM1"/></office:master-styles></office:document-styles>'
    )
    meta = (
        f'<?xml version="1.0" encoding="UTF-8"?>'
        f'<office:document-meta {PREFIXES} office:version="1.3"><office:meta>'
        '<dc:title>Previsão mensal de notificações de malária em Breves</dc:title>'
        '<dc:language>pt-BR</dc:language><meta:generator>Bioinfo research presentation</meta:generator>'
        '</office:meta></office:document-meta>'
    )
    settings = (
        f'<?xml version="1.0" encoding="UTF-8"?>'
        f'<office:document-settings {PREFIXES} office:version="1.3">'
        '<office:settings/></office:document-settings>'
    )
    manifest = (
        f'<?xml version="1.0" encoding="UTF-8"?>'
        f'<manifest:manifest {PREFIXES} manifest:version="1.3">'
        '<manifest:file-entry manifest:full-path="/" manifest:media-type="application/vnd.oasis.opendocument.presentation"/>'
        '<manifest:file-entry manifest:full-path="content.xml" manifest:media-type="text/xml"/>'
        '<manifest:file-entry manifest:full-path="styles.xml" manifest:media-type="text/xml"/>'
        '<manifest:file-entry manifest:full-path="meta.xml" manifest:media-type="text/xml"/>'
        '<manifest:file-entry manifest:full-path="settings.xml" manifest:media-type="text/xml"/>'
        '</manifest:manifest>'
    )
    with zipfile.ZipFile(ODP_PATH, "w") as archive:
        archive.writestr("mimetype", "application/vnd.oasis.opendocument.presentation", compress_type=zipfile.ZIP_STORED)
        archive.writestr("content.xml", content)
        archive.writestr("styles.xml", styles)
        archive.writestr("meta.xml", meta)
        archive.writestr("settings.xml", settings)
        archive.writestr("META-INF/manifest.xml", manifest)


def main() -> None:
    global COLOR_NAMES
    auto_styles, COLOR_NAMES = make_automatic_styles()
    # Ensure the style table and the shapes share the exact same color mapping.
    del auto_styles
    slides = build_slides()
    write_odp(slides, COLOR_NAMES)
    print(f"Apresentação editável criada: {ODP_PATH}")


if __name__ == "__main__":
    COLOR_NAMES: dict[str, str] = {}
    main()
