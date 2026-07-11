"""Render the bundled User Guide to ``lt2_gui/assets/LT2_Help.pdf``.

Run as a module to refresh the PDF after editing :mod:`lt2_gui.help_text`::

    python -m lt2_gui.build_help_pdf

The output path is fixed so the Help tab can always find the file
relative to the installed package.  No new dependency is added —
matplotlib's PdfPages backend is already used by the figure exporter.
"""
from __future__ import annotations

import os
import textwrap
from pathlib import Path

from .__version__ import APP_NAME, APP_VERSION, AUTHOR_NAME
from .help_text import help_plain_text
from .paths import resource_path


# When building the PDF at package time we always write into the source tree.
HELP_PDF_PATH = Path(__file__).resolve().parent / "assets" / "LT2_Help.pdf"


def build_help_pdf(out_path: str | os.PathLike[str] | None = None) -> Path:
    """Write the Help PDF to *out_path* (default :data:`HELP_PDF_PATH`)."""
    target = Path(out_path) if out_path is not None else HELP_PDF_PATH
    target.parent.mkdir(parents=True, exist_ok=True)

    # Defer the matplotlib import so importing this module is cheap.
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib.backends.backend_pdf import PdfPages
    import matplotlib.pyplot as plt

    text = help_plain_text()
    # Letter-size page with generous margins.
    page_w, page_h = 8.5, 11.0
    margin_x, margin_y = 0.7, 0.85
    max_chars = 90
    font_size = 9
    line_h_inches = 0.16
    max_lines = int((page_h - 2 * margin_y) / line_h_inches)

    wrapped: list[str] = []
    for raw_line in text.splitlines():
        if not raw_line.strip():
            wrapped.append("")
            continue
        wrapped.extend(
            textwrap.wrap(
                raw_line,
                width=max_chars,
                replace_whitespace=False,
                drop_whitespace=False,
            )
            or [""]
        )

    with PdfPages(str(target)) as pdf:
        for chunk_start in range(0, len(wrapped), max_lines):
            chunk = wrapped[chunk_start: chunk_start + max_lines]
            fig = plt.figure(figsize=(page_w, page_h))
            ax = fig.add_axes([0, 0, 1, 1])
            ax.set_xlim(0, page_w)
            ax.set_ylim(0, page_h)
            ax.axis("off")
            y = page_h - margin_y
            page_no = chunk_start // max_lines + 1
            ax.text(
                margin_x,
                page_h - 0.4,
                f"{APP_NAME} v{APP_VERSION} — User Guide",
                fontsize=10,
                fontweight="bold",
            )
            for line in chunk:
                ax.text(margin_x, y, line, fontsize=font_size,
                        family="DejaVu Sans Mono")
                y -= line_h_inches
            ax.text(
                page_w - margin_x,
                0.45,
                f"page {page_no}",
                fontsize=8,
                ha="right",
                color="gray",
            )
            pdf.savefig(fig)
            plt.close(fig)
        info = pdf.infodict()
        info["Title"] = f"{APP_NAME} v{APP_VERSION} — User Guide"
        info["Author"] = AUTHOR_NAME
        info["Subject"] = "Luminescence thermometry benchmark — help"
    return target


if __name__ == "__main__":
    out = build_help_pdf()
    print(f"Wrote {out}  ({out.stat().st_size:,} bytes)")
