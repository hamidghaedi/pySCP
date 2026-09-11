"""Bundled data.

``palette_list.json`` holds all 229 palettes from ``SCP/data/palette_list.rda``,
extracted by parsing R's XDR serialization directly (no R installation was
involved, so the values are the package's own, not a re-derivation from the
upstream palette packages).  Each entry keeps the ``type`` attribute R used,
because ``palette_scp`` branches on it.

Regenerate with ``tools/extract_palettes.py`` if the R package updates.
"""
