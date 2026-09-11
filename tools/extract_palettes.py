#!/usr/bin/env python3
"""Regenerate src/scp/data/palette_list.json from SCP's palette_list.rda.

No R required: `_rds.py` parses R's XDR serialization directly (bzip2/gzip/xz
outer layer, then the RDX2/RDX3 body). Only enough of the format is
implemented to read a named list of character vectors with attributes, which
is exactly what palette_list is.

    git clone --depth 1 https://github.com/zhanghao-njmu/SCP.git /tmp/SCP
    python tools/extract_palettes.py /tmp/SCP/data/palette_list.rda
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _rds import load_rda  # noqa: E402

OUT = Path(__file__).parent.parent / "src" / "scp" / "data" / "palette_list.json"


def main(rda: str) -> None:
    obj = load_rda(rda)["palette_list"]
    names = obj["attrs"]["names"]
    out = {}
    for nm, v in zip(names, obj["values"]):
        if isinstance(v, dict):
            out[nm] = {"type": v["attrs"].get("type", ["discrete"])[0], "colors": v["values"]}
        else:
            out[nm] = {"type": "discrete", "colors": v}
    OUT.write_text(json.dumps(out, indent=1))
    n_d = sum(1 for v in out.values() if v["type"] == "discrete")
    print(f"wrote {len(out)} palettes ({n_d} discrete, {len(out) - n_d} continuous) -> {OUT}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "/tmp/SCP/data/palette_list.rda")
