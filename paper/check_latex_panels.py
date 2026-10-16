"""Verify the separate panels referenced by the manuscript against their manifest."""
import json
from pathlib import Path
from prepare_identification_figures import ROOT, digest


def main():
    record = json.loads((ROOT / 'results/tables/paper_latex_panels_manifest.json').read_text())
    assert record['status'] == 'generated_from_verified_frozen_evidence'
    assert record['generator_sha256'] == digest(ROOT / 'paper/prepare_latex_panels.py')
    assert record['rendering_helper_sha256'] == digest(ROOT / 'paper/prepare_identification_figures.py')
    for relative, expected in record['source_sha256'].items():
        assert digest(ROOT / relative) == expected, relative
    tex = (ROOT / 'paper/main.tex').read_text()
    outputs = record['output_sha256']
    assert len(outputs) == 8
    for relative, expected in outputs.items():
        assert digest(ROOT / relative) == expected, relative
        assert '{' + Path(relative).name + '}' in tex, relative
    assert tex.count('\\subfloat[') == 8
    print(json.dumps(dict(status='verified', panels=8, operation='read_only_no_regeneration')))


if __name__ == '__main__':
    main()
