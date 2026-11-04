"""Frozen-trace checks and final-size layout of the publication figure."""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import pytest

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'paper'))
import prepare_oe_transient_figure as p


@pytest.fixture(scope='module')
def context():
    return p.prepare()


def test_final_size_all_traces_visible_and_no_embedded_panel_letters(context):
    spec,refs,traces,checks,sources=context
    panels,limits=p.make_panels(spec,refs,traces)
    assert len(panels)==10 and len(checks)==18
    assert limits['error_limits_deg_s']==[-.12,.12]
    for name,fig in panels.items():
        fig.canvas.draw();renderer=fig.canvas.get_renderer()
        if name=='legend':
            boxes=[fig.legends[0].get_window_extent(renderer)]
        else:
            ax=fig.axes[0]
            assert fig.get_figwidth()==2.30
            assert ax.get_title()=='' and ax.get_legend() is None
            assert ax.yaxis.label.get_fontsize()==8
            if not name.endswith('reference'):
                assert list(ax.get_ylim())==limits['error_limits_deg_s']
            for line in ax.lines:
                y=np.asarray(line.get_ydata(),float)
                assert y.min()>=ax.get_ylim()[0] and y.max()<=ax.get_ylim()[1]
            boxes=[ax.get_tightbbox(renderer)]
        for box in boxes:
            assert box.x0>=-1 and box.y0>=-1
            assert box.x1<=fig.bbox.width+1 and box.y1<=fig.bbox.height+1
        p.plt.close(fig)
    tex=p.latex_fragment(spec)
    assert tex.count('\\subfloat[')==3
    assert tex.count('\\includegraphics')==10
    assert 'width=\\linewidth' in tex


def test_missing_or_edited_trace_is_rejected(context):
    spec=context[0]
    frame=pd.read_csv(ROOT/'results/tables/paper_oe_transient_benchmark_traces.csv',float_precision='round_trip')
    metrics=pd.read_csv(ROOT/'results/tables/paper_oe_transient_benchmark_metrics.csv',float_precision='round_trip')
    with pytest.raises(ValueError,match='Incomplete or duplicate'):
        p.validate_traces(frame.iloc[1:],metrics,spec)
    changed=frame.copy();changed.loc[100,'yaw_error_deg_s']+=.01
    with pytest.raises(AssertionError):
        p.validate_traces(changed,metrics,spec)
    missing=frame[frame.method!='Global']
    with pytest.raises(ValueError,match='Missing or unexpected'):
        p.validate_traces(missing,metrics,spec)
