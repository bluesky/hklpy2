"""
Resolve issues involving UB matrix and cahkl().

* Issue 183: UB matrix reported by pa() and configuration should not be different.
* Issue 186: UB matrix computed from two reflections should match known value.
"""

import ast
import pathlib
from contextlib import nullcontext as does_not_raise

import numpy as np
import pytest
import yaml

from ..diffract import creator
from ..user import cahkl
from ..user import pa
from ..user import set_diffractometer

THIS_DIR = pathlib.Path(__file__).parent


@pytest.mark.parametrize(
    "specs, cfg_file, reals, hkl1, context",
    [
        pytest.param(
            dict(geometry="E4CV", solver="hkl_soleil"),
            THIS_DIR / "fourc-i183.yml",
            dict(omega=0, chi=0, phi=40, tth=0),
            (0.25, 0.25, 2 * 3.8995 / 20.5311),
            does_not_raise(),
            id="Ok: issue 183",
        ),
        pytest.param(
            dict(geometry="E4CV", solver="hkl_soleil"),
            THIS_DIR / "e4cv_orient.yml",
            dict(omega=0, chi=0, phi=40, tth=0),
            (0, 0.5, 0),
            does_not_raise(),
            id="Ok: e4cv_orient",
        ),
        pytest.param(
            dict(geometry="E4CV", solver="hkl_soleil"),
            THIS_DIR / "fourc-configuration.yml",
            dict(omega=0, chi=0, phi=40, tth=0),
            (0, 0.5, 0),
            does_not_raise(),
            id="Ok: fourc-configuration",
        ),
    ],
)
def test_issue183(specs, cfg_file, reals, hkl1, context, capsys):
    with context:
        with open(cfg_file) as fp:
            cfg = yaml.load(fp.read(), Loader=yaml.SafeLoader)

        sim = creator(**specs)
        sim.move_reals(reals)
        sim.restore(cfg_file)

        assert np.allclose(
            np.array(cfg["samples"][cfg["sample_name"]]["UB"]),
            np.array(sim.core.sample.UB),
            atol=0.01,
        )

        assert np.allclose(
            np.array(sim.core.sample.UB),
            np.array(sim.configuration["samples"][sim.core.sample.name]["UB"]),
            atol=0.01,
        )

        # Compare with value reported in pa()
        set_diffractometer(sim)
        pa()
        out, err = capsys.readouterr()
        assert err == ""
        assert "UB=" in out

        # pick out the one line with the UB matrix
        line = out[out.find("UB=") + 3 :].splitlines()[0]
        ub = ast.literal_eval(line)
        assert np.allclose(
            np.array(sim.core.sample.UB),
            np.array(ub),
            atol=0.01,
        )

        # should not raise NoForwardSolutions or return ()
        cahkl(*hkl1)
