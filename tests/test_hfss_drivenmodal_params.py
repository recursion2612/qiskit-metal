# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""The legacy HFSS renderer's driven-modal setup edits and network-data
reads, as used by ScatteringImpedanceSim and tutorials 4.16 / 4.17.

The pyEPR objects are mocks: this checks what the renderer asks of pyEPR and
how it reports problems, not the result in Ansys.
"""

import importlib.util
import os
import unittest
from unittest.mock import MagicMock

import matplotlib.pyplot as plt
import numpy as np

from qiskit_metal import Dict, designs
from qiskit_metal.renderers.renderer_ansys.hfss_renderer import QHFSSRenderer

HAS_PYEPR = importlib.util.find_spec("pyEPR") is not None


# Property names AEDT 2025 R2 shows for a driven setup, without and with ports.
PROPS_NO_PORTS = ["Name", "Enabled", "Passes", "Delta Energy", "Solution Freq"]
PROPS_PORTS = ["Name", "Enabled", "Passes", "Delta S", "Solution Freq"]


def _renderer(props=PROPS_PORTS):
    """Renderer on a mocked pyEPR connection. Edits go to
    ``renderer._pinfo.design.get_setup.return_value``, a setup named 'Setup'."""
    renderer = QHFSSRenderer(designs.DesignPlanar(), initiate=False)
    renderer._pinfo = MagicMock()
    renderer._pinfo.design.solution_type = "DrivenModal"
    renderer._pinfo.setup_name = "Setup"
    setup = renderer._pinfo.design.get_setup.return_value
    setup.name = "Setup"
    setup.prop_holder.GetProperties.return_value = props
    return renderer


def _setup(renderer):
    return renderer._pinfo.design.get_setup.return_value


NOMINAL = "Cj='0fF' Lj='10nH'"  # what ScatteringImpedanceSim's vars give
FREQS = np.array([4e9, 5e9, 6e9])
NETWORK = {  # 2-port data: {"S": {(i, j): values}}, as AEDT exports it
    kind: {
        (i, j): np.array([0.1 * i + 0.2j * j, 0.3 - 0.1j * i, 0.5j + 0.01 * j])
        * (1 + "SYZ".index(kind))
        for i in (1, 2)
        for j in (1, 2)
    }
    for kind in "SYZ"
}


def _export_tab(variation, selection, file_format, path, freqs, *rest):
    """ExportNetworkData stand-in: writes the tab file AEDT 2025 R2 writes."""
    data_type = rest[2]
    columns = ["Frequency"] + [
        f"{data_type}[{i},{j}]_{part}"
        for i in (1, 2)
        for j in (1, 2)
        for part in ("Re", "Im")
    ]
    lines = [
        "Row0=Port(LumpPort_LP1_in):Mode(1)\tRow1=Port(LumpPort_LP2_in):Mode(1)\t",
        "".join(f"{c}   \t" for c in columns),
    ]
    for k, f in enumerate(FREQS):
        row = [f"{f:.15E}"]
        for i in (1, 2):
            for j in (1, 2):
                v = NETWORK[data_type][(i, j)][k]
                row += [f"{v.real:.15E}", f"{v.imag:.15E}"]
        lines.append("".join(f"{x} \t" for x in row))
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def _sweep(solutions, solved=(NOMINAL,)):
    """A pyEPR HfssFrequencySweep stand-in for 'Setup : Sweep', solved for
    the design's nominal variation."""
    sweep = MagicMock()
    sweep.solution_name = "Setup : Sweep"
    sweep.parent._solutions = solutions
    sweep.parent.parent.get_nominal_variation.return_value = NOMINAL
    if solutions is not None:
        solutions.ListVariations.return_value = solved
        solutions.ExportNetworkData.side_effect = _export_tab
    return sweep


@unittest.skipUnless(HAS_PYEPR, "pyEPR not installed")
class TestEditDrivenmodalSetup(unittest.TestCase):
    def test_int_freq_ghz_is_accepted(self):
        """ScatteringImpedanceSim.default_setup has freq_ghz=5, an int, which
        was rejected because only a float passed the check."""
        renderer = _renderer()
        with self.assertNoLogs(renderer.logger, level="WARNING"):
            renderer.edit_drivenmodal_setup(Dict(name="Setup", freq_ghz=5))
        self.assertEqual(_setup(renderer).solution_freq, "5GHz")

    def test_property_window_keys_are_set(self):
        renderer = _renderer()
        renderer.edit_drivenmodal_setup(
            Dict(name="Setup", freq_ghz=6.5, max_passes=4, pct_refinement=20)
        )
        setup = _setup(renderer)
        self.assertEqual(setup.solution_freq, "6.5GHz")
        self.assertEqual(setup.passes, 4)
        self.assertEqual(setup.pct_refinement, 20)
        setup._setup_module.EditSetup.assert_not_called()

    def test_max_delta_s_and_min_passes_in_one_edit_setup(self):
        """These three were 'not implemented': the property window has no
        field for min passes or min converged passes, and pyEPR's 'Delta S'
        property is not shown until the design has ports. EditSetup with only
        these fields keeps the others (checked in AEDT 2025 R2)."""
        renderer = _renderer()
        with self.assertNoLogs(renderer.logger, level="WARNING"):
            renderer.edit_drivenmodal_setup(
                Dict(name="Setup", max_delta_s=0.001, min_passes=2, min_converged=3)
            )
        _setup(renderer)._setup_module.EditSetup.assert_called_once_with(
            "Setup",
            [
                "NAME:Setup",
                "MaxDeltaS:=",
                0.001,
                "MinimumPasses:=",
                2,
                "MinimumConvergedPasses:=",
                3,
            ],
        )

    def test_basis_order_through_edit_setup(self):
        """pyEPR's 'Basis Order' property takes the order's name ("Mixed
        Order"); passing -1 there is a COM error on AEDT 2025 R2."""
        renderer = _renderer()
        renderer.edit_drivenmodal_setup(Dict(name="Setup", basis_order=-1))
        _setup(renderer)._setup_module.EditSetup.assert_called_once_with(
            "Setup", ["NAME:Setup", "BasisOrder:=", -1]
        )

    def test_max_delta_s_before_the_design_has_ports(self):
        """Tutorial 4.16 edits the setup pyEPR adds to a new design, before
        render_design() draws the ports. Ansys then converges on Delta Energy,
        ignores MaxDeltaS, and puts its default Max Delta S once there are
        ports, so the value is lost: say so."""
        renderer = _renderer(props=PROPS_NO_PORTS)
        with self.assertLogs(renderer.logger, level="WARNING") as caught:
            renderer.edit_drivenmodal_setup(Dict(name="Setup", max_delta_s=0.001))
        self.assertEqual(len(caught.records), 1, caught.output)
        self.assertIn("after render_design()", caught.output[0])
        _setup(renderer)._setup_module.EditSetup.assert_called_once()

    def test_edits_the_named_setup_of_the_active_design(self):
        """activate_ansys_design() of an existing design leaves pinfo.setup on
        the previous design's setup; the edit must not go there."""
        renderer = _renderer()
        stale = MagicMock()
        renderer._pinfo.setup = stale
        renderer.edit_drivenmodal_setup(Dict(name="Setup", max_passes=3))
        renderer._pinfo.design.get_setup.assert_called_once_with("Setup")
        self.assertEqual(_setup(renderer).passes, 3)
        self.assertNotEqual(stale.passes, 3)

    def test_setup_missing_from_the_active_design(self):
        renderer = _renderer()
        renderer._pinfo.design.get_setup.side_effect = EnvironmentError(
            "Setup Setup not found: ['Setup1']"
        )
        with self.assertLogs(renderer.logger, level="WARNING") as caught:
            renderer.edit_drivenmodal_setup(Dict(name="Setup", max_delta_s=0.01))
        self.assertIn("not found", caught.output[0])

    def test_min_passes_must_be_an_int(self):
        renderer = _renderer()
        with self.assertLogs(renderer.logger, level="WARNING") as caught:
            renderer.edit_drivenmodal_setup(Dict(name="Setup", min_passes=1.5))
        self.assertEqual(len(caught.records), 1, caught.output)
        _setup(renderer)._setup_module.EditSetup.assert_not_called()

    def test_one_warning_for_a_value_of_the_wrong_type(self):
        """A wrong type logged the type warning and then also 'not in
        pinfo.setup', as if the key were unknown."""
        renderer = _renderer()
        with self.assertLogs(renderer.logger, level="WARNING") as caught:
            renderer.edit_drivenmodal_setup(Dict(name="Setup", max_passes=2.5))
        self.assertEqual(len(caught.records), 1, caught.output)
        self.assertIn("should be an int", caught.output[0])

    def test_bool_is_not_a_frequency(self):
        renderer = _renderer()
        with self.assertLogs(renderer.logger, level="WARNING") as caught:
            renderer.edit_drivenmodal_setup(Dict(name="Setup", freq_ghz=True))
        self.assertEqual(len(caught.records), 1, caught.output)


@unittest.skipUnless(HAS_PYEPR, "pyEPR not installed")
class TestAddDrivenmodalSetup(unittest.TestCase):
    def _add(self, props):
        renderer = _renderer()
        created = renderer._pinfo.design.create_dm_setup.return_value
        created.name = "Setup1"
        created.prop_holder.GetProperties.return_value = props
        return renderer, created

    def test_added_before_the_design_has_ports(self):
        """Ansys drops a max_delta_s given before the ports exist (tutorial
        4.16's order); 4.17 adds its setup after rendering and keeps it."""
        renderer, created = self._add(PROPS_NO_PORTS)
        with self.assertLogs(renderer.logger, level="WARNING") as caught:
            setup = renderer.add_drivenmodal_setup(name="Setup1", max_delta_s=0.005)
        self.assertIs(setup, created)
        self.assertIn("max_delta_s=0.005", caught.output[0])

    def test_added_with_ports(self):
        renderer, created = self._add(PROPS_PORTS)
        with self.assertNoLogs(renderer.logger, level="WARNING"):
            setup = renderer.add_drivenmodal_setup(name="Setup1", max_delta_s=0.005)
        self.assertIs(setup, created)
        kwargs = renderer._pinfo.design.create_dm_setup.call_args.kwargs
        self.assertEqual(kwargs["max_delta_s"], 0.005)

    def test_property_names_unavailable(self):
        renderer, created = self._add(PROPS_PORTS)
        created.prop_holder.GetProperties.side_effect = Exception("busy")
        with self.assertNoLogs(renderer.logger, level="WARNING"):
            self.assertIs(renderer.add_drivenmodal_setup(name="Setup1"), created)


@unittest.skipUnless(HAS_PYEPR, "pyEPR not installed")
class TestGetParams(unittest.TestCase):
    def test_exports_the_nominal_variation(self):
        """pyEPR exports with an empty variation key, which AEDT 2025 R2
        refuses (COM error -2147024381) once the design has variables, as
        every ScatteringImpedanceSim.run_sim() design does (Lj, Cj)."""
        renderer = _renderer()
        renderer.current_sweep = sweep = _sweep(MagicMock())

        freqs, curves, params = renderer.get_params(["S11", "S21"])

        export = sweep.parent._solutions.ExportNetworkData
        export.assert_called_once()
        args = export.call_args.args
        self.assertEqual(args[0], NOMINAL)
        self.assertEqual(args[1], "Setup : Sweep")
        self.assertEqual(args[2], 2)  # tab-separated
        self.assertEqual(args[7], "S")
        sweep.get_network_data.assert_not_called()  # pyEPR's empty-key export
        np.testing.assert_array_equal(freqs, FREQS)
        self.assertEqual(list(params.columns), ["S11", "S21"])
        np.testing.assert_allclose(params.index.to_numpy(), [4.0, 5.0, 6.0])
        np.testing.assert_allclose(params["S21"].to_numpy(), NETWORK["S"][(2, 1)])
        np.testing.assert_allclose(curves[0], NETWORK["S"][(1, 1)])

    def test_one_export_per_matrix(self):
        renderer = _renderer()
        renderer.current_sweep = sweep = _sweep(MagicMock())
        _, curves, params = renderer.get_params(["Z21", "Y11", "Y12"])
        kinds = [
            c.args[7] for c in sweep.parent._solutions.ExportNetworkData.call_args_list
        ]
        self.assertEqual(sorted(kinds), ["Y", "Z"])
        self.assertEqual(list(params.columns), ["Z21", "Y11", "Y12"])
        np.testing.assert_allclose(curves[0], NETWORK["Z"][(2, 1)])
        np.testing.assert_allclose(curves[2], NETWORK["Y"][(1, 2)])

    def test_export_file_is_removed(self):
        renderer = _renderer()
        renderer.current_sweep = sweep = _sweep(MagicMock())
        renderer.get_params(["S21"])
        path = sweep.parent._solutions.ExportNetworkData.call_args.args[3]
        self.assertFalse(os.path.exists(os.path.dirname(path)))

    def test_no_parameters(self):
        renderer = _renderer()
        renderer.current_sweep = _sweep(MagicMock())
        with self.assertRaisesRegex(ValueError, "S11"):
            renderer.get_params()

    def test_no_analyzed_sweep(self):
        """Used to fail with UnboundLocalError (freqs was never assigned)."""
        renderer = _renderer()
        with self.assertRaisesRegex(RuntimeError, "analyze_sweep"):
            renderer.get_params(["S21"])

    def test_sweep_not_solved_for_the_current_variation(self):
        """ListVariations is empty for a sweep that was never solved, and
        after any change to the design since the solve (AEDT 2025 R2); the
        export would fail with a COM error."""
        renderer = _renderer()
        renderer.current_sweep = sweep = _sweep(MagicMock(), solved=())
        with self.assertRaisesRegex(
            RuntimeError, "'Setup : Sweep' has no solution for the design's current"
        ):
            renderer.get_params(["S21"])
        sweep.parent._solutions.ExportNetworkData.assert_not_called()

    def test_solved_for_another_variation_only(self):
        renderer = _renderer()
        renderer.current_sweep = _sweep(MagicMock(), solved=("Cj='0fF' Lj='12nH'",))
        with self.assertRaisesRegex(RuntimeError, "Lj='12nH'"):
            renderer.get_params(["S21"])

    def test_closed_connection(self):
        """pyEPR's disconnect() sets every COM handle it made to None."""
        renderer = _renderer()
        renderer.current_sweep = _sweep(None)
        with self.assertRaisesRegex(RuntimeError, "closed"):
            renderer.get_params(["S21"])

    def test_export_error_names_the_sweep_and_variation(self):
        renderer = _renderer()
        renderer.current_sweep = sweep = _sweep(MagicMock())
        com_error = Exception(
            -2147352567,
            "Exception occurred.",
            (0, None, None, None, 0, -2147024381),
            None,
        )
        sweep.parent._solutions.ExportNetworkData.side_effect = com_error
        with self.assertRaisesRegex(
            RuntimeError, "'Setup : Sweep' for variation"
        ) as ctx:
            renderer.get_params(["S21"])
        self.assertIs(ctx.exception.__cause__, com_error)

    def test_variation_list_unavailable_does_not_block_the_export(self):
        renderer = _renderer()
        renderer.current_sweep = sweep = _sweep(MagicMock())
        sweep.parent._solutions.ListVariations.side_effect = Exception("busy")
        _, _, params = renderer.get_params(["S21"])
        self.assertEqual(params.shape, (3, 1))

    def test_plot_params_returns_frame_and_figure(self):
        renderer = _renderer()
        renderer.current_sweep = _sweep(MagicMock())
        params, fig = renderer.plot_params(["S21"])
        try:
            self.assertEqual(list(params.columns), ["S21"])
            self.assertEqual(len(fig.axes), 2)
        finally:
            plt.close(fig)


if __name__ == "__main__":
    unittest.main(verbosity=2)
