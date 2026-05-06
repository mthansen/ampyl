#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import unittest

import ampyl


class TestThreeBodyInteractionScheme(unittest.TestCase):
    def test_kkpi_pion_spectator_cutoff_support(self):
        mpi = 1.0
        mk = 2.5
        pion = ampyl.flavor.Particle(
            mass=mpi, spin=0.0, flavor="pi",
            isospin_multiplet=True, isospin=1.0)
        kaon = ampyl.flavor.Particle(
            mass=mk, spin=0.0, flavor="K",
            isospin_multiplet=True, isospin=0.5)

        fc_kkpi = ampyl.flavor.FlavorChannel(
            3, particles=[kaon, kaon, pion], isospin=2.0)
        fcs = ampyl.flavor.FlavorChannelSpace(fc_list=[fc_kkpi])
        tbis = ampyl.spaces.ThreeBodyInteractionScheme(fcs=fcs)

        pion_spectator = fcs.sc_list_sorted[0]
        self.assertEqual(pion_spectator.flavors_indexed[0], "pi")
        self.assertEqual(pion_spectator.flavors_indexed[1:], ["K", "K"])

        ESQmin = 4.0*mk**2 - 4.0*mpi**2
        thresholdSQ = 4.0*mk**2
        alpha, beta = tbis.scheme_data[0]

        self.assertAlmostEqual(pion_spectator.thresholdSQ, thresholdSQ)
        self.assertAlmostEqual(pion_spectator.ESQmin, ESQmin)
        self.assertAlmostEqual(pion_spectator.ESQMIN, ESQmin)
        self.assertAlmostEqual(pion_spectator.alpha, alpha)
        self.assertAlmostEqual(pion_spectator.beta, beta)
        self.assertEqual(pion_spectator.scheme_data, [alpha, beta])

        self.assertEqual(tbis.thresholdSQs,
                         [sc.thresholdSQ for sc in fcs.sc_list_sorted])
        self.assertEqual(tbis.ESQmins,
                         [sc.ESQmin for sc in fcs.sc_list_sorted])
        self.assertEqual(tbis.scheme_data,
                         [sc.scheme_data for sc in fcs.sc_list_sorted])
        self.assertFalse(hasattr(tbis, 'ESQmin'))
        self.assertFalse(hasattr(tbis, 'ESQMIN'))
        self.assertAlmostEqual(tbis.ESQmins[0], ESQmin)
        self.assertAlmostEqual(alpha, 3.0 - 4.0*mpi**2/mk**2)
        self.assertAlmostEqual(beta, 0.0)
        self.assertEqual(
            ampyl.kinematic_functions.H(ESQmin, 2.0*mk, alpha, beta),
            0.0)
        self.assertEqual(
            ampyl.kinematic_functions.H(thresholdSQ, 2.0*mk, alpha, beta),
            1.0)

    def test_external_scheme_data_overrides_channel_defaults(self):
        fcs = ampyl.flavor.FlavorChannelSpace(
            fc_list=[ampyl.flavor.FlavorChannel(3)])
        scheme_data = [[-0.7, 0.0]]

        tbis = ampyl.spaces.ThreeBodyInteractionScheme(
            fcs=fcs, scheme_data=scheme_data)

        self.assertIs(tbis.scheme_data, scheme_data)
        self.assertEqual(tbis.ESQmins,
                         [0.25*(1.0+scheme_data[0][0])
                          * tbis.thresholdSQs[0]])


if __name__ == "__main__":
    unittest.main()
