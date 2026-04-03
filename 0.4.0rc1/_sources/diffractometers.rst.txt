.. author: make_geometries_doc.py
.. date: 2025-03-11 13:31:21.554744

.. _geometries:

===============
Diffractometers
===============

.. index:: diffractometers
.. index:: geometries


Tables are provided for the different geometries (sorted by number of real axes)
and then, for each geometry, the calculation engines, modes of operation, pseudo
axes required, and any additional parameters required by the
:meth:`~hklpy2.backends.base.SolverBase.mode`.  The mode defines which axes will
be computed, which will be held constant, and any relationships between axes.

.. _geometries.number_of_reals:

Geometries, by number of real axes
==================================

.. index:: geometries; by number of reals

The different diffractometer geometries are distinguished,
primarily, by the number of real axes.  This
table is sorted first by the number of real axes, then by
solver and geometry names.

====== ============================================================================================
#reals solver, geometry
====== ============================================================================================
2      :ref:`th_tth, TH TTH Q <geometries-th_tth-th-tth-q>`
4      :ref:`hkl_soleil, E4CH <geometries-hkl_soleil-e4ch>`
4      :ref:`hkl_soleil, E4CV <geometries-hkl_soleil-e4cv>`
4      :ref:`hkl_soleil, K4CV <geometries-hkl_soleil-k4cv>`
4      :ref:`hkl_soleil, PETRA3 P23 4C <geometries-hkl_soleil-petra3-p23-4c>`
4      :ref:`hkl_soleil, SOLEIL MARS <geometries-hkl_soleil-soleil-mars>`
4      :ref:`hkl_soleil, SOLEIL SIXS MED1+2 <geometries-hkl_soleil-soleil-sixs-med1+2>`
4      :ref:`hkl_soleil, ZAXIS <geometries-hkl_soleil-zaxis>`
5      :ref:`hkl_soleil, ESRF ID01 PSIC <geometries-hkl_soleil-esrf-id01-psic>`
5      :ref:`hkl_soleil, SOLEIL SIXS MED2+2 <geometries-hkl_soleil-soleil-sixs-med2+2>`
5      :ref:`hkl_soleil, SOLEIL SIXS MED2+3 v2 <geometries-hkl_soleil-soleil-sixs-med2+3-v2>`
6      :ref:`hkl_soleil, APS POLAR <geometries-hkl_soleil-aps-polar>`
6      :ref:`hkl_soleil, E6C <geometries-hkl_soleil-e6c>`
6      :ref:`hkl_soleil, K6C <geometries-hkl_soleil-k6c>`
6      :ref:`hkl_soleil, PETRA3 P09 EH2 <geometries-hkl_soleil-petra3-p09-eh2>`
6      :ref:`hkl_soleil, SOLEIL NANOSCOPIUM ROBOT <geometries-hkl_soleil-soleil-nanoscopium-robot>`
6      :ref:`hkl_soleil, SOLEIL SIRIUS KAPPA <geometries-hkl_soleil-soleil-sirius-kappa>`
6      :ref:`hkl_soleil, SOLEIL SIRIUS TURRET <geometries-hkl_soleil-soleil-sirius-turret>`
6      :ref:`hkl_soleil, SOLEIL SIXS MED2+3 <geometries-hkl_soleil-soleil-sixs-med2+3>`
7      :ref:`hkl_soleil, PETRA3 P23 6C <geometries-hkl_soleil-petra3-p23-6c>`
====== ============================================================================================

.. _geometries.summary_tables:

Available Solver Geometry Tables
================================

.. index:: geometries; tables

.. seealso:: :func:`hklpy2.user.solver_summary()`

.. _geometries-hkl_soleil-aps-polar:

solver='hkl_soleil', geometry='APS POLAR'
-----------------------------------------

.. index:: geometries; hkl_soleil; APS POLAR

====== ================================= ========= =============================== ===================== ===============
engine mode                              pseudo(s) real(s)                         writable(s)           extra(s)
====== ================================= ========= =============================== ===================== ===============
hkl    4-circles constant phi horizontal h, k, l   tau, mu, chi, phi, gamma, delta mu, chi, gamma
hkl    zaxis + alpha-fixed               h, k, l   tau, mu, chi, phi, gamma, delta mu, gamma, delta
hkl    zaxis + beta-fixed                h, k, l   tau, mu, chi, phi, gamma, delta tau, gamma, delta
hkl    zaxis + alpha=beta                h, k, l   tau, mu, chi, phi, gamma, delta tau, mu, gamma, delta
hkl    4-circles bissecting horizontal   h, k, l   tau, mu, chi, phi, gamma, delta mu, chi, phi, gamma
hkl    4-circles constant mu horizontal  h, k, l   tau, mu, chi, phi, gamma, delta chi, phi, gamma
hkl    4-circles constant chi horizontal h, k, l   tau, mu, chi, phi, gamma, delta mu, phi, gamma
hkl    lifting detector tau              h, k, l   tau, mu, chi, phi, gamma, delta tau, gamma, delta
hkl    lifting detector mu               h, k, l   tau, mu, chi, phi, gamma, delta mu, gamma, delta
hkl    lifting detector chi              h, k, l   tau, mu, chi, phi, gamma, delta chi, gamma, delta
hkl    lifting detector phi              h, k, l   tau, mu, chi, phi, gamma, delta phi, gamma, delta
hkl    psi constant horizontal           h, k, l   tau, mu, chi, phi, gamma, delta mu, chi, phi, gamma   h2, k2, l2, psi
hkl    psi constant vertical             h, k, l   tau, mu, chi, phi, gamma, delta tau, chi, phi, delta  h2, k2, l2, psi
psi    psi_vertical                      psi       tau, mu, chi, phi, gamma, delta mu, chi, phi, delta   h2, k2, l2
====== ================================= ========= =============================== ===================== ===============

.. _geometries-hkl_soleil-e4ch:

solver='hkl_soleil', geometry='E4CH'
------------------------------------

.. index:: geometries; hkl_soleil; E4CH

========= ================== ================== ==================== ==================== ===============
engine    mode               pseudo(s)          real(s)              writable(s)          extra(s)
========= ================== ================== ==================== ==================== ===============
hkl       bissector          h, k, l            omega, chi, phi, tth omega, chi, phi, tth
hkl       constant_omega     h, k, l            omega, chi, phi, tth chi, phi, tth
hkl       constant_chi       h, k, l            omega, chi, phi, tth omega, phi, tth
hkl       constant_phi       h, k, l            omega, chi, phi, tth omega, chi, tth
hkl       double_diffraction h, k, l            omega, chi, phi, tth omega, chi, phi, tth h2, k2, l2
hkl       psi_constant       h, k, l            omega, chi, phi, tth omega, chi, phi, tth h2, k2, l2, psi
psi       psi                psi                omega, chi, phi, tth omega, chi, phi, tth h2, k2, l2
q         q                  q                  tth                  tth
incidence incidence          incidence, azimuth omega, chi, phi                           x, y, z
emergence emergence          emergence, azimuth omega, chi, phi, tth                      x, y, z
========= ================== ================== ==================== ==================== ===============

.. _geometries-hkl_soleil-e4cv:

solver='hkl_soleil', geometry='E4CV'
------------------------------------

.. index:: geometries; hkl_soleil; E4CV

========= ================== ================== ==================== ==================== ===============
engine    mode               pseudo(s)          real(s)              writable(s)          extra(s)
========= ================== ================== ==================== ==================== ===============
hkl       bissector          h, k, l            omega, chi, phi, tth omega, chi, phi, tth
hkl       constant_omega     h, k, l            omega, chi, phi, tth chi, phi, tth
hkl       constant_chi       h, k, l            omega, chi, phi, tth omega, phi, tth
hkl       constant_phi       h, k, l            omega, chi, phi, tth omega, chi, tth
hkl       double_diffraction h, k, l            omega, chi, phi, tth omega, chi, phi, tth h2, k2, l2
hkl       psi_constant       h, k, l            omega, chi, phi, tth omega, chi, phi, tth h2, k2, l2, psi
psi       psi                psi                omega, chi, phi, tth omega, chi, phi, tth h2, k2, l2
q         q                  q                  tth                  tth
incidence incidence          incidence, azimuth omega, chi, phi                           x, y, z
emergence emergence          emergence, azimuth omega, chi, phi, tth                      x, y, z
========= ================== ================== ==================== ==================== ===============

.. _geometries-hkl_soleil-e6c:

solver='hkl_soleil', geometry='E6C'
-----------------------------------

.. index:: geometries; hkl_soleil; E6C

========= ============================= ================== ================================= ========================== ===============
engine    mode                          pseudo(s)          real(s)                           writable(s)                extra(s)
========= ============================= ================== ================================= ========================== ===============
hkl       bissector_vertical            h, k, l            mu, omega, chi, phi, gamma, delta omega, chi, phi, delta
hkl       constant_omega_vertical       h, k, l            mu, omega, chi, phi, gamma, delta chi, phi, delta
hkl       constant_chi_vertical         h, k, l            mu, omega, chi, phi, gamma, delta omega, phi, delta
hkl       constant_phi_vertical         h, k, l            mu, omega, chi, phi, gamma, delta omega, chi, delta
hkl       lifting_detector_phi          h, k, l            mu, omega, chi, phi, gamma, delta phi, gamma, delta
hkl       lifting_detector_omega        h, k, l            mu, omega, chi, phi, gamma, delta omega, gamma, delta
hkl       lifting_detector_mu           h, k, l            mu, omega, chi, phi, gamma, delta mu, gamma, delta
hkl       double_diffraction_vertical   h, k, l            mu, omega, chi, phi, gamma, delta omega, chi, phi, delta     h2, k2, l2
hkl       bissector_horizontal          h, k, l            mu, omega, chi, phi, gamma, delta mu, omega, chi, phi, gamma
hkl       double_diffraction_horizontal h, k, l            mu, omega, chi, phi, gamma, delta mu, chi, phi, gamma        h2, k2, l2
hkl       psi_constant_vertical         h, k, l            mu, omega, chi, phi, gamma, delta omega, chi, phi, delta     h2, k2, l2, psi
hkl       psi_constant_horizontal       h, k, l            mu, omega, chi, phi, gamma, delta omega, chi, phi, gamma     h2, k2, l2, psi
hkl       constant_mu_horizontal        h, k, l            mu, omega, chi, phi, gamma, delta chi, phi, gamma
psi       psi_vertical                  psi                mu, omega, chi, phi, gamma, delta omega, chi, phi, delta     h2, k2, l2
q2        q2                            q, alpha           gamma, delta                      gamma, delta
qper_qpar qper_qpar                     qper, qpar         gamma, delta                      gamma, delta               x, y, z
tth2      tth2                          tth, alpha         gamma, delta                      gamma, delta
incidence incidence                     incidence, azimuth mu, omega, chi, phi                                          x, y, z
emergence emergence                     emergence, azimuth mu, omega, chi, phi, gamma, delta                            x, y, z
========= ============================= ================== ================================= ========================== ===============

.. _geometries-hkl_soleil-esrf-id01-psic:

solver='hkl_soleil', geometry='ESRF ID01 PSIC'
----------------------------------------------

.. index:: geometries; hkl_soleil; ESRF ID01 PSIC

====== ======================== ========= ======================= =============== ========
engine mode                     pseudo(s) real(s)                 writable(s)     extra(s)
====== ======================== ========= ======================= =============== ========
hkl    constant_nu_coplanar     h, k, l   mu, eta, phi, nu, delta eta, phi, delta
hkl    constant_delta_coplanar  h, k, l   mu, eta, phi, nu, delta eta, phi, nu
hkl    constant_eta_noncoplanar h, k, l   mu, eta, phi, nu, delta phi, nu, delta
====== ======================== ========= ======================= =============== ========

.. _geometries-hkl_soleil-k4cv:

solver='hkl_soleil', geometry='K4CV'
------------------------------------

.. index:: geometries; hkl_soleil; K4CV

========= ================== ================== ======================== ======================== ===============
engine    mode               pseudo(s)          real(s)                  writable(s)              extra(s)
========= ================== ================== ======================== ======================== ===============
hkl       bissector          h, k, l            komega, kappa, kphi, tth komega, kappa, kphi, tth
hkl       constant_omega     h, k, l            komega, kappa, kphi, tth komega, kappa, kphi, tth omega
hkl       constant_chi       h, k, l            komega, kappa, kphi, tth komega, kappa, kphi, tth chi
hkl       constant_phi       h, k, l            komega, kappa, kphi, tth komega, kappa, kphi, tth phi
hkl       double_diffraction h, k, l            komega, kappa, kphi, tth komega, kappa, kphi, tth h2, k2, l2
hkl       psi_constant       h, k, l            komega, kappa, kphi, tth komega, kappa, kphi, tth h2, k2, l2, psi
eulerians eulerians          omega, chi, phi    komega, kappa, kphi      komega, kappa, kphi      solutions
psi       psi                psi                komega, kappa, kphi, tth komega, kappa, kphi, tth h2, k2, l2
q         q                  q                  tth                      tth
incidence incidence          incidence, azimuth komega, kappa, kphi                               x, y, z
emergence emergence          emergence, azimuth komega, kappa, kphi, tth                          x, y, z
========= ================== ================== ======================== ======================== ===============

.. _geometries-hkl_soleil-k6c:

solver='hkl_soleil', geometry='K6C'
-----------------------------------

.. index:: geometries; hkl_soleil; K6C

========= ============================= ================== ===================================== ================================= ===========================
engine    mode                          pseudo(s)          real(s)                               writable(s)                       extra(s)
========= ============================= ================== ===================================== ================================= ===========================
hkl       bissector_vertical            h, k, l            mu, komega, kappa, kphi, gamma, delta komega, kappa, kphi, delta
hkl       constant_omega_vertical       h, k, l            mu, komega, kappa, kphi, gamma, delta komega, kappa, kphi, delta        omega
hkl       constant_chi_vertical         h, k, l            mu, komega, kappa, kphi, gamma, delta komega, kappa, kphi, delta        chi
hkl       constant_phi_vertical         h, k, l            mu, komega, kappa, kphi, gamma, delta komega, kappa, kphi, delta        phi
hkl       lifting_detector_kphi         h, k, l            mu, komega, kappa, kphi, gamma, delta kphi, gamma, delta
hkl       lifting_detector_komega       h, k, l            mu, komega, kappa, kphi, gamma, delta komega, gamma, delta
hkl       lifting_detector_mu           h, k, l            mu, komega, kappa, kphi, gamma, delta mu, gamma, delta
hkl       double_diffraction_vertical   h, k, l            mu, komega, kappa, kphi, gamma, delta komega, kappa, kphi, delta        h2, k2, l2
hkl       bissector_horizontal          h, k, l            mu, komega, kappa, kphi, gamma, delta mu, komega, kappa, kphi, gamma
hkl       constant_phi_horizontal       h, k, l            mu, komega, kappa, kphi, gamma, delta mu, komega, kappa, kphi, gamma    phi
hkl       constant_kphi_horizontal      h, k, l            mu, komega, kappa, kphi, gamma, delta mu, komega, kappa, gamma
hkl       double_diffraction_horizontal h, k, l            mu, komega, kappa, kphi, gamma, delta mu, komega, kappa, kphi, gamma    h2, k2, l2
hkl       psi_constant_vertical         h, k, l            mu, komega, kappa, kphi, gamma, delta komega, kappa, kphi, delta        h2, k2, l2, psi
hkl       constant_incidence            h, k, l            mu, komega, kappa, kphi, gamma, delta komega, kappa, kphi, gamma, delta x, y, z, incidence, azimuth
eulerians eulerians                     omega, chi, phi    komega, kappa, kphi                   komega, kappa, kphi               solutions
psi       psi_vertical                  psi                mu, komega, kappa, kphi, gamma, delta komega, kappa, kphi, delta        h2, k2, l2
q2        q2                            q, alpha           gamma, delta                          gamma, delta
qper_qpar qper_qpar                     qper, qpar         gamma, delta                          gamma, delta                      x, y, z
incidence incidence                     incidence, azimuth mu, komega, kappa, kphi                                                 x, y, z
tth2      tth2                          tth, alpha         gamma, delta                          gamma, delta
emergence emergence                     emergence, azimuth mu, komega, kappa, kphi, gamma, delta                                   x, y, z
========= ============================= ================== ===================================== ================================= ===========================

.. _geometries-hkl_soleil-petra3-p09-eh2:

solver='hkl_soleil', geometry='PETRA3 P09 EH2'
----------------------------------------------

.. index:: geometries; hkl_soleil; PETRA3 P09 EH2

====== =================================== ========= ================================= ======================= ========
engine mode                                pseudo(s) real(s)                           writable(s)             extra(s)
====== =================================== ========= ================================= ======================= ========
hkl    zaxis + alpha-fixed                 h, k, l   mu, omega, chi, phi, delta, gamma omega, delta, gamma
hkl    zaxis + beta-fixed                  h, k, l   mu, omega, chi, phi, delta, gamma mu, delta, gamma
hkl    zaxis + alpha=beta                  h, k, l   mu, omega, chi, phi, delta, gamma mu, omega, delta, gamma
hkl    4-circles bissecting horizontal     h, k, l   mu, omega, chi, phi, delta, gamma omega, chi, phi, delta
hkl    4-circles constant omega horizontal h, k, l   mu, omega, chi, phi, delta, gamma chi, phi, delta
hkl    4-circles constant chi horizontal   h, k, l   mu, omega, chi, phi, delta, gamma omega, phi, delta
hkl    4-circles constant phi horizontal   h, k, l   mu, omega, chi, phi, delta, gamma omega, chi, delta
hkl    lifting detector mu                 h, k, l   mu, omega, chi, phi, delta, gamma mu, delta, gamma
hkl    lifting detector omega              h, k, l   mu, omega, chi, phi, delta, gamma omega, delta, gamma
hkl    lifting detector chi                h, k, l   mu, omega, chi, phi, delta, gamma chi, delta, gamma
hkl    lifting detector phi                h, k, l   mu, omega, chi, phi, delta, gamma phi, delta, gamma
====== =================================== ========= ================================= ======================= ========

.. _geometries-hkl_soleil-petra3-p23-4c:

solver='hkl_soleil', geometry='PETRA3 P23 4C'
---------------------------------------------

.. index:: geometries; hkl_soleil; PETRA3 P23 4C

========= ======================== ================== ========================= ========================= ===============
engine    mode                     pseudo(s)          real(s)                   writable(s)               extra(s)
========= ======================== ================== ========================= ========================= ===============
hkl       bissector_vertical       h, k, l            omega_t, mu, gamma, delta omega_t, mu, delta
hkl       lifting_detector_omega_t h, k, l            omega_t, mu, gamma, delta omega_t, gamma, delta
hkl       lifting_detector_mu      h, k, l            omega_t, mu, gamma, delta mu, gamma, delta
hkl       bissector_horizontal     h, k, l            omega_t, mu, gamma, delta omega_t, mu, gamma
hkl       psi_constant             h, k, l            omega_t, mu, gamma, delta omega_t, mu, gamma, delta h2, k2, l2, psi
q2        q2                       q, alpha           gamma, delta              gamma, delta
qper_qpar qper_qpar                qper, qpar         gamma, delta              gamma, delta              x, y, z
tth2      tth2                     tth, alpha         gamma, delta              gamma, delta
incidence incidence                incidence, azimuth omega_t, mu                                         x, y, z
emergence emergence                emergence, azimuth omega_t, mu, gamma, delta                           x, y, z
========= ======================== ================== ========================= ========================= ===============

.. _geometries-hkl_soleil-petra3-p23-6c:

solver='hkl_soleil', geometry='PETRA3 P23 6C'
---------------------------------------------

.. index:: geometries; hkl_soleil; PETRA3 P23 6C

========= ============================= ================== ========================================== ========================== ===============
engine    mode                          pseudo(s)          real(s)                                    writable(s)                extra(s)
========= ============================= ================== ========================================== ========================== ===============
hkl       bissector_vertical            h, k, l            omega_t, mu, omega, chi, phi, gamma, delta omega, chi, phi, delta
hkl       constant_omega_vertical       h, k, l            omega_t, mu, omega, chi, phi, gamma, delta chi, phi, delta
hkl       constant_chi_vertical         h, k, l            omega_t, mu, omega, chi, phi, gamma, delta omega, phi, delta
hkl       constant_phi_vertical         h, k, l            omega_t, mu, omega, chi, phi, gamma, delta omega, chi, delta
hkl       lifting_detector_phi          h, k, l            omega_t, mu, omega, chi, phi, gamma, delta phi, gamma, delta
hkl       lifting_detector_omega        h, k, l            omega_t, mu, omega, chi, phi, gamma, delta omega, gamma, delta
hkl       lifting_detector_mu           h, k, l            omega_t, mu, omega, chi, phi, gamma, delta mu, gamma, delta
hkl       double_diffraction_vertical   h, k, l            omega_t, mu, omega, chi, phi, gamma, delta omega, chi, phi, delta     h2, k2, l2
hkl       bissector_horizontal          h, k, l            omega_t, mu, omega, chi, phi, gamma, delta mu, omega, chi, phi, gamma
hkl       double_diffraction_horizontal h, k, l            omega_t, mu, omega, chi, phi, gamma, delta mu, chi, phi, gamma        h2, k2, l2
hkl       psi_constant_vertical         h, k, l            omega_t, mu, omega, chi, phi, gamma, delta omega, chi, phi, delta     h2, k2, l2, psi
hkl       psi_constant_horizontal       h, k, l            omega_t, mu, omega, chi, phi, gamma, delta omega, chi, phi, gamma     h2, k2, l2, psi
hkl       constant_mu_horizontal        h, k, l            omega_t, mu, omega, chi, phi, gamma, delta chi, phi, gamma
psi       psi_vertical                  psi                omega_t, mu, omega, chi, phi, gamma, delta omega, chi, phi, delta     h2, k2, l2
q2        q2                            q, alpha           gamma, delta                               gamma, delta
qper_qpar qper_qpar                     qper, qpar         gamma, delta                               gamma, delta               x, y, z
tth2      tth2                          tth, alpha         gamma, delta                               gamma, delta
incidence incidence                     incidence, azimuth omega_t, mu, omega, chi, phi                                          x, y, z
emergence emergence                     emergence, azimuth omega_t, mu, omega, chi, phi, gamma, delta                            x, y, z
========= ============================= ================== ========================================== ========================== ===============

.. _geometries-hkl_soleil-soleil-mars:

solver='hkl_soleil', geometry='SOLEIL MARS'
-------------------------------------------

.. index:: geometries; hkl_soleil; SOLEIL MARS

========= ================== ================== ==================== ==================== ===============
engine    mode               pseudo(s)          real(s)              writable(s)          extra(s)
========= ================== ================== ==================== ==================== ===============
hkl       bissector          h, k, l            omega, chi, phi, tth omega, chi, phi, tth
hkl       constant_omega     h, k, l            omega, chi, phi, tth chi, phi, tth
hkl       constant_chi       h, k, l            omega, chi, phi, tth omega, phi, tth
hkl       constant_phi       h, k, l            omega, chi, phi, tth omega, chi, tth
hkl       double_diffraction h, k, l            omega, chi, phi, tth omega, chi, phi, tth h2, k2, l2
hkl       psi_constant       h, k, l            omega, chi, phi, tth omega, chi, phi, tth h2, k2, l2, psi
psi       psi                psi                omega, chi, phi, tth omega, chi, phi, tth h2, k2, l2
q         q                  q                  tth                  tth
incidence incidence          incidence, azimuth omega, chi, phi                           x, y, z
emergence emergence          emergence, azimuth omega, chi, phi, tth                      x, y, z
========= ================== ================== ==================== ==================== ===============

.. _geometries-hkl_soleil-soleil-nanoscopium-robot:

solver='hkl_soleil', geometry='SOLEIL NANOSCOPIUM ROBOT'
--------------------------------------------------------

.. index:: geometries; hkl_soleil; SOLEIL NANOSCOPIUM ROBOT

====== =================== ========= =========================== ================ ========
engine mode                pseudo(s) real(s)                     writable(s)      extra(s)
====== =================== ========= =========================== ================ ========
hkl    lifting detector rz h, k, l   rz, rs, rx, r, delta, gamma rz, delta, gamma
hkl    lifting detector rs h, k, l   rz, rs, rx, r, delta, gamma rs, delta, gamma
hkl    lifting detector rx h, k, l   rz, rs, rx, r, delta, gamma rx, delta, gamma
====== =================== ========= =========================== ================ ========

.. _geometries-hkl_soleil-soleil-sirius-kappa:

solver='hkl_soleil', geometry='SOLEIL SIRIUS KAPPA'
---------------------------------------------------

.. index:: geometries; hkl_soleil; SOLEIL SIRIUS KAPPA

========= ================================ ================== ===================================== ================================= ===========================
engine    mode                             pseudo(s)          real(s)                               writable(s)                       extra(s)
========= ================================ ================== ===================================== ================================= ===========================
hkl       bissector_vertical               h, k, l            mu, komega, kappa, kphi, delta, gamma komega, kappa, kphi, gamma
hkl       constant_omega_vertical          h, k, l            mu, komega, kappa, kphi, delta, gamma komega, kappa, kphi, gamma        omega
hkl       constant_chi_vertical            h, k, l            mu, komega, kappa, kphi, delta, gamma komega, kappa, kphi, gamma        chi
hkl       constant_phi_vertical            h, k, l            mu, komega, kappa, kphi, delta, gamma komega, kappa, kphi, gamma        phi
hkl       lifting_detector_kphi            h, k, l            mu, komega, kappa, kphi, delta, gamma kphi, delta, gamma
hkl       lifting_detector_komega          h, k, l            mu, komega, kappa, kphi, delta, gamma komega, delta, gamma
hkl       lifting_detector_mu              h, k, l            mu, komega, kappa, kphi, delta, gamma mu, delta, gamma
hkl       double_diffraction_vertical      h, k, l            mu, komega, kappa, kphi, delta, gamma komega, kappa, kphi, gamma        h2, k2, l2
hkl       bissector_horizontal             h, k, l            mu, komega, kappa, kphi, delta, gamma mu, komega, kappa, kphi, delta
hkl       constant_phi_horizontal          h, k, l            mu, komega, kappa, kphi, delta, gamma mu, komega, kappa, kphi, delta    phi
hkl       constant_kphi_horizontal         h, k, l            mu, komega, kappa, kphi, delta, gamma mu, komega, kappa, delta
hkl       double_diffraction_horizontal    h, k, l            mu, komega, kappa, kphi, delta, gamma mu, komega, kappa, kphi, delta    h2, k2, l2
hkl       psi_constant_vertical            h, k, l            mu, komega, kappa, kphi, delta, gamma komega, kappa, kphi, gamma        h2, k2, l2, psi
hkl       constant_incidence               h, k, l            mu, komega, kappa, kphi, delta, gamma komega, kappa, kphi, delta, gamma x, y, z, incidence, azimuth
eulerians eulerians                        omega, chi, phi    komega, kappa, kphi                   komega, kappa, kphi               solutions
psi       psi_vertical_soleil_sirius_kappa psi                mu, komega, kappa, kphi, delta, gamma komega, kappa, kphi, gamma        h2, k2, l2
q2        q2                               q, alpha           gamma, delta                          gamma, delta
qper_qpar qper_qpar                        qper, qpar         gamma, delta                          gamma, delta                      x, y, z
tth2      tth2                             tth, alpha         gamma, delta                          gamma, delta
incidence incidence                        incidence, azimuth mu, komega, kappa, kphi                                                 x, y, z
emergence emergence                        emergence, azimuth mu, komega, kappa, kphi, gamma, delta                                   x, y, z
========= ================================ ================== ===================================== ================================= ===========================

.. _geometries-hkl_soleil-soleil-sirius-turret:

solver='hkl_soleil', geometry='SOLEIL SIRIUS TURRET'
----------------------------------------------------

.. index:: geometries; hkl_soleil; SOLEIL SIRIUS TURRET

========= ======================= ================== =============================================== ==================== ========
engine    mode                    pseudo(s)          real(s)                                         writable(s)          extra(s)
========= ======================= ================== =============================================== ==================== ========
hkl       lifting_detector_thetah h, k, l            basepitch, thetah, alphay, alphax, delta, gamma thetah, delta, gamma
q2        q2                      q, alpha           gamma, delta                                    gamma, delta
qper_qpar qper_qpar               qper, qpar         gamma, delta                                    gamma, delta         x, y, z
tth2      tth2                    tth, alpha         gamma, delta                                    gamma, delta
incidence incidence               incidence, azimuth basepitch, thetah, alphay, alphax                                    x, y, z
emergence emergence               emergence, azimuth basepitch, thetah, alphay, alphax, delta, gamma                      x, y, z
========= ======================= ================== =============================================== ==================== ========

.. _geometries-hkl_soleil-soleil-sixs-med1+2:

solver='hkl_soleil', geometry='SOLEIL SIXS MED1+2'
--------------------------------------------------

.. index:: geometries; hkl_soleil; SOLEIL SIXS MED1+2

========= =========== ================== ======================= ================ ========
engine    mode        pseudo(s)          real(s)                 writable(s)      extra(s)
========= =========== ================== ======================= ================ ========
hkl       pitch_fixed h, k, l            pitch, mu, gamma, delta mu, gamma, delta
hkl       delta_fixed h, k, l            pitch, mu, gamma, delta pitch, mu, gamma
q2        q2          q, alpha           gamma, delta            gamma, delta
qper_qpar qper_qpar   qper, qpar         gamma, delta            gamma, delta     x, y, z
tth2      tth2        tth, alpha         gamma, delta            gamma, delta
incidence incidence   incidence, azimuth pitch, mu                                x, y, z
emergence emergence   emergence, azimuth pitch, mu, gamma, delta                  x, y, z
========= =========== ================== ======================= ================ ========

.. _geometries-hkl_soleil-soleil-sixs-med2+2:

solver='hkl_soleil', geometry='SOLEIL SIXS MED2+2'
--------------------------------------------------

.. index:: geometries; hkl_soleil; SOLEIL SIXS MED2+2

========= =============== ================== ============================= ======================= ==================
engine    mode            pseudo(s)          real(s)                       writable(s)             extra(s)
========= =============== ================== ============================= ======================= ==================
hkl       mu_fixed        h, k, l            beta, mu, omega, gamma, delta omega, gamma, delta
hkl       reflectivity    h, k, l            beta, mu, omega, gamma, delta mu, omega, gamma, delta
hkl       emergence_fixed h, k, l            beta, mu, omega, gamma, delta mu, omega, gamma, delta x, y, z, emergence
q2        q2              q, alpha           gamma, delta                  gamma, delta
qper_qpar qper_qpar       qper, qpar         gamma, delta                  gamma, delta            x, y, z
tth2      tth2            tth, alpha         gamma, delta                  gamma, delta
incidence incidence       incidence, azimuth beta, mu, omega                                       x, y, z
emergence emergence       emergence, azimuth beta, mu, omega, gamma, delta                         x, y, z
========= =============== ================== ============================= ======================= ==================

.. _geometries-hkl_soleil-soleil-sixs-med2+3:

solver='hkl_soleil', geometry='SOLEIL SIXS MED2+3'
--------------------------------------------------

.. index:: geometries; hkl_soleil; SOLEIL SIXS MED2+3

========= =============== ================== ==================================== ======================= ==================
engine    mode            pseudo(s)          real(s)                              writable(s)             extra(s)
========= =============== ================== ==================================== ======================= ==================
hkl       mu_fixed        h, k, l            beta, mu, omega, gamma, delta, eta_a omega, gamma, delta
hkl       gamma_fixed     h, k, l            beta, mu, omega, gamma, delta, eta_a mu, omega, delta
hkl       emergence_fixed h, k, l            beta, mu, omega, gamma, delta, eta_a mu, omega, gamma, delta x, y, z, emergence
q2        q2              q, alpha           gamma, delta                         gamma, delta
qper_qpar qper_qpar       qper, qpar         gamma, delta                         gamma, delta            x, y, z
tth2      tth2            tth, alpha         gamma, delta                         gamma, delta
incidence incidence       incidence, azimuth beta, mu, omega                                              x, y, z
emergence emergence       emergence, azimuth beta, mu, omega, gamma, delta                                x, y, z
========= =============== ================== ==================================== ======================= ==================

.. _geometries-hkl_soleil-soleil-sixs-med2+3-v2:

solver='hkl_soleil', geometry='SOLEIL SIXS MED2+3 v2'
-----------------------------------------------------

.. index:: geometries; hkl_soleil; SOLEIL SIXS MED2+3 v2

========= =============== ================== ============================== ======================= ==================
engine    mode            pseudo(s)          real(s)                        writable(s)             extra(s)
========= =============== ================== ============================== ======================= ==================
hkl       mu_fixed        h, k, l            mu, omega, gamma, delta, eta_a omega, gamma, delta
hkl       gamma_fixed     h, k, l            mu, omega, gamma, delta, eta_a mu, omega, delta
hkl       emergence_fixed h, k, l            mu, omega, gamma, delta, eta_a mu, omega, gamma, delta x, y, z, emergence
q2        q2              q, alpha           gamma, delta                   gamma, delta
qper_qpar qper_qpar       qper, qpar         gamma, delta                   gamma, delta            x, y, z
tth2      tth2            tth, alpha         gamma, delta                   gamma, delta
incidence incidence       incidence, azimuth beta, mu, omega                                        x, y, z
emergence emergence       emergence, azimuth beta, mu, omega, gamma, delta                          x, y, z
========= =============== ================== ============================== ======================= ==================

.. _geometries-hkl_soleil-zaxis:

solver='hkl_soleil', geometry='ZAXIS'
-------------------------------------

.. index:: geometries; hkl_soleil; ZAXIS

========= ============ ================== ======================= ======================= ========
engine    mode         pseudo(s)          real(s)                 writable(s)             extra(s)
========= ============ ================== ======================= ======================= ========
hkl       zaxis        h, k, l            mu, omega, delta, gamma omega, delta, gamma
hkl       reflectivity h, k, l            mu, omega, delta, gamma mu, omega, delta, gamma
q2        q2           q, alpha           gamma, delta            gamma, delta
qper_qpar qper_qpar    qper, qpar         gamma, delta            gamma, delta            x, y, z
tth2      tth2         tth, alpha         gamma, delta            gamma, delta
incidence incidence    incidence, azimuth mu, omega                                       x, y, z
emergence emergence    emergence, azimuth mu, omega, delta, gamma                         x, y, z
========= ============ ================== ======================= ======================= ========

.. _geometries-th_tth-th-tth-q:

solver='th_tth', geometry='TH TTH Q'
------------------------------------

.. index:: geometries; th_tth; TH TTH Q

========= ========= ======= =========== ========
mode      pseudo(s) real(s) writable(s) extra(s)
========= ========= ======= =========== ========
bissector q         th, tth th, tth
========= ========= ======= =========== ========
