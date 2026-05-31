"""
Calculate the atomic form factor (fqdam) for a low-Z
(i.e. with electrons to the third orbital) atom
using a Slater-rule effective charge model.

This implementation supports electronic configurations up to the n=3 shell
(1s, 2s/2p, 3s/3p), which is sufficient for biologically relevant elements
(C, N, O, S). The form factor is evaluated in momentum space and can be used
directly in coherent scattering or nanocrystal diffraction simulations.

The model follows the standard Slater shielding rules:

    - slater2()  → effective nuclear charge for 2s/2p electrons
    - slater3()  → effective nuclear charge for 3s/3p electrons

Given the electron counts (p1s, p2, p3) and the true nuclear charge Z,
the function constructs the effective Slater exponents and evaluates the
analytic hydrogenic form-factor expressions for each shell. The total
form factor is the weighted sum:

        f(q) = p1s * f1(q) + p2 * f2(q) + p3 * f3(q)



"""


def slater2(p1s, p2):
    """_summary_

    Args:
        p1s (int): number of electrons in p1s orbital
        p2 (int): number of electrons in p2s orbital

    Returns:
        _float_: Shielding constant calculated
    """
    # first shell
    n1 = p1s
    n2 = p2
    # second shell
    sum2 = 0.35 * (n2 - 1) + 0.85 * n1
    s = sum2
    return s


def slater3(p1s, p2, p3):
    """_summary_

    Args:
        p1s (int): number of electrons in p1s orbital
        p2 (int): number of electrons in p2s orbital
        p3 (int): number of electrons in p3 orbital

    Returns:
        _float_: Shielding constant
    """

    n1 = p1s
    n2 = p2
    n3 = p3

    s = 0.35 * (n3 - 1) + 0.85 * (n1 + n2)

    return s


def fq_dam(q, p1s, p2, p3, znuc):
    """
    Vectorised calculation of form factor incorporating electronic damage
    q      : numpy array (same shape as radial_q)
    p1s    : 1s electron count
    p2     : 2s/2p electron count
    p3     : 3s/3p electron count (0 if not used)
    znuc   : nuclear charge (atomic number)
    """

    # q^2 - Square of the magnitude
    Q2 = q * q

    # Slater shielding for 1s
    S = 0.3 if p1s == 2 else 0.0

    # 1s shell
    zeta1 = znuc - S
    zeta2 = 4.0 * zeta1 * zeta1
    f1 = (zeta2 / (zeta2 + Q2)) ** 2

    # 2s / 2p shielding
    S2 = slater2(p1s, p2)

    zeta3 = (znuc - S2) / 2.0
    zeta4 = 4.0 * zeta3 * zeta3

    rnum = (zeta4**3) * (zeta4 - Q2)
    rden = (zeta4 + Q2) ** 4
    f2 = rnum / rden

    # 3s / 3p contribution
    if p3 != 0:
        S3 = slater3(p1s, p2, p3)

        zeta5 = (znuc - S3) / 3.0
        zeta6 = ((2 * zeta5) ** 8) / 3.0
        zeta7 = 48 * zeta5**4 - 40 * zeta5**2 * Q2 + Q2**4
        zeta8 = (4 * zeta5**2 + Q2**2) ** 6

        f3 = zeta6 * zeta7 / zeta8
    else:
        f3 = 0.0

    # Total form factor
    fq = p1s * f1 + p2 * f2 + p3 * f3

    return fq
