import numpy as np


def slater2(p1s, p2):
    # first shell
    n1tot = p1s
    n2tot = p2
    # second shell
    sum2 = 0.35 * (n2tot - 1) + 0.85 * n1tot
    s = sum2
    return s


def slater3(p1s, p2, p3):

    n1tot = p1s
    n2tot = p2
    n3tot = p3
    # third shell

    s = 0.35 * (n3tot - 1) + 0.85 * (n1tot + n2tot)

    return s


def fqdam(q, p1s, p2, p3, ZNUC):
    """
    Vectorised calculation of form factor incorporating electronic damage
    q      : numpy array (same shape as radial_q)
    p1s    : 1s electron count
    p2     : 2s/2p electron count
    p3     : 3s/3p electron count (0 if not used)
    ZNUC   : nuclear charge (atomic number)
    """

    # q^2
    Q2 = q * q

    # Slater shielding for 1s
    S = 0.3 if p1s == 2 else 0.0

    # 1s shell
    ZETA1 = ZNUC - S
    ZETA2 = 4.0 * ZETA1 * ZETA1
    F1 = (ZETA2 / (ZETA2 + Q2)) ** 2

    # 2s / 2p shielding
    S2 = slater2(p1s, p2)

    ZETA3 = (ZNUC - S2) / 2.0
    ZETA4 = 4.0 * ZETA3 * ZETA3

    RNUM = (ZETA4**3) * (ZETA4 - Q2)
    RDEN = (ZETA4 + Q2) ** 4
    F2 = RNUM / RDEN

    # 3s / 3p contribution
    if p3 != 0:
        S3 = slater3(p1s, p2, p3)

        ZETA5 = (ZNUC - S3) / 3.0
        ZETA6 = ((2 * ZETA5) ** 8) / 3.0
        ZETA7 = 48 * ZETA5**4 - 40 * ZETA5**2 * Q2 + Q2**4
        ZETA8 = (4 * ZETA5**2 + Q2**2) ** 6

        F3 = ZETA6 * ZETA7 / ZETA8
    else:
        F3 = 0.0

    # Total form factor
    fq = p1s * F1 + p2 * F2 + p3 * F3

    return fq
