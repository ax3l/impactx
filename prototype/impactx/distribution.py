"""Pure-Python port of ImpactX beam distributions.

Currently provides the Waterbag distribution used by the FODO example, a
vectorized (numpy) reimplementation of
src/particles/distribution/Waterbag.H.
"""

import numpy as np


class Waterbag:
    """A 6D Waterbag distribution (uniformly filled phase-space ball).

    Parameters mirror the ImpactX C++ constructor. ``lambda*`` are the RMS-like
    intercepts, ``mu*`` the position-momentum correlations, ``mean*`` centroid
    offsets, ``disp*`` dispersion.
    """

    def __init__(
        self,
        lambdaX,
        lambdaY,
        lambdaT,
        lambdaPx,
        lambdaPy,
        lambdaPt,
        muxpx=0.0,
        muypy=0.0,
        mutpt=0.0,
        meanX=0.0,
        meanY=0.0,
        meanT=0.0,
        meanPx=0.0,
        meanPy=0.0,
        meanPt=0.0,
        dispX=0.0,
        dispPx=0.0,
        dispY=0.0,
        dispPy=0.0,
    ):
        self.lambdaX = lambdaX
        self.lambdaY = lambdaY
        self.lambdaT = lambdaT
        self.lambdaPx = lambdaPx
        self.lambdaPy = lambdaPy
        self.lambdaPt = lambdaPt
        self.muxpx = muxpx
        self.muypy = muypy
        self.mutpt = mutpt
        self.meanX = meanX
        self.meanY = meanY
        self.meanT = meanT
        self.meanPx = meanPx
        self.meanPy = meanPy
        self.meanPt = meanPt
        self.dispX = dispX
        self.dispPx = dispPx
        self.dispY = dispY
        self.dispPy = dispPy

    def sample(self, npart, rng=None):
        """Return six numpy arrays (x, y, t, px, py, pt) of length ``npart``.

        Reproduces the algorithm in Waterbag.H: draw 6 standard normals per
        particle, project onto the unit 6-sphere, scale by ``sqrt(8)*u^(1/6)``
        to fill the ball with unit variance, then apply correlations,
        dispersion and centroid offsets.
        """
        if rng is None:
            rng = np.random.default_rng()

        g = rng.standard_normal((6, npart))
        norm = np.sqrt(np.sum(g * g, axis=0))
        g = g / norm  # uniform on the unit 6-sphere

        u1 = rng.random(npart)
        scale = np.sqrt(8.0) * u1 ** (1.0 / 6.0)  # uniform in the ball

        x, y, t, px, py, pt = (g[i] * scale for i in range(6))

        # correlations
        root = np.sqrt(1.0 - self.muxpx**2)
        x, px = self.lambdaX * x / root, self.lambdaPx * (-self.muxpx * x / root + px)
        root = np.sqrt(1.0 - self.muypy**2)
        y, py = self.lambdaY * y / root, self.lambdaPy * (-self.muypy * y / root + py)
        root = np.sqrt(1.0 - self.mutpt**2)
        t, pt = self.lambdaT * t / root, self.lambdaPt * (-self.mutpt * t / root + pt)

        # dispersion
        x = x - self.dispX * pt
        px = px - self.dispPx * pt
        y = y - self.dispY * pt
        py = py - self.dispPy * pt

        # centroid offsets
        x = x + self.meanX
        px = px + self.meanPx
        y = y + self.meanY
        py = py + self.meanPy
        t = t + self.meanT
        pt = pt + self.meanPt

        return x, y, t, px, py, pt
