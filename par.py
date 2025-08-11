import numpy as np
from regularizer import Regularizer
import matplotlib.pyplot as plt


class PAR(Regularizer):
    def __init__(self, slopes, qs):
        """
        Initialize the piecewise affine regularizer
        """
        self.slopes = slopes  # slopes of the pieces

        if qs is None:
            self.qs = np.arange(len(slopes))
        else:
            if len(qs) != len(slopes):
                raise ValueError("Length of qs must be the length of slopes")
            if not all(q1 < q2 for q1, q2 in zip(qs[:-1], qs[1:])):
                raise ValueError("Knots must be strictly increasing")
            self.qs = qs  # quantization points

        # precompute cumulative sums for calculating the function value
        self.compute_cumulative_sums()

    def is_convex(self):
        """
        Check if the function is convex
        """
        return (self.slopes[0] >= 0) and all(
            slope1 <= slope2
            for slope1, slope2 in zip(self.slopes[:-1], self.slopes[1:])
        )

    def func(self, x):
        """
        Compute the value of the regularizer at x
        """
        x = np.asarray(x)
        is_scalar = x.ndim == 0
        if is_scalar:
            x = x[None]
        abs_x = np.abs(x)
        sum = 0.0

        for i, val in enumerate(abs_x):
            value = 0.0
            period = np.searchsorted(self.qs, val, side="right") - 1

            if period > 0:
                value = self.cumsum_cache[period - 1].item()
            value += self.slopes[period].item() * \
                (abs_x[i] - self.qs[period]).item()
            sum += value

        return sum

    def compute_cumulative_sums(self):
        """
        Precompute cumulative sums for each period
        """
        period_contributions = self.slopes[:-1] * (self.qs[1:] - self.qs[:-1])
        self.cumsum_cache = np.cumsum(period_contributions)

    def _prox_mapping_1d(self, z, lambda_):
        """
        Compute the proximal mapping for 1d input
        """
        slopes = self.slopes
        qs = self.qs
        m = len(slopes)  # Number of pieces in the p.w. function
        abs_z = np.abs(z)
        if lambda_ < 0:
            raise ValueError("lambda_ must be nonnegative")

        # convex case
        if self.is_convex():
            # Handle values below the first breakpoint
            if abs_z < qs[0]:
                return 0.0

            # Check if z corresponds to any breakpoint
            for k in range(m):
                if k < m - 1:  # Not the last breakpoint
                    # Check if z is in the flat region at breakpoint q_k
                    if abs_z <= lambda_ * slopes[k] + qs[k] and (
                        k == 0 or abs_z >= lambda_ * slopes[k - 1] + qs[k]
                    ):
                        return np.sign(z) * qs[k]

                    # Check if z is in the linear region between breakpoints q_k and q_{k+1}
                    if (
                        abs_z >= lambda_ * slopes[k] + qs[k]
                        and abs_z <= lambda_ * slopes[k] + qs[k + 1]
                    ):
                        return z - np.sign(z) * lambda_ * slopes[k]
                else:  # Last breakpoint
                    # Check if z is in the flat region at the last breakpoint
                    if abs_z <= lambda_ * slopes[k] + qs[k]:
                        return np.sign(z) * qs[k]

                    # z is beyond the last breakpoint
                    return z - np.sign(z) * lambda_ * slopes[k]

            # This point should not be reached for a properly constructed PAR with
            # correct breakpoints, but we include it as a fallback
            return z - np.sign(z) * lambda_ * slopes[-1]

        # special nonconvex case where the slopes are [1, -1, 1, -1, ...]
        elif np.allclose(slopes[::2], 1.0) and np.allclose(
                slopes[1::2], -1.0):
            abs_z = np.abs(z)
            z_floor = np.floor(abs_z)
            z_sign = np.sign(z)
            return (
                z_sign * max(z_floor, abs_z - lambda_)
                if abs_z - z_floor < 0.5
                else z_sign * min(z_floor + 1, abs_z + lambda_)
            )

        # general nonconvex case
        else:
            abs_z = np.abs(z)
            candidates = qs.tolist()
            for k in range(m - 1):
                z_opt = abs_z - lambda_ * slopes[k]
                if qs[k] <= z_opt and z_opt <= qs[k + 1]:
                    candidates.append(z_opt)
            z_opt = abs_z - lambda_ * slopes[-1]
            if z_opt >= qs[-1]:
                candidates.append(z_opt)

            abs_z_prox = candidates[0]
            func_opt = lambda_ * \
                self.func(abs_z_prox) + 0.5 * (abs_z_prox - abs_z) ** 2
            for candidate in candidates:
                if (
                    lambda_ * self.func(candidate) + 0.5 *
                    (candidate - abs_z) ** 2
                    < func_opt
                ):
                    abs_z_prox = candidate
                    func_opt = (
                        lambda_ * self.func(candidate) +
                        0.5 * (candidate - abs_z) ** 2
                    )
            return np.sign(z) * abs_z_prox

    def prox_mapping(self, z, lambda_):
        """
        Compute the proximal mapping for the input z
        """
        return np.array(
            [
                self._prox_mapping_1d(z=z[i, 0], lambda_=lambda_)
                for i in range(z.shape[0])
            ]
        ).reshape(-1, 1)

    def scaled_prox_mapping(self, b, z, g, lambda_):
        """
        Compute the proximal mapping for the input z with scaling, i.e., return argmin_x {sum_i [(1/2)bi*(xi-zi)^2 + gi*xi] + lambda*R(x)}
        """
        result = np.zeros_like(z)
        for i in range(len(z)):
            result[i] = self._prox_mapping_1d(
                z=z[i] - g[i] / b[i], lambda_=lambda_ / b[i]
            )
        return result

    def calculate_quantization_rate(self, x, rtol=0, atol=1e-3):
        x = np.abs(x).flatten()
        return np.mean(
            np.any(
                np.isclose(x[:, None], self.qs[None, :], rtol=rtol, atol=atol), axis=1
            )
        )

    def get_quantized_solution(self, x):
        x_flatten = np.asarray(x).flatten()  # Ensure 1D
        sign = np.sign(x_flatten)
        abs_x = np.abs(x_flatten)

        # Compute distance matrix between abs_x and qs
        dists = np.abs(abs_x[:, None] - self.qs[None, :])

        # Find index of closest qs for each x
        closest_indices = np.argmin(dists, axis=1)
        closest_qs = self.qs[closest_indices]

        # Restore original sign
        quantized = sign * closest_qs
        return quantized.reshape(x.shape if x.ndim == 1 else (-1, 1))