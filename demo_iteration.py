#!/usr/bin/env python3
"""确定性离线演示：完整后验协方差的相关 GP 函数抽样。"""

import math
import time

from gprbf import Hyperparameters, predict, sample_posterior


def main():
    start = time.perf_counter()
    X = [(-1.0,), (0.0,), (1.0,)]
    y = [-0.4, 0.8, -0.2]
    hp = Hyperparameters(alpha=1.2, lengthscale=0.8, noise=0.0)
    queries = [(0.5,), (0.5,), (0.0,)]
    normal = [(1.5, 0.0, 0.0), (-0.7, 1.2, 0.0)]

    print("=" * 66)
    print("joint posterior sampling demo (Python standard library)")
    print("=" * 66)
    print(f"training={X}, observations={y}, noise={hp.noise:g}")
    print(f"queries={queries}")
    print()

    baseline = predict(
        X, y, queries, Hyperparameters(hp.alpha, hp.lengthscale, 1e-8)
    )
    assert baseline.status == "ok"
    print("[baseline predict interface] mean and latent variance")
    for index, point in enumerate(queries):
        print(
            f"  x={point[0]: .1f}: mean={baseline.mean[index]: .9f}, "
            f"variance={baseline.latent_variance[index]: .3e}"
        )
    print()

    joint = sample_posterior(X, y, queries, hp, normal)
    assert joint.status == "ok", joint.status
    print("[joint samples] one normal matrix row produces one correlated function")
    print(f"  covariance rank={joint.rank}, residual={joint.covariance_residual:.3e}")
    print(f"  corrected eigenvalues={joint.corrected_eigenvalues}")
    for index, sample in enumerate(joint.samples):
        print(f"  sample {index}: {tuple(round(v, 9) for v in sample)}")
    print("  duplicate query values are identical; training point remains exact.")
    print()

    fixed = sample_posterior(
        X, y, queries, hp, [(0.8,)], fixed_values={0: 0.25, 1: 0.25}
    )
    assert fixed.status == "ok", fixed.status
    print("[fixed observation] conditioning on the two identical coordinates")
    print(f"  sample={tuple(round(v, 9) for v in fixed.samples[0])}")
    assert fixed.samples[0][0] == 0.25 and fixed.samples[0][1] == 0.25
    print()

    incompatible = sample_posterior(
        X, y, queries, hp, [(0.8,)], fixed_values={0: 0.25, 1: 0.26}
    )
    print("[zero-variance rejection] contradictory duplicate fixed values")
    print(f"  status={incompatible.status!r}; samples={incompatible.samples}")
    assert incompatible.status == "incompatible_observation"
    print()

    empty = sample_posterior(X, y, [], hp, [(), ()])
    assert empty.status == "ok" and empty.samples == ((), ())
    print(f"[empty query] status={empty.status!r}, samples={empty.samples}")
    elapsed = time.perf_counter() - start
    print(f"\ncompleted in {elapsed:.3f} seconds (offline, deterministic inputs).")
    if not math.isfinite(elapsed) or elapsed > 90.0:
        raise AssertionError("demo exceeded 90 seconds")


if __name__ == "__main__":
    main()
