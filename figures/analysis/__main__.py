"""Draw the analysis figures, one file per plot, into ``data.OUT_DIR``.

    python -m analysis                      # every figure
    python -m analysis detour_by_session    # just this one
    python -m analysis --rebuild            # re-walk the sessions first
    HM_FIG_DIR=~/Desktop/figs python -m analysis    # somewhere else
"""

import argparse
import sys

from analysis import data, plots


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("names", nargs="*", default=None,
                    help=f"figures to draw (default: all). Known: "
                         f"{', '.join(sorted(plots.REGISTRY))}")
    ap.add_argument("--rebuild", action="store_true",
                    help="re-walk the session folders instead of using the CSV cache")
    ap.add_argument("--root", default=None, help="session root (default $HM_TASK_ROOT)")
    a = ap.parse_args(argv)

    names = a.names or sorted(plots.REGISTRY)
    unknown = [n for n in names if n not in plots.REGISTRY]
    if unknown:
        ap.error(f"unknown figure(s): {', '.join(unknown)}")

    print(f"sessions: {a.root or data.ROOT}")
    print(f"figures:  {data.OUT_DIR}")
    trials = data.trials(root=a.root, rebuild=a.rebuild)
    print(f"trials:   {len(trials)} rows")
    for n in names:
        plots.REGISTRY[n](trials)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
