"""Report generators — each module owns one PDF and has a ``main()`` CLI entry.

* :mod:`~hm_rat_analysis.reports.session_summary` — cross-session, per-animal
  neural summary over a tree of session NWBs (``hm-session-summary``).
* :mod:`~hm_rat_analysis.reports.trial_report` — per-session behavioural report
  from one session folder's tracker logs (``hm-trial-report``).
"""
