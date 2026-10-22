"""Communications pair: independent SWE-bench Verified with structured_soft inter-agent reports."""

from communications import communication_formats

communication_formats.install(globals(), topology="independent", dataset="swe", fmt="structured_soft")
