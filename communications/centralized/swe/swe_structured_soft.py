"""Communications pair: centralized SWE-bench Verified with structured_soft inter-agent reports."""

from communications import communication_formats

communication_formats.install(globals(), topology="centralized", dataset="swe", fmt="structured_soft")
