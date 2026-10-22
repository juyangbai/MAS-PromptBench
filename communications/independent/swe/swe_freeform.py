"""Communications pair: independent SWE-bench Verified with freeform inter-agent reports."""

from communications import communication_formats

communication_formats.install(globals(), topology="independent", dataset="swe", fmt="freeform")
