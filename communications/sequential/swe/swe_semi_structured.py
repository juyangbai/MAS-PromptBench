"""Communications pair: sequential SWE-bench Verified with semi_structured inter-agent reports."""

from communications import communication_formats

communication_formats.install(globals(), topology="sequential", dataset="swe", fmt="semi_structured")
