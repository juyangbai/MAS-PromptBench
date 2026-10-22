"""Communications pair: sequential HotpotQA with structured_soft inter-agent reports."""

from communications import communication_formats

communication_formats.install(globals(), topology="sequential", dataset="hotpotqa", fmt="structured_soft")
