"""Communications pair: sequential BFCL with structured_soft inter-agent reports."""

from communications import communication_formats

communication_formats.install(globals(), topology="sequential", dataset="bfcl", fmt="structured_soft")
