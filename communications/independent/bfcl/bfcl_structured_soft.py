"""Communications pair: independent BFCL with structured_soft inter-agent reports."""

from communications import communication_formats

communication_formats.install(globals(), topology="independent", dataset="bfcl", fmt="structured_soft")
