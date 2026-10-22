"""Communications pair: centralized BFCL with structured_soft inter-agent reports."""

from communications import communication_formats

communication_formats.install(globals(), topology="centralized", dataset="bfcl", fmt="structured_soft")
