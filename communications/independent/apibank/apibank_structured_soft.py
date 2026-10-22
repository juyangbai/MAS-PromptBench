"""Communications pair: independent API-Bank with structured_soft inter-agent reports."""

from communications import communication_formats

communication_formats.install(globals(), topology="independent", dataset="apibank", fmt="structured_soft")
