"""Communications pair: centralized LCB with structured_soft inter-agent reports."""

from communications import communication_formats

communication_formats.install(globals(), topology="centralized", dataset="lcb", fmt="structured_soft")
