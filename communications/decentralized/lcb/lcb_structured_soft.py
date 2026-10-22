"""Communications pair: decentralized LCB with structured_soft inter-agent reports."""

from communications import communication_formats

communication_formats.install(globals(), topology="decentralized", dataset="lcb", fmt="structured_soft")
