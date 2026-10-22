"""Communications pair: decentralized BFCL with structured_soft inter-agent reports."""

from communications import communication_formats

communication_formats.install(globals(), topology="decentralized", dataset="bfcl", fmt="structured_soft")
