"""Communications pair: decentralized HotpotQA with structured_soft inter-agent reports."""

from communications import communication_formats

communication_formats.install(globals(), topology="decentralized", dataset="hotpotqa", fmt="structured_soft")
