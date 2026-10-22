"""Communications pair: decentralized HotpotQA with freeform inter-agent reports."""

from communications import communication_formats

communication_formats.install(globals(), topology="decentralized", dataset="hotpotqa", fmt="freeform")
