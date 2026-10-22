"""Communications pair: decentralized LCB with freeform inter-agent reports."""

from communications import communication_formats

communication_formats.install(globals(), topology="decentralized", dataset="lcb", fmt="freeform")
