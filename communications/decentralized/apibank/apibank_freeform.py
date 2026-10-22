"""Communications pair: decentralized API-Bank with freeform inter-agent reports."""

from communications import communication_formats

communication_formats.install(globals(), topology="decentralized", dataset="apibank", fmt="freeform")
