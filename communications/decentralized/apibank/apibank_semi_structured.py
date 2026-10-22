"""Communications pair: decentralized API-Bank with semi_structured inter-agent reports."""

from communications import communication_formats

communication_formats.install(globals(), topology="decentralized", dataset="apibank", fmt="semi_structured")
