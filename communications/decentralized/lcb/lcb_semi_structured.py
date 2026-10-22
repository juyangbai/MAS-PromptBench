"""Communications pair: decentralized LCB with semi_structured inter-agent reports."""

from communications import communication_formats

communication_formats.install(globals(), topology="decentralized", dataset="lcb", fmt="semi_structured")
