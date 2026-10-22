"""Communications pair: decentralized BFCL with semi_structured inter-agent reports."""

from communications import communication_formats

communication_formats.install(globals(), topology="decentralized", dataset="bfcl", fmt="semi_structured")
