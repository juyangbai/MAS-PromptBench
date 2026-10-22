"""Communications pair: decentralized HotpotQA with semi_structured inter-agent reports."""

from communications import communication_formats

communication_formats.install(globals(), topology="decentralized", dataset="hotpotqa", fmt="semi_structured")
