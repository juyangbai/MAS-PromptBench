"""Communications pair: decentralized ToolHop with semi_structured inter-agent reports."""

from communications import communication_formats

communication_formats.install(globals(), topology="decentralized", dataset="toolhop", fmt="semi_structured")
