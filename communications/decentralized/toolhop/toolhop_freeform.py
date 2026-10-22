"""Communications pair: decentralized ToolHop with freeform inter-agent reports."""

from communications import communication_formats

communication_formats.install(globals(), topology="decentralized", dataset="toolhop", fmt="freeform")
