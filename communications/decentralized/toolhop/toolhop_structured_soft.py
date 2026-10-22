"""Communications pair: decentralized ToolHop with structured_soft inter-agent reports."""

from communications import communication_formats

communication_formats.install(globals(), topology="decentralized", dataset="toolhop", fmt="structured_soft")
