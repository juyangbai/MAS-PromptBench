"""Communications pair: independent ToolHop with structured_soft inter-agent reports."""

from communications import communication_formats

communication_formats.install(globals(), topology="independent", dataset="toolhop", fmt="structured_soft")
