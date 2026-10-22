"""Communications pair: centralized ToolHop with structured_soft inter-agent reports."""

from communications import communication_formats

communication_formats.install(globals(), topology="centralized", dataset="toolhop", fmt="structured_soft")
