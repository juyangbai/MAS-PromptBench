"""Communications pair: independent ToolHop with freeform inter-agent reports."""

from communications import communication_formats

communication_formats.install(globals(), topology="independent", dataset="toolhop", fmt="freeform")
