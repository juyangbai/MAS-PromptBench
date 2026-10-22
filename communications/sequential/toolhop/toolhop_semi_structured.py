"""Communications pair: sequential ToolHop with semi_structured inter-agent reports."""

from communications import communication_formats

communication_formats.install(globals(), topology="sequential", dataset="toolhop", fmt="semi_structured")
