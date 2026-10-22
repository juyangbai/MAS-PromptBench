"""Communications pair: centralized ToolHop with semi_structured inter-agent reports."""

from communications import communication_formats

communication_formats.install(globals(), topology="centralized", dataset="toolhop", fmt="semi_structured")
