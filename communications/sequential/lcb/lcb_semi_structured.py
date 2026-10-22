"""Communications pair: sequential LCB with semi_structured inter-agent reports."""

from communications import communication_formats

communication_formats.install(globals(), topology="sequential", dataset="lcb", fmt="semi_structured")
