"""Communications pair: sequential LCB with freeform inter-agent reports."""

from communications import communication_formats

communication_formats.install(globals(), topology="sequential", dataset="lcb", fmt="freeform")
