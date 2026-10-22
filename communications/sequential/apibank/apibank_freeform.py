"""Communications pair: sequential API-Bank with freeform inter-agent reports."""

from communications import communication_formats

communication_formats.install(globals(), topology="sequential", dataset="apibank", fmt="freeform")
