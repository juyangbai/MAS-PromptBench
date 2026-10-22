"""Communications pair: independent API-Bank with freeform inter-agent reports."""

from communications import communication_formats

communication_formats.install(globals(), topology="independent", dataset="apibank", fmt="freeform")
