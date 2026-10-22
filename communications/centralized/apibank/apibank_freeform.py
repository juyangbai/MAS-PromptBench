"""Communications pair: centralized API-Bank with freeform inter-agent reports."""

from communications import communication_formats

communication_formats.install(globals(), topology="centralized", dataset="apibank", fmt="freeform")
