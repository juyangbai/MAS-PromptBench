"""Communications pair: centralized LCB with freeform inter-agent reports."""

from communications import communication_formats

communication_formats.install(globals(), topology="centralized", dataset="lcb", fmt="freeform")
