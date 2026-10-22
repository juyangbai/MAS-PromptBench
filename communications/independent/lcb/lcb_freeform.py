"""Communications pair: independent LCB with freeform inter-agent reports."""

from communications import communication_formats

communication_formats.install(globals(), topology="independent", dataset="lcb", fmt="freeform")
