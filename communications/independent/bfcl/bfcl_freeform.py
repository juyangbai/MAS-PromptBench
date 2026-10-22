"""Communications pair: independent BFCL with freeform inter-agent reports."""

from communications import communication_formats

communication_formats.install(globals(), topology="independent", dataset="bfcl", fmt="freeform")
