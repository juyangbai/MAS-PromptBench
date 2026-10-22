"""Communications pair: centralized BFCL with freeform inter-agent reports."""

from communications import communication_formats

communication_formats.install(globals(), topology="centralized", dataset="bfcl", fmt="freeform")
