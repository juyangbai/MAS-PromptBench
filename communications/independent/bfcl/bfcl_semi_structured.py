"""Communications pair: independent BFCL with semi_structured inter-agent reports."""

from communications import communication_formats

communication_formats.install(globals(), topology="independent", dataset="bfcl", fmt="semi_structured")
