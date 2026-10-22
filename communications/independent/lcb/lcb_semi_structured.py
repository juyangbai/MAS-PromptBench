"""Communications pair: independent LCB with semi_structured inter-agent reports."""

from communications import communication_formats

communication_formats.install(globals(), topology="independent", dataset="lcb", fmt="semi_structured")
