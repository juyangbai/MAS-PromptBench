"""Communications pair: independent API-Bank with semi_structured inter-agent reports."""

from communications import communication_formats

communication_formats.install(globals(), topology="independent", dataset="apibank", fmt="semi_structured")
