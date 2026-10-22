"""Communications pair: centralized BFCL with semi_structured inter-agent reports."""

from communications import communication_formats

communication_formats.install(globals(), topology="centralized", dataset="bfcl", fmt="semi_structured")
