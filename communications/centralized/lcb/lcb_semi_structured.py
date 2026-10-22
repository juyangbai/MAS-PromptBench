"""Communications pair: centralized LCB with semi_structured inter-agent reports."""

from communications import communication_formats

communication_formats.install(globals(), topology="centralized", dataset="lcb", fmt="semi_structured")
