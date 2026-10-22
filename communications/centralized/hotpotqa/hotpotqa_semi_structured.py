"""Communications pair: centralized HotpotQA with semi_structured inter-agent reports."""

from communications import communication_formats

communication_formats.install(globals(), topology="centralized", dataset="hotpotqa", fmt="semi_structured")
