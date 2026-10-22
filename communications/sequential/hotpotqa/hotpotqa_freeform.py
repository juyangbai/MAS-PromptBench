"""Communications pair: sequential HotpotQA with freeform inter-agent reports."""

from communications import communication_formats

communication_formats.install(globals(), topology="sequential", dataset="hotpotqa", fmt="freeform")
