"""Pre-AOT-partitioning FX passes.

These passes run on the joint graph AFTER torch's built-in
``joint_graph_passes`` and the Vulkan device-stamping pass, but BEFORE
the AOT partitioner splits the graph into forward / backward sub-graphs.

They tag individual nodes with ``partitioner_tag`` metadata so the
min-cut partitioner places them correctly.
"""

from .mark_0d_div import mark_0d_div_must_be_in_forward
