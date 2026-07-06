"""Skip adapter tests for QRules versions that lack `~qrules.topology.FrozenTransition`."""

from ampform._qrules import get_qrules_version

collect_ignore = []
if get_qrules_version() < (0, 10):
    collect_ignore.append("test_qrules.py")
