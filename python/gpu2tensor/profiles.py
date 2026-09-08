"""Decode vendor measurements without importing a device SDK."""


def proton_kernels(profile):
    kernels = []

    def visit(node):
        metrics = node.get("metrics", {})
        # ROOT and user scopes may carry inclusive summary metrics.
        if "time (ns)" in metrics and metrics.get("device_type") == "CUDA":
            kernels.append({"name": node.get("frame", {}).get("name", "unknown"),
                            "duration_ns": metrics["time (ns)"], "metrics": metrics})
        for child in node.get("children", []):
            visit(child)

    for node in profile:
        if isinstance(node, dict):
            visit(node)
    if not kernels:
        raise RuntimeError("Proton returned no CUDA kernel measurements.")
    return kernels
