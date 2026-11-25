"""Retain the selected centered-value and representative tile launch shapes."""

from ..workspace import active_call, count_launch


def launch_merge(kernel, grid, *args, **options):
    call = active_call()
    stages = call.profile.representative_stages if call and call.optimized and kernel.fn.__name__ == "rep_merge" else None
    if stages is not None:
        options["num_stages"] = stages
    result = kernel[grid](*args, **options)
    if stages is not None:
        count_launch("rep")
    return result


def launch_centered(kernel, grid, *args, **options):
    call = active_call()
    tiles = call.profile.centered_tiles if call and call.optimized else None
    if tiles is None or len(args) != 8 or len(grid) != 2 or args[7] != 128:
        return kernel[grid](*args, **options)

    import triton

    bk, bv = tiles
    grid = (triton.cdiv(args[7], bk) * triton.cdiv(args[7], bv), grid[1])
    result = kernel[grid](*args, **dict(options, BK=bk, BV=bv))
    count_launch("centered")
    return result
