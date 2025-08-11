import cutlass.cute as cute


@cute.jit
def compare_and_swap(arr: cute.Tensor, i: int, j: int, ascending: bool = True) -> None:
    """Compare and swap elements at indices i and j in ascending or descending order."""
    a, b = arr[i], arr[j]
    if (a > b) ^ (not ascending):
        arr[i] = b
        arr[j] = a
    # if cutlass.const_expr(ascending):
    #     if a > b:
    #         arr[i] = b
    #         arr[j] = a
    # else:
    #     if a < b:
    #         arr[i] = b
    #         arr[j] = a
