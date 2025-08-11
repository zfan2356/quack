import cutlass
import cutlass.cute as cute


@cute.jit
def compare_and_swap(
    arr: cute.Tensor, i: int, j: int, ascending: cutlass.Constexpr[bool] = True
) -> None:
    """Compare and swap elements at indices i and j in ascending or descending order."""
    a, b = arr[i], arr[j]
    if cutlass.const_expr(ascending):
        if a > b:
            arr[i] = b
            arr[j] = a
    else:
        if a < b:
            arr[i] = b
            arr[j] = a
