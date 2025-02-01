import cupy as cp

""" here im going to dump all the functions that dont have to do with the specific project """


def format_time(seconds):

    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    seconds = seconds % 60



    return f"{int(hours)}:{int(minutes)}:{seconds}"



def safe_norm(x, axis=-1):
    """
    Computes the L2 norm safely using float32 to prevent overflow in float16 computations.
    
    Args:
        x (cp.array): Input array.
        axis (int): Axis along which to compute the norm.

    Returns:
        cp.array: Computed norms (returned as float16 to save memory).
    """
    return cp.sqrt(cp.sum(x.astype(cp.float32) ** 2, axis=axis)).astype(cp.float16)  # Compute in float32, store in float16

if __name__ == "__main__":

    print(format_time(10000.123486))
    print (100//6)