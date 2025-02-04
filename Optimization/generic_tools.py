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

    #print(format_time(10000.123486))
    #print (100//6)
    a = cp.array([[1,2,3],
                  [4,5,6],
                  [7,8,9]])
    
    b = cp.array([[0,1,2,3,4],
                  [5,6,7,8,9]])
    
    pos = cp.array([[[1,1,1],[2,2,2],[3,3,3]],
                    [[4,4,4],[5,5,5],[6,6,6]],
                    [[7,7,7],[8,8,8],[9,9,9]]])
    #print(a[0:5,:])
    print(b[0:10,2])
    print(b[1,1])
    print(pos[0:2,:,:])