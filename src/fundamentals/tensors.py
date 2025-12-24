import torch


def main():
    print("PyTorch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    print("-" * 50)

    # =========================================================
    # A. Tensor creation & Shapes
    # =========================================================

    # 1. Scalar (0-D tensor)
    scalar = torch.tensor(7)
    print("Scalar:", scalar)
    print("Shape:", scalar.shape)
    print("Dtype:", scalar.dtype)
    print()

    # 2. Vector (1-D tensor)
    vector = torch.tensor([1, 2, 3])
    print("Vector:", vector)
    print("Shape:", vector.shape)
    print("Dtype:", vector.dtype)
    print()

    # 3. Matrix (2-D tensor)
    matrix = torch.tensor([[1, 2, 3],
                            [4, 5, 6]])
    print("Matrix:\n", matrix)
    print("Shape:", matrix.shape)
    print("Dtype:", matrix.dtype)
    print()

    # 4. 3D Tensor
    tensor_3d = torch.tensor([
        [[1, 2], [3, 4]],
        [[5, 6], [7, 8]]
    ])
    print("3D Tensor:\n", tensor_3d)
    print("Shape:", tensor_3d.shape)
    print("Dtype:", tensor_3d.dtype)
    print()

    # =========================================================
    # B. Indexing & Slicing
    # =========================================================

    # 1. Indexing a single element
    print("Single element (matrix[0, 1]):", matrix[0, 1])
    print()

    # 2. Row slicing
    print("First row:", matrix[0])
    print("Second column:", matrix[:, 1])
    print()

    # 3. Slicing a sub-matrix
    sub_matrix = matrix[:, 1:]
    print("Sub-matrix (all rows, cols 1+):\n", sub_matrix)
    print()

    # =========================================================
    # C. Broadcasting
    # =========================================================

    mat = torch.tensor([[1, 2, 3],
                        [4, 5, 6]])
    vec = torch.tensor([10, 20, 30])

    print("Matrix shape:", mat.shape)
    print("Vector shape:", vec.shape)

    result = mat + vec
    print("Broadcasted result:\n", result)
    print()
    # Broadcasting works because PyTorch automatically expands
    # the vector to match the matrix shape along dimension 0.

    # =========================================================
    # D. Matrix multiplication
    # =========================================================

    a = torch.tensor([[1, 2],
                      [3, 4]])
    b = torch.tensor([[5, 6],
                      [7, 8]])

    matmul_1 = torch.matmul(a, b)
    matmul_2 = a @ b

    print("Matrix A shape:", a.shape)
    print("Matrix B shape:", b.shape)
    print("Matmul result (torch.matmul):\n", matmul_1)
    print("Matmul result (@ operator):\n", matmul_2)
    print()
    # Matrix multiplication combines rows and columns.
    # This is different from element-wise multiplication (*),
    # which multiplies corresponding elements only.

    # =========================================================
    # E. Device placement (CPU / GPU)
    # =========================================================

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tensor_on_device = torch.tensor([1.0, 2.0, 3.0]).to(device)

    print("Tensor device:", tensor_on_device.device)
    print("-" * 50)


if __name__ == "__main__":
    main()
