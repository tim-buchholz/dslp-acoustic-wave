from mpi4py import MPI
import numpy as np
import scipy.sparse as scp
import scipy.sparse.linalg as scplin
from petsc4py import PETSc


def get_scipy_sparse_matrix(dfPet_mat, bool_save=False, filename="spmat.npz"):
    # function copies the values!

    # get sparse CSR indices
    indptr, indices, data = dfPet_mat.getValuesCSR()
    # construct scipy sparse csr matrix
    scp_mat = scp.csr_matrix((data, indices, indptr))

    if bool_save:
        # save scipy sparse csr matrix in compressed file
        out_mat = filename
        scp.save_npz(out_mat, scp_mat)

    return scp_mat


def scipy_csr_to_petsc_matrix(scipy_csr):
    num_rows, num_cols = scipy_csr.shape
    # Extract the data from the scipy sparse matrix
    csr_data = scipy_csr.data
    csr_indices = scipy_csr.indices
    csr_indptr = scipy_csr.indptr

    # Create a PETSc matrix and set it up
    petsc_matrix = PETSc.Mat().createAIJ([num_rows, num_cols])
    petsc_matrix.setUp()

    # Populate the PETSc matrix with the extracted data
    for i in range(num_rows):
        row_start = csr_indptr[i]
        row_end = csr_indptr[i + 1]
        cols = csr_indices[row_start:row_end]
        values = csr_data[row_start:row_end]
        petsc_matrix.setValues(i, cols, values)

    # Assemble the PETSc matrix
    petsc_matrix.assemble()

    return petsc_matrix


def sum_rows_to_diagonal(original_matrix):
    comm = MPI.COMM_SELF
    n = original_matrix.getSize()[0]

    # Create a new diagonal matrix in sparse format
    diagonal_matrix = PETSc.Mat().create(comm=comm)
    diagonal_matrix.setSizes([n, n])
    diagonal_matrix.setType(PETSc.Mat.Type.SEQAIJ)
    diagonal_matrix.setFromOptions()
    diagonal_matrix.setUp()

    # Iterate over each row and calculate the sum
    for row in range(n):
        row_sum = sum(original_matrix.getRow(row)[1])
        # Set the diagonal entry in the new matrix
        diagonal_matrix.setValue(row, row, row_sum)

    # Assemble the new matrix
    diagonal_matrix.assemble()

    return diagonal_matrix


def is_diagonal_matrix(matrix):
    # Check if the matrix is square
    if matrix.shape[0] != matrix.shape[1]:
        return False

    # Check if all off-diagonal elements are zero
    off_diag_elements = np.nonzero(matrix - np.diag(np.diagonal(matrix)))
    if len(off_diag_elements[0]) > 0:
        return False

    return True
