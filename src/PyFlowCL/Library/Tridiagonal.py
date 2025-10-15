"""
------------------------------------------------------------------------
PyFlowCL: A Python-native, compressible Navier-Stokes solver for
curvilinear grids
------------------------------------------------------------------------

@file Tridiagonal.py

"""

__copyright__ = """
Copyright (c) 2022 Jonathan F. MacArt
"""

__license__ = """
 Permission is hereby granted, free of charge, to any person 
 obtaining a copy of this software and associated documentation 
 files (the "Software"), to deal in the Software without 
 restriction, including without limitation the rights to use, 
 copy, modify, merge, publish, distribute, sublicense, and/or 
 sell copies of the Software, and to permit persons to whom the 
 Software is furnished to do so, subject to the following conditions:

 The above copyright notice and this permission notice shall be 
 included in all copies or substantial portions of the Software.

 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, 
 EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES 
 OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND 
 NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT 
 HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, 
 WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING 
 FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR 
 OTHER DEALINGS IN THE SOFTWARE.
"""


import torch

import solver_cpp

def solve( decomp, dir, a, b, c, r, device ):
    """General parallel solver for Ax=r for x with tridigonal A having
    diagonals a, b, c

    USAGE:
        x = solve( decomp, dir, a, b, c, r )

    INPUT:
        decomp     - PyFlowCL parallel decomposition object
        dir        - Cartesian direction along which to solve the system
        a, b, c    - PyTorch tensors specifying the diagonals of the
                     tridiagonal matrix A.  
        r[:,N]     - vector of N right-hand-sides

    OUTPUT:
        x[:,N]     - solution vectors returned in storage of r

    """
    from . import Pentadiagonal

    # Get parallel info
    if ( dir == 'x' ):
        proc = decomp.npx
        rank = decomp.iproc
        ncom = decomp.cartCommX
        nper = decomp.isper[0]
    elif ( dir == 'y' ):
        proc = decomp.npy
        rank = decomp.jproc
        ncom = decomp.cartCommY
        nper = decomp.isper[1]
    elif ( dir == 'z' ):
        proc = decomp.npz
        rank = decomp.kproc
        ncom = decomp.cartCommZ
        nper = decomp.isper[2]
    else:
        raise Exception("Tridiagonal.solve: unknown direction")

    n,n2,n3,nvar = r.shape
    lot = n2 * n3 * nvar
    
    # If serial
    if ( proc == 1 ):
        if ( nper ):
            #solve_periodic_serial( a, b, c, r )
            solver_cpp.tridiagonal_periodic_serial(a, b, c, r, n, lot)
        else:
            #solve_serial_nonfactored( a, b, c, r )
            solver_cpp.tridiagonal_serial_nonfactored(a, b, c, r, n, lot)
        return

    # Partition the lot
    # Leading dimension is rank of the tridiagonal system to be solved
    if (lot < proc):
        raise Exception("Tridiagonal.solve: cannot handle so many proc for such a small problem.")
    ngroup  = torch.ones((proc,), dtype=torch.int) * (lot // proc)
    nremain = lot % proc
    ngroup[:nremain] += 1
    nlot = int(ngroup[0])

    # Initialize boundary values
    s1 = torch.zeros_like(b).to(device)
    s2 = torch.zeros_like(b).to(device)
    s1[0] = a[0]
    s2[n-1] = c[n-1]
    if (not nper):
        if (rank==0): s1[0] = 0.0
        if (rank==proc-1): s2[n-1] = 0.0

    # Forward & backward elimination
    bc = torch.clone(b)
    #parallel_step1( a, bc, c, s1, s2, r, n )
    solver_cpp.tridiagonal_parallel_step_1( a, bc, c, s1, s2, r, n, lot )

    # All dependence has been shifted to the boundary elements. Now
    # communicate boundary values to the root process and solve
    # reduced pentadiagonal systems. Use of a pentadiagonal system is
    # more robust than the reordered tridiagonal system.
    #
    # Send rows of the pentadiagonal system
    # [ 0, s1, b, 0, s2; r ]
    #     [s1, 0, b, s2, 0; r ]
    #
    sendbuf1 = torch.zeros((2*proc, 5),    dtype=torch.float64).to(device)
    recvbuf1 = torch.zeros((2*proc, 5),    dtype=torch.float64).to(device)
    sendbuf2 = torch.zeros((2*proc, nlot), dtype=torch.float64).to(device)
    recvbuf2 = torch.zeros((2*proc, nlot), dtype=torch.float64).to(device)

    # Coefficients -- same for each problem
    sendbuf1[0::2,0] = 0.0
    sendbuf1[0::2,1] = s1[0]
    sendbuf1[0::2,2] = bc[0]
    sendbuf1[0::2,3] = 0.0
    sendbuf1[0::2,4] = s2[0]
    sendbuf1[1::2,0] = s1[n-1]
    sendbuf1[1::2,1] = 0.0
    sendbuf1[1::2,2] = bc[n-1]
    sendbuf1[1::2,3] = s2[n-1]
    sendbuf1[1::2,4] = 0.0
        
    L  = 0
    k1 = 0
    r_first = torch.ravel( r[0  ,:,:,:] )
    r_last  = torch.ravel( r[n-1,:,:,:] )
    for igroup in range( 0, proc ):
        nk = ngroup[igroup]
        k2 = k1 + nk

        # RHS problems
        sendbuf2[2*igroup  ,:nk] = r_first[k1:k2]
        sendbuf2[2*igroup+1,:nk] = r_last [k1:k2]

        k1 = k2

    # Gather the boundary data
    #ncom.Alltoall(sendbuf1.cpu(), recvbuf1.cpu()); recvbuf1 = recvbuf1.to(device)
    #ncom.Alltoall(sendbuf2.cpu(), recvbuf2.cpu()); recvbuf2 = recvbuf2.to(device)
    ncom.Alltoall(sendbuf1, recvbuf1)
    ncom.Alltoall(sendbuf2, recvbuf2)

    # Clear unused values
    recvbuf2[:,ngroup[rank]:] = 0.0

    # Solve reduced systems
    if nper:
        #Pentadiagonal.solve_periodic_serial(
        #    recvbuf1[:,0], recvbuf1[:,1], recvbuf1[:,2],
        #    recvbuf1[:,3], recvbuf1[:,4], recvbuf2,
        #    sendbuf1[:,0], sendbuf1[:,1] )
        solver_cpp.pentadiagonal_periodic_serial(
            recvbuf1[:,0], recvbuf1[:,1], recvbuf1[:,2],
            recvbuf1[:,3], recvbuf1[:,4], recvbuf2,
            sendbuf1[:,0], sendbuf1[:,1], 2*proc, nlot )
    else:
        #Pentadiagonal.solve_serial(
        #    recvbuf1[1:2*proc-1,0], recvbuf1[1:2*proc-1,1], recvbuf1[1:2*proc-1,2],
        #    recvbuf1[1:2*proc-1,3], recvbuf1[1:2*proc-1,4], recvbuf2[1:2*proc-1,:] )
        solver_cpp.pentadiagonal_serial(
            recvbuf1[1:2*proc-1,0], recvbuf1[1:2*proc-1,1], recvbuf1[1:2*proc-1,2],
            recvbuf1[1:2*proc-1,3], recvbuf1[1:2*proc-1,4], recvbuf2[1:2*proc-1,:], 2*proc-2, nlot )

    # Solution is in recvbuf2
    # Permute the order
    for i in range( 1, proc ):
        const = torch.clone(recvbuf2[2*i-1,:])
        recvbuf2[2*i-1,:] = recvbuf2[2*i,:]
        recvbuf2[2*i,:] = const

    # If periodic, don't forget end points
    if nper:
        const = torch.clone(recvbuf2[0,:])
        recvbuf2[0,:] = recvbuf2[2*proc-1,:]
        recvbuf2[2*proc-1,:] = const

    # Scatter back the solution
    #ncom.Alltoall(recvbuf2.cpu(), sendbuf2.cpu()); sendbuf2 = sendbuf2.to(device)
    ncom.Alltoall(recvbuf2, sendbuf2)

    r1 = torch.zeros((lot,), dtype=torch.float64).to(device)
    r2 = torch.zeros((lot,), dtype=torch.float64).to(device)
    k1 = 0
    for igroup in range( 0, proc ):
        nk = ngroup[igroup]
        k2 = k1 + nk

        r1[k1:k2] = sendbuf2[2*igroup  ,:nk]
        r2[k1:k2] = sendbuf2[2*igroup+1,:nk]

        k1 = k2

    # Only if not periodic
    if (not nper):
        if (rank==0):      r1[:] = 0.0
        if (rank==proc-1): r2[:] = 0.0

    # Forward-substitution
    #parallel_step3( bc, r1.reshape(n2,n3,nvar), r2.reshape(n2,n3,nvar), s1, s2, r, n, n2, n3 )
    solver_cpp.tridiagonal_parallel_step_3( bc, r1.reshape(n2,n3,nvar), r2.reshape(n2,n3,nvar),
                                            s1, s2, r, n, lot )

    # Clean up
    del r1,r2
    del ngroup,sendbuf1,sendbuf2,recvbuf1,recvbuf2
    del s1,s2,bc

    return


@torch.jit.script
def parallel_step1( a, bc, c, s1, s2, r, n : int ):
    # Forward elimination
    # Upper boundary is in s1[i]
    for i in range( 1, n ):
        const    = a[i] / bc[i-1]
        bc[i]   -= c[i-1] * const
        r[i,:,:,:] = r[i,:,:,:] - r[i-1,:,:,:] * const
        s1[i]    = -s1[i-1] * const

    # Backward elimination
    # Lower boundary is in s2[i]
    for i in range( n-2, -1, -1 ):
        const = c[i] / bc[i+1]
        r[i,:,:,:] = r[i,:,:,:] - r[i+1,:,:,:] * const
        s1[i] -= s1[i+1] * const
        s2[i] = -s2[i+1] * const

    return


@torch.jit.script
def parallel_step3( bc, r1, r2, s1, s2, r,
                    n : int, n2 : int, n3 : int ):
    # Forward-substitution
    for i in range( 0, n ):
        r[i,:,:,:] = ( r[i,:,:,:] -
                     s1[i] * r1 -
                     s2[i] * r2 ) / bc[i]
    return


@torch.jit.script
def factor( a, b, c ):
    """
    Performs LU factorization on tridiagonal matrix A

    USAGE:
        factor( a, b, c )

    INPUT:
        a, b, c    - PyTorch tensors specifying the diagonals of the
                     tridiagonal matrix A.  a is the subdiagonal with a[0]
                     being the A[1,0] value, b is the main diagonal with
                     b[0] being the A[0,0] value and c is the superdiagonal
                     with c[0] being the A[0,1] value.

    OUTPUT:
        a, b, c    - arrays containing the data for the factored matrix

    NOTE:
        For this to be sure to work A should be strictly diagonally
        dominant, meaning that |b(i)| > |a(i-1)| + |c(i)| for each i.
        This ensures that pivoting will not be necessary.

    DIMENSIONS:
        a = [1,n-1]
        b = [0,n-1]
        c = [0,n-2]
    """

    for i in range(1, len(b)):
        a[i-1] /= b[i-1]
        b[i]   -= a[i-1] * c[i-1]

    return


@torch.jit.script
def solve_serial( a, b, c, r ):
    """
    Solves Ax=r for x with factored tridigonal A having diagonals a, b, c

    USAGE:
        x = solve( a, b, c, r )

    INPUT:
        a, b, c    - PyTorch tensors specifying the diagonals of the
                     factored tridiagonal matrix A.  These are produced by
                     factor().
        r[:,N]     - vector of N right-hand-sides

    OUTPUT:
        x[:,N]     - solution vectors returned in storage of r
    """

    n = r.shape[0]
    
    # Solve
    for i in range(1, n):
        r[i,:] = r[i,:] - a[i-1] * r[i-1,:]

    r[n-1,:] /= b[n-1]

    for i in range( n-2, -1, -1 ):
        r[i,:] = ( r[i,:] - c[i] * r[i+1,:] ) / b[i]

    return


@torch.jit.script
def solve_serial_nonfactored( a, b, c, r ):
    """
    Solves Ax=r for x with tridigonal A having diagonals a, b, c

    USAGE:
        x = solve( a, b, c, r )

    INPUT:
        a, b, c    - PyTorch tensors specifying the diagonals of the
                     NON-factored tridiagonal matrix A. 
        r[:,N]     - vector of N right-hand-sides

    OUTPUT:
        x[:,N]     - solution vectors returned in storage of r
    """

    ac = torch.clone( a )
    bc = torch.clone( b )

    # Factor the matrix A
    for i in range(1, len(b)):
        ac[i-1] /= bc[i-1]
        bc[i]   -= ac[i-1] * c[i-1]

    n = r.shape[0]
    
    # Solve
    for i in range(1, n):
        r[i,:] = r[i,:] - ac[i-1] * r[i-1,:]

    r[n-1,:] /= bc[n-1]

    for i in range( n-2, -1, -1 ):
        r[i,:] = ( r[i,:] - c[i] * r[i+1,:] ) / bc[i]

    del ac, bc

    return


@torch.jit.script
def solve_periodic_serial( a, b, c, r ):
    """
    Solves Ax=r for x with tridigonal A having diagonals a, b, c
    with periodic boundary conditions

    USAGE:
        x = solve( a, b, c, r )

    INPUT:
        a, b, c    - PyTorch tensors specifying the diagonals of the
                     tridiagonal matrix A.  
        r[:,N]     - vector of N right-hand-sides

    OUTPUT:
        x[:,N]     - solution vectors returned in storage of r
    """

    n = r.shape[0]

    ac = torch.clone( a )
    bc = torch.clone( b )
        
    # Forward elimination
    for i in range(1, n-2):
        const  = ac[i] / bc[i-1]
        bc[i]  = bc[i]  - c[i-1]   * const        
        r[i,:] = r[i,:] - r[i-1,:] * const
        # Boundary is stored in ac[i]
        ac[i] = -ac[i-1] * const

    i = n-2
    const  = ac[i] / bc[i-1]
    bc[i]  = bc[i]  - c[i-1]   * const        
    r[i,:] = r[i,:] - r[i-1,:] * const
    ac[i]  = c[i]  - ac[i-1]   * const
    i = n-1
    const    = ac[i] / bc[i-1]
    r[i,:] = r[i,:] - r[i-1,:] * const
    ac[i]  = bc[i] - ac[i-1]   * const

    # Backward elimination
    for i in range( n-3, -1, -1 ):
        const  = c[i] / bc[i+1]
        r[i,:] = r[i,:] - r[i+1,:] * const
        ac[i]  = ac[i] - ac[i+1]   * const

    # Eliminate oddball
    const = c[n-1] / bc[0]
    r[ n-1,:] =  r[n-1,:] - r[0,:] * const
    ac[n-1]   = ac[n-1]  - ac[0]   * const

    # Backward substitution
    r[n-1,:] /= ac[n-1]

    for i in range( n-2, -1, -1 ):
        r[i,:] = ( r[i,:] - ac[i] * r[n-1,:] ) / bc[i]

    del ac,bc

    return
