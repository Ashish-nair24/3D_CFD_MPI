"""
------------------------------------------------------------------------
PyFlowCL: A Python-native, compressible Navier-Stokes solver for
curvilinear grids
------------------------------------------------------------------------

@file Pentadiagonal.py

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


@torch.jit.script
def solve_serial( a, b, c, d, e, r ):
    """
    Solves Ax=r for x with pentadiagonal A having diagonals a, b, c, d, e

    Warning! Overwrites storage of b, c, and d!

    USAGE:
        x = solve( a, b, c, d, e, r )

    INPUT:
        a,b,c,d,e  - PyTorch tensors specifying the diagonals of the
                     pentadiagonal matrix A.
        r[:,N]     - vector of N right-hand-sides

    OUTPUT:
        x[:,N]     - solution vectors returned in storage of r
    """

    n  = r.shape[0]
    if (n==1):
        # Solve 1x1 system
        r[0,:] = r[0,:] / c[0]
        return

    if (n==2):
        # Solve 2x2 system
        const  = b[1] / c[0]
        c[1] -= d[0] * const
        r[1,:] = r[1,:] - r[0,:] * const
        r[1,:] = r[1,:] / c[1]
        r[0,:] = (r[0,:] - d[0] * r[1,:]) / c[0]
        
        return

    # Forward elimination
    for i in range( 0, n-2 ):
        # Eliminate b[i+1]
        const = b[i+1] / c[i]
        c[i+1] -= d[i] * const
        d[i+1] -=  e[i] * const
        r[i+1,:] = r[i+1,:] - r[i,:] * const

        # Eliminate a[i+2]
        const = a[i+2] / c[i]
        b[i+2] -= d[i] * const
        c[i+2] -=  e[i] * const
        r[i+2,:] = r[i+2,:] - r[i,:] * const

    # Eliminate A[1,n-1]
    const = b[n-1] / c[n-2]
    c[n-1] -= d[n-2] * const
    r[n-1,:] = r[n-1,:] - r[n-2,:] * const

    # Back-substitution
    r[n-1,:] = r[n-1,:] / c[n-1]
    r[n-2,:] = ( r[n-2,:] - d[n-2] * r[n-1,:] ) / c[n-2]
    for i in range( n-3, -1, -1 ):
        r[i,:] = ( r[i,:] - d[i] * r[i+1,:] - e[i] * r[i+2,:] ) / c[i]

    return


@torch.jit.script
def solve_periodic_serial( a, b, c, d, e, r, s1, s2 ):
    """
    Solves Ax=r for x with pentadiagonal A having diagonals a, b, c, d, e
    with periodic boundary conditions

    Warning! Overwrites storage of b, c, and d!

    USAGE:
        x = solve( a, b, c, d, e, r )

    INPUT:
        a,b,c,d,e  - PyTorch tensors specifying the diagonals of the
                     pentadiagonal matrix A.
        r[:,N]     - vector of N right-hand-sides
        s1, s2     - Workspace arrays on same device as a-r

    OUTPUT:
        x[:,N]     - solution vectors returned in storage of r
    """

    n  = r.shape[0]
    if (n==1):
        # Solve 1x1 system
        r = r / (a + b + c + d + e)
        return

    if (n==2):
        # Solve 2x2 system
        c += a + e
        d[0] += b[0]
        b[1] += d[1]
        const = b[1] / c[0]
        c[1] -= d[0] * const
        r[1,:] = r[1,:] - r[0,:] * const
        r[1,:] = r[1,:] / c[1]
        r[0,:] = (r[0,:] - d[0] * r[1,:]) / c[0]
        return
    #elif (n==3):
    #    # Solve 3x3 system
    #    b += e
    #    d += a
    #    from . import Tridiagonal
    #    Tridiagonal.solve_periodic_serial(b, c, d, r)
    #    
    #    return
    elif (n==4):
        a   += e
        e[:] = 0.0
    
    # Initialize boundary data
    s1[:] = 0.0
    s1[0] = a[0]
    s1[n-4] = s1[n-4] + e[n-4]
    s1[n-3] = d[n-3]
    s1[n-2] = c[n-2]
    s1[n-1] = b[n-1]
    
    s2[:] = 0.0
    s2[0] = b[0]
    s2[1] = a[1]
    s2[n-3] = s2[n-3] + e[n-3]
    s2[n-2] = d[n-2]
    s2[n-1] = c[n-1]

    # Forward elimination
    for i in range( 0, n-2 ):
        # Eliminate b[i+1]
        const = b[i+1] / c[i]
        c[i+1] -= d[i] * const
        d[i+1] -=  e[i] * const
        s1[i+1] -= s1[i] * const
        s2[i+1] -= s2[i] * const
        r[i+1,:] = r[i+1,:] - r[i,:] * const

        # Eliminate a[i+2]
        const = a[i+2] / c[i]
        b[i+2] -= d[i] * const
        c[i+2] -=  e[i] * const
        s1[i+2] -= s1[i] * const
        s2[i+2] -= s2[i] * const
        r[i+2,:] = r[i+2,:] - r[i,:] * const
        
    # Backward elimination
    for i in range( n-3, 1, -1 ):
        # Eliminate d[i-1]
        const = d[i-1] / c[i]
        r[i-1,:] = r[i-1,:] - r[i,:] * const
        s1[i-1] -= s1[i] * const
        s2[i-1] -= s2[i] * const

        # Eliminate e[i-2]
        const = e[i-2] / c[i]
        r[i-2,:] = r[i-2,:] - r[i,:] * const
        s1[i-2] -= s1[i] * const
        s2[i-2] -= s2[i] * const

    # Eliminate d[i-1]
    i = 1
    const = d[i-1] / c[i]
    r[i-1,:] = r[i-1,:] - r[i,:] * const
    s1[i-1] -= s1[i] * const
    s2[i-1] -= s2[i] * const

    # Eliminate oddball region
    const = e[n-2] / c[0]
    r[n-2,:] = r[n-2,:] - r[0] * const
    s1[n-2] -= s1[0] * const
    s2[n-2] -= s2[0] * const

    const = d[n-1] / c[0]
    r[n-1,:] = r[n-1,:] - r[0] * const
    s1[n-1] -= s1[0] * const
    s2[n-1] -= s2[0] * const

    const = e[n-1] / c[1]
    r[n-1,:] = r[n-1,:] - r[1] * const
    s1[n-1] -= s1[1] * const
    s2[n-1] -= s2[1] * const

    # Eliminate corner region
    const = s1[n-1] / s1[n-2]
    r[n-1,:] = r[n-1,:] - r[n-2,:] * const
    s2[n-1] -= s2[n-2] * const

    # Back-substitution
    r[n-1,:] = r[n-1,:] / s2[n-1]
    r[n-2,:] = ( r[n-2,:] - s2[n-2] * r[n-1,:] ) / s1[n-2]
    for i in range( n-3, -1, -1 ):
        r[i,:] = ( r[i,:] - s1[i] * r[n-2,:] - s2[i] * r[n-1,:] ) / c[i]

    return
