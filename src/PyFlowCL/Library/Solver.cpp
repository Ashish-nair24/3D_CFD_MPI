/*
------------------------------------------------------------------------
PyFlowCL: A Python-native, compressible Navier-Stokes solver for
curvilinear grids
------------------------------------------------------------------------

@file Solver.cpp

Copyright (c) 2022 Jonathan F. MacArt

LICENSE:
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
*/
  

#include <torch/extension.h>


void tridiagonal_serial_nonfactored( torch::Tensor a,
					   torch::Tensor b,
					   torch::Tensor c,
					   torch::Tensor r,
					   int64_t n,
					   int64_t lot ) {
  /*
    Solves Ax=r for x with tridigonal A having diagonals a, b, c

    USAGE:
        x = solve( a, b, c, r )

    INPUT:
        a, b, c    - PyTorch tensors specifying the diagonals of the
                     NON-factored tridiagonal matrix A. 
        r[:,N]     - vector of N right-hand-sides

    OUTPUT:
        x[:,N]     - solution vectors returned in storage of r
  */

  auto ac = torch::clone(a);
  auto bc = torch::clone(b);

  auto slice = torch::indexing::Slice(0,lot);

  // Factor the matrix A
  for (int i=1; i<n; i++) {
    ac[i-1] /= bc[i-1];
    bc[i]   -= ac[i-1] * c[i-1];
  }

  // Solve
  for (int i=1; i<n; i++) {
    // r[i,:] = r[i,:] - ac[i-1] * r[i-1,:]
    r.index({ i, slice }) = r.index({ i, slice }) - ac[i-1] * r.index({ i-1, slice });
  }

  r.index({ n-1, slice }) /= bc[n-1];

  // Back-substitution
  for (int i=n-2; i>=0; i--) {
    r.index({ i, slice }) = ( r.index({ i, slice }) - c[i] *
			      r.index({ i+1, slice }) ) / bc[i];
  }

  return;
}


void tridiagonal_periodic_serial( torch::Tensor a,
				  torch::Tensor b,
				  torch::Tensor c,
				  torch::Tensor r,
				  int64_t n,
				  int64_t lot ) {
  /*
    Solves Ax=r for x with tridigonal A having diagonals a, b, c
    with periodic boundary conditions

    USAGE:
        x = solve( a, b, c, r )

    INPUT:
        a, b, c    - PyTorch tensors specifying the diagonals of the
                     NON-factored tridiagonal matrix A. 
        r[:,N]     - vector of N right-hand-sides

    OUTPUT:
        x[:,N]     - solution vectors returned in storage of r
  */

  auto ac = torch::clone(a);
  auto bc = torch::clone(b);

  auto slice = torch::indexing::Slice(0,lot);

  // Forward elimination
  for (int i=1; i<n-2; i++) {
    auto cc = ac[i] / bc[i-1];
    bc[i] = bc[i] - c[i-1] * cc;
    r.index({ i, slice }) = r.index({ i, slice }) - r.index({ i-1, slice }) * cc;
    // Boundary is stored in ac[i]
    ac[i] = -ac[i-1] * cc;
  }

  int ii = n-2;
  auto cc = ac[ii] / bc[ii-1];
  bc[ii] = bc[ii] - c[ii-1] * cc;
  r.index({ ii, slice }) = r.index({ ii, slice }) - r.index({ ii-1, slice }) * cc;
  ac[ii] = c[ii] - ac[ii-1] * cc;
  
  ii = n-1;
  cc = ac[ii] / bc[ii-1];
  r.index({ ii, slice }) = r.index({ ii, slice }) - r.index({ ii-1, slice }) * cc;
  ac[ii] = bc[ii] - ac[ii-1] * cc;

  // Backward elimination
  for (int i=n-3; i>=0; i--) {
    cc = c[i] / bc[i+1];
    r.index({ i, slice }) = r.index({ i, slice }) - r.index({ i+1, slice }) * cc;
    ac[i] = ac[i] - ac[i+1] * cc;
  }

  // Eliminate oddball
  cc = c[n-1] / bc[0];
  r.index({ n-1, slice }) = r.index({ n-1, slice }) - r.index({ 0, slice }) * cc;
  ac[n-1] = ac[n-1] - ac[0] * cc;

  // Backward substitution
  r.index({ n-1, slice }) /= ac[n-1];

  for (int i=n-2; i>=0; i--) {
    r.index({ i, slice }) = ( r.index({ i, slice }) -
			      ac[i] * r.index({ n-1, slice }) ) / bc[i];
  }

  return;
}


void pentadiagonal_serial( torch::Tensor a,
			   torch::Tensor b,
			   torch::Tensor c,
			   torch::Tensor d,
			   torch::Tensor e,
			   torch::Tensor r,
			   int64_t n,
			   int64_t lot ) {
  /*
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
  */

  auto slice = torch::indexing::Slice(0,lot);
  
  if (n==1) {
    // Solve 1x1 system
    //r[0,:] = r[0,:] / c[0];
    r.index({ 0, slice }) = r.index({ 0, slice }) / c.index({0});
    return;
  }

  if (n==2) {
    // Solve 2x2 system
    torch::Tensor cst  = b[1] / c[0];
    c[1] -= d[0] * cst;
    //r[1,:] = r[1,:] - r[0,:] * cst;
    //r[1,:] = r[1,:] / c[1];
    //r[0,:] = (r[0,:] - d[0] * r[1,:]) / c[0];
    r.index({ 1, slice }) = r.index({ 1, slice }) - r.index({ 0, slice }) * cst;
    r.index({ 1, slice }) = r.index({ 1, slice }) / c[1];
    r.index({ 0, slice }) = (r.index({ 0, slice }) - d[0] * r.index({ 1, slice })) / c[0];
    
    return;
  }

  // Forward elimination
  for (int i=0; i<n-2; i++) {
    // Eliminate b[i+1]
    torch::Tensor cst = b[i+1] / c[i];
    c[i+1] -= d[i] * cst;
    d[i+1] -=  e[i] * cst;
    //r[i+1,:] = r[i+1,:] - r[i,:] * cst;
    r.index({ i+1, slice }) = r.index({ i+1, slice }) - r.index({ i, slice }) * cst;

    // Eliminate a[i+2]
    cst = a[i+2] / c[i];
    b[i+2] -= d[i] * cst;
    c[i+2] -=  e[i] * cst;
    //r[i+2,:] = r[i+2,:] - r[i,:] * cst;
    r.index({ i+2, slice }) = r.index({ i+2, slice }) - r.index({ i, slice }) * cst;
  }

  // Eliminate A[1,n-1]
  torch::Tensor cst = b[n-1] / c[n-2];
  c[n-1] -= d[n-2] * cst;
  //r[n-1,:] = r[n-1,:] - r[n-2,:] * cst;
  r.index({ n-1, slice }) = r.index({ n-1, slice }) - r.index({ n-2, slice }) * cst;

  // Back-substitution
  //r[n-1,:] = r[n-1,:] / c[n-1];
  //r[n-2,:] = ( r[n-2,:] - d[n-2] * r[n-1,:] ) / c[n-2];
  r.index({ n-1, slice }) = r.index({ n-1, slice }) / c[n-1];
  r.index({ n-2, slice }) = ( r.index({ n-2, slice }) -
			      r.index({ n-1, slice }) * d[n-2] ) / c[n-2];
  for (int i=n-3; i>=0; i-- ) {
    //r[i,:] = ( r[i,:] - d[i] * r[i+1,:] - e[i] * r[i+2,:] ) / c[i];
    r.index({ i, slice }) = ( r.index({ i,   slice }) -
			      r.index({ i+1, slice }) * d[i] -
			      r.index({ i+2, slice }) * e[i] ) / c[i];
  }

  return;
}


void pentadiagonal_periodic_serial( torch::Tensor a,
				    torch::Tensor b,
				    torch::Tensor c,
				    torch::Tensor d,
				    torch::Tensor e,
				    torch::Tensor r,
				    torch::Tensor s1,
				    torch::Tensor s2,
				    int64_t n,
				    int64_t lot ) {
  /*
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
  */

  auto slice = torch::indexing::Slice(0,lot);
  
  if (n==1){
    // Solve 1x1 system
    r = r / (a + b + c + d + e);
    return;
  }

  if (n==2) {
    // Solve 2x2 system
    c += a + e;
    d[0] += b[0];
    b[1] += d[1];
    auto cst = b[1] / c[0];
    c[1] -= d[0] * cst;
    r.index({ 1, slice }) = r.index({ 1, slice }) - r.index({ 0, slice }) * cst;
    r.index({ 1, slice }) = r.index({ 1, slice }) / c[1];
    r.index({ 0, slice }) = (r.index({ 0, slice }) - d[0] * r.index({ 1, slice })) / c[0];
    return;
  }
  // 3x3 systems: revisit
  if (n==4) {
    a += e;
    e *= 0.0;
  }
     
  // Initialize boundary data
  s1 *= 0.0;
  s1[0] = a[0];
  s1[n-4] = s1[n-4] + e[n-4];
  s1[n-3] = d[n-3];
  s1[n-2] = c[n-2];
  s1[n-1] = b[n-1];
     
  s2 *= 0.0;
  s2[0] = b[0];
  s2[1] = a[1];
  s2[n-3] = s2[n-3] + e[n-3];
  s2[n-2] = d[n-2];
  s2[n-1] = c[n-1];
 
    // Forward elimination
  for (int i=0; i<n-2; i++) {
    // Eliminate b[i+1]
    auto cst = b[i+1] / c[i];
    c[i+1] -= d[i] * cst;
    d[i+1] -=  e[i] * cst;
    s1[i+1] -= s1[i] * cst;
    s2[i+1] -= s2[i] * cst;
    r.index({ i+1, slice }) = r.index({ i+1, slice }) - r.index({ i, slice }) * cst;
  
    // Eliminate a[i+2]
    cst = a[i+2] / c[i];
    b[i+2] -= d[i] * cst;
    c[i+2] -=  e[i] * cst;
    s1[i+2] -= s1[i] * cst;
    s2[i+2] -= s2[i] * cst;
    r.index({ i+2, slice }) = r.index({ i+2, slice }) - r.index({ i, slice }) * cst;
  }
         
    // Backward elimination
  for (int i=n-3; i>1; i--) { // or i>=1??
    // Eliminate d[i-1]
    auto cst = d[i-1] / c[i];
    r.index({ i-1, slice }) = r.index({ i-1, slice }) - r.index({ i, slice }) * cst;
    s1[i-1] -= s1[i] * cst;
    s2[i-1] -= s2[i] * cst;
    
    // Eliminate e[i-2]
    cst = e[i-2] / c[i];
    r.index({ i-2, slice }) = r.index({ i-2, slice }) - r.index({ i, slice }) * cst;
    s1[i-2] -= s1[i] * cst;
    s2[i-2] -= s2[i] * cst;
  }
 
  // Eliminate d[i-1]
  int ii = 1;
  auto cst = d[ii-1] / c[ii];
  r.index({ ii-1, slice }) = r.index({ ii-1, slice }) - r.index({ ii, slice }) * cst;
  s1[ii-1] -= s1[ii] * cst;
  s2[ii-1] -= s2[ii] * cst;
 
  // Eliminate oddball region
  cst = e[n-2] / c[0];
  r.index({ n-2, slice }) = r.index({ n-2, slice }) - r.index({ 0, slice }) * cst;
  s1[n-2] -= s1[0] * cst;
  s2[n-2] -= s2[0] * cst;
  
  cst = d[n-1] / c[0];
  r.index({ n-1, slice }) = r.index({ n-1, slice }) - r.index({ 0, slice }) * cst;
  s1[n-1] -= s1[0] * cst;
  s2[n-1] -= s2[0] * cst;
  
  cst = e[n-1] / c[1];
  r.index({ n-1, slice }) = r.index({ n-1, slice }) - r.index({ 1, slice }) * cst;
  s1[n-1] -= s1[1] * cst;
  s2[n-1] -= s2[1] * cst;
  
  // Eliminate corner region
  cst = s1[n-1] / s1[n-2];
  r.index({ n-1, slice }) = r.index({ n-1, slice }) - r.index({ n-2, slice }) * cst;
  s2[n-1] -= s2[n-2] * cst;
 
  // Back-substitution;
  r.index({ n-1, slice }) = r.index({ n-1, slice }) / s2[n-1];
  r.index({ n-2, slice }) = ( r.index({ n-2, slice }) - s2[n-2] * r.index({ n-1, slice }) ) / s1[n-2];
  for (int i=n-3; i>=0; i--) {
    r.index({ i, slice }) = ( r.index({ i, slice }) -
			      s1[i] * r.index({ n-2, slice }) -
			      s2[i] * r.index({ n-1, slice }) ) / c[i];
  }

  return;
}



void tridiagonal_parallel_step_1( torch::Tensor a,
				  torch::Tensor bc,
				  torch::Tensor c,
				  torch::Tensor s1,
				  torch::Tensor s2,
				  torch::Tensor r,
				  int64_t n,
				  int64_t lot ) {

  auto slice = torch::indexing::Slice(0,lot);
  
  // Forward elimination
  // Upper boundary is in s1[i]
  for (int i=1; i<n; i++) {
    auto cst = a[i] / bc[i-1];
    bc[i] -= c[i-1] * cst;
    r.index({ i, slice }) = r.index({ i, slice }) - r.index({ i-1, slice }) * cst;
    s1[i] = -s1[i-1] * cst;
  }

  // Backward elimination
  // Lower boundary is in s2[i]
  for (int i=n-2; i>=0; i--) {
    auto cst = c[i] / bc[i+1];
    r.index({ i, slice }) = r.index({ i, slice }) - r.index({ i+1, slice }) * cst;
    s1[i] -= s1[i+1] * cst;
    s2[i] = -s2[i+1] * cst;
  }
  
  return;
}


void tridiagonal_parallel_step_3( torch::Tensor bc,
				  torch::Tensor r1,
				  torch::Tensor r2,
				  torch::Tensor s1,
				  torch::Tensor s2,
				  torch::Tensor r,
				  int64_t n,
				  int64_t lot ) {

  auto slice = torch::indexing::Slice(0,lot);
  
  // Forward substitution
  for (int i=0; i<n; i++) {
    r.index({ i, slice }) = ( r.index({ i, slice }) -
			      s1[i] * r1 -
			      s2[i] * r2 ) / bc[i];
  }

  return;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("tridiagonal_serial_nonfactored", &tridiagonal_serial_nonfactored);
  m.def("tridiagonal_periodic_serial", &tridiagonal_periodic_serial);
  m.def("pentadiagonal_serial", &pentadiagonal_serial);
  m.def("pentadiagonal_periodic_serial", &pentadiagonal_periodic_serial);
  m.def("tridiagonal_parallel_step_1", &tridiagonal_parallel_step_1);
  m.def("tridiagonal_parallel_step_3", &tridiagonal_parallel_step_3);
}
