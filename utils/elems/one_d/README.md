Sub-package elem.one_d
======================

1. p-type Jacobi expansion
2. p-type Legendre expansion
3. p-type commonly-used Jacobi expansion (alpha=beta=1)
4. Lagrange expansion
5. Gauss-Lobatto-Jacobi Lagrange expansion
6. Moment expansion
7. pure Legendre expansion

## Note:

p-type Legendre expansion and commonly-used Jacobi expansion are special cases
of p-type Jacobi expansion. When calculating the mass matrix and weak-form 
Laplacian, they will use analytical solutions instead of numerical integration.
This can gaurantee zero-entities are always zero and can hence lower down 
rounding errors in subsequent calculations.
