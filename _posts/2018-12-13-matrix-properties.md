This post summarizes the properties of commonly-used types of matrices.

- [General matrix ($A \in \Re^{m \times n}$)](#general-matrix-a-in-rem-times-n)
- [Square matrix ($A \in \Re^{n \times n}$)](#square-matrix-a-in-ren-times-n)
- [Symetric matrix / Hermitian matrix](#symetric-matrix--hermitian-matrix)
- [Positive semi-definite matrix (PSD)](#positive-semi-definite-matrix-psd)
- [Non-negative square matrix](#non-negative-square-matrix)
- [Other points](#other-points)

## General matrix ($A \in \Re^{m \times n}$)

**P0:** Basics

- $tr(AB) = tr(BA)$
- $rank(A) = rank(A^T)$
- $Range(A)^{\perp} = Null(A^T)$
- $dim(Range(A)) + dim(Null(A)) = n$

**P1:** SVD

$$ A = U \Sigma V^H = \sum_{i} \sigma_{i}^{r} u v^H = U_1 \tilde{\Sigma} V_1^H $$

where $U,V$ are unitary, and $\Sigma$ is triangular.

- $Range(A) = Range(U_1), Null(A) = Null(U_2)$
- $A^+ = V \begin{bmatrix} \tilde{\Sigma}^{-1} & \\ & 0\end{bmatrix} U^H = V_1 \tilde{\Sigma}^{-1} U_1^H$
- For $A x = y$, the solution is $x = A^+ y + \gamma, \gamma \in Null(A)=Range(U_2)$
- Projection: (Todo)

**P2:** Full column-rank matrix

- $A x = 0 \implies x = 0$

**P3:** Full row-rank matrix 
- $Range(A) = \Re^m$
- $Range(B) = Range(BA)$

## Square matrix ($A \in \Re^{n \times n}$)

**P0:** Basics

- $tr(A) = \sum_{i} \lambda_{i}$
- $det(A) = \prod_{i} \lambda_{i}$

**P1:** Schur Triangularization (always exists)

$$A = U T U^H$$
where $U$ is unitary matrix, and $T$ is upper triangular matrix.

**P2:** LU decompostion and its variations (+condtion: Every principal submatrix $A_{\{1,2,...n-1\}}$ nonsingular. If $A$ is nonsingular, ($L,U$) is unique.)

$$ A = LU = L D M^T $$
$L, M$ are lower triangular with unit diagnols, and $U$ is upper triangular, which is obtained by Gaussian-elimination.

**P3:** Eigenvalue and eigenvector

$$A V = V D$$
Eigenvalue decompostion (EVD) exists if $V$ is invertible.
- $\{\lambda(A)\} = \{\lambda(A^T)\}$ since $det(A - \lambda I) = det(A^T - \lambda I)$

**P4:** Matrix norm

- $|||A||| \geq \rho(A)$
 for any matrix norm.
- $\forall \epsilon > 0, \exists |||\cdot|||$ such that $\rho(A) \leq |||A||| \leq \rho(A) + \epsilon.$
- Cauthy inequality: 
$|||AB||| \leq |||A||| \cdot |||B|||$
- Examples: 
  - Operator norm (vector-induced norm)
    - $|||A|||_1 = \max_{||x||_1 = 1} ||Ax||_1$ 
    -- max column sum norm
    - $|||A|||_{\infty} = \max_{||x||_{\infty} = 1} ||Ax||_{\infty}$
     -- max row sum norm
    - $|||A|||_2 = \max_{||x||_2 = 1} ||Ax||_2 = \sigma_{max}(A)$
  - Schatten norm
    - $\sum_{i}^{p} \sigma_i(A)$ -- nuclear norm
    - $\sqrt{\sum_{i}^{p} \sigma_i(A)^2} = tr(A^H A) = \sqrt{\sum_i \sum_j a_{ij}^2}$
     -- Frobenius norm
    - $\sigma_{max}(A) = |||A|||_2$


**P5:** Spectral radius

- $\rho(A) = max\{|\lambda_i|\} = \inf_{|||\cdot|||} |||A||| = \lim_{k->\infty} (|||A^k|||)^{1/k}$
- $\lim_{k->\infty} A^k = 0 \iff \rho(A) < 1$.

## Symetric matrix / Hermitian matrix

**P0:** Basics
- All the eigenvalues are real
- Eigenvectors associated with different eigenvalues are orthogonal.

**P1:** EVD always exists

$$A = U \Lambda U^H$$
where $U$ is unitary, $D$ is diagonal with eigenvalues

## Positive semi-definite matrix (PSD)

**P1:** Square-root decomposition

$$ A = U \Lambda^{1/2} \Lambda^{1/2} U^H = B B^H, where \: B = U \Lambda^{1/2}.$$

**P2:** For $G=A A^T$

- $rank(A A^T) = rank(A) = rank(A^T) = rank(A^T A)$

## Non-negative square matrix

**P1:** For $A \geq 0$

- $A \leq B \implies \rho(A) \leq \rho(B)$
- $min\{A \: row/col \: sum\} \leq \rho(A) \leq max\{A \: row/col \: sum\}$
- $A$ has a positiv eigenvector $\implies$ the associated eigenvalue $\lambda_{i} = \rho(A)$

**P2:** For $A \gt 0$ (useful for PageRank)

- $\rho(A) = max\{|\lambda|\}$
 is an eigenvalue of $A$.
- The corresponding eigenvetor 
$v_{\rho}$ to $\rho(A)$ could be $v_{\rho} > 0$.
- $|\lambda| < \rho$ 
for $\lambda \neq \rho$.
- $dim(Null(A - \rho(A) I)) = 1$
(Geometric and algebra multiplicity of $\rho$ is 1.)



## Other points

**P1:** Kronecker product

- $(A \otimes B)^H = A^H \otimes B^H$
- $(A \otimes B)^{-1} = A^{-1} \otimes B^{-1}$
- $(A \otimes B) (C \otimes D) = (A C) \otimes (B D)$
- $vec(A X B) = (B^T \otimes A) \cdot vec(X)$
