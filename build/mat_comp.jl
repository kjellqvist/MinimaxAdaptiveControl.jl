"""`dare(A, B, Q, R)`
Compute `X`, the solution to the discrete-time algebraic Riccati equation,
defined as A'XA - X - (A'XB)(B'XB + R)^-1(B'XA) + Q = 0, where Q>=0
Algorithm taken from:
Laub, "A Schur Method for Solving Algebraic Riccati Equations.
Relaxes the positive definiteness of R, useful for robust control.
"
http://dspace.mit.edu/bitstream/handle/1721.1/1301/R-0859-05666488.pdf
"""
function dare(A, B, Q, R)
    if !issemiposdef(Q)
        error("Q must be positive-semidefinite.");
    end
    n = size(A, 1);

    E = [
        Matrix{Float64}(I, n, n) B*(R\B');
        zeros(size(A)) A'
    ];
    F = [
        A zeros(size(A));
        -Q Matrix{Float64}(I, n, n)
    ];

    QZ = schur(F, E);
    QZ = ordschur(QZ, abs.(QZ.alpha./QZ.beta) .< 1);

    return QZ.Z[(n+1):end, 1:n]/QZ.Z[1:n, 1:n];
end

issemiposdef(A) = ishermitian(A) && minimum(real.(eigvals(A))) >= 0
issemiposdef(A::UniformScaling) = real(A.λ) >= 0
issquare(A) = size(A)[1] == size(A)[2]