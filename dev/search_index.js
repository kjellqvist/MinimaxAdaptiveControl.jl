var documenterSearchIndex = {"docs":
[{"location":"","page":"-","title":"-","text":"","category":"page"},{"location":"","page":"-","title":"-","text":"noteId: \"2828347064af11eb8191153eb319e31b\" tags: []","category":"page"},{"location":"","page":"-","title":"-","text":"","category":"page"},{"location":"","page":"-","title":"-","text":"CurrentModule = MinimaxAdaptiveControl","category":"page"},{"location":"","page":"-","title":"-","text":"Modules = [MinimaxAdaptiveControl]\nOrder   = [:function, :type]","category":"page"},{"location":"#MinimaxAdaptiveControl.K-Tuple{MAController}","page":"-","title":"MinimaxAdaptiveControl.K","text":"K(mac::MAController)\n\nSelect the feedback gain such that u = -Kx\n\n\n\n\n\n","category":"method"},{"location":"#MinimaxAdaptiveControl.Tsyn-Union{Tuple{M}, Tuple{MAController{M},JuMP.Model}, Tuple{MAController{M},JuMP.Model,Real}} where M<:Number","page":"-","title":"MinimaxAdaptiveControl.Tsyn","text":"T(mac, model, tol = 0.01)\n\nsynthesize a common T using convex programming such that inequalities (19) and (20) are fulfilled. ...\n\nArguments:\n\nmac::MAController MinimaxAdaptiveControl controller object\nmodel::JuMP.Model A user supplied JuMP model. Currently the solvers Mosek and Hypatia works.\ntol::Real = 0.01  Forcing (19) and (20) to hold with margin, i.e. T - tol*I >= ...\n\n...\n\n\n\n\n\n","category":"method"},{"location":"#MinimaxAdaptiveControl.Vbar-Tuple{MAController,AbstractArray{T,2} where T,AbstractArray}","page":"-","title":"MinimaxAdaptiveControl.Vbar","text":"Vbar(mac, T, x)\n\nCompute the current upper bound of the value function.\n\n...\n\nArguments\n\nmac::MAController: Controller object\nT::AbstractMatrix: T - matrix, can be synthesized using Tval(...)\nx::AbstractArray: Next state\n\n...\n\n\n\n\n\n","category":"method"},{"location":"#MinimaxAdaptiveControl.X-Union{Tuple{P}, Tuple{MAController{P},AbstractArray{T,2} where T,Integer,Integer}, Tuple{MAController{P},AbstractArray{T,2} where T,Integer,Integer,Real}} where P<:Number","page":"-","title":"MinimaxAdaptiveControl.X","text":"X(mac,T,i, k, tol) where P<:Number\n\nSynthesize a symmetric matrix X such that inequality (19) is satisfied iff X is positive semidefinite. ...\n\nArguments:\n\nmac::MAController   MinimaxAdaptiveControl controller object\nT::AbstractMatrix   Either a matrix or a JuMP variable\ni::Integer          Index variable,\nk::Integer          Index Variable\ntol::Real = 0.01    Forcing (19) to hold with margin, i.e. T - tol*I >= ...\n\n...\n\n\n\n\n\n","category":"method"},{"location":"#MinimaxAdaptiveControl.Z-Union{Tuple{P}, Tuple{MAController{P},AbstractArray{T,2} where T,Integer,Integer,Integer}, Tuple{MAController{P},AbstractArray{T,2} where T,Integer,Integer,Integer,Real}} where P<:Number","page":"-","title":"MinimaxAdaptiveControl.Z","text":"Z(mac, T, i, j, k)\n\nSynthesize a symmetric matrix Z such that inequality (20) is satisfied iff Z is positive semidefinite.\n\n...\n\nArguments:\n\nmac::MAController   MinimaxAdaptiveControl controller object\nT::AbstractMatrix   Either a matrix or a JuMP variable\ni::Integer          Index variable\nj::Integer          Index variable\nk::Integer          Index Variable\ntol::Real = 0.01    Forcing (20) to hold with margin, i.e. T - tol*I >= ...\n\n...\n\n\n\n\n\n","category":"method"},{"location":"#MinimaxAdaptiveControl.dare-NTuple{4,Any}","page":"-","title":"MinimaxAdaptiveControl.dare","text":"dare(A, B, Q, R)\n\nCompute X, the solution to the discrete-time algebraic Riccati equation, defined as A'XA - X - (A'XB)(B'XB + R)^-1(B'XA) + Q = 0, where Q>=0.\n\nThis version relaxes the requirement that R>0, rather R = [R1 0;0 R2] where R1 >0, R2 < 0. This is a usuful formulation for H_∞ synthesis.\n\nAlgorithm taken from: Laub, \"A Schur Method for Solving Algebraic Riccati Equations.\" Relaxes the positive definiteness of R, useful for robust control.\n\nhttp://dspace.mit.edu/bitstream/handle/1721.1/1301/R-0859-05666488.pdf\n\nImplementation stolen from: ControlSystems.jl\n\n\n\n\n\n","category":"method"},{"location":"#MinimaxAdaptiveControl.update!-Union{Tuple{T}, Tuple{MAController,AbstractArray{T,1},AbstractArray{T,1}}} where T<:Number","page":"-","title":"MinimaxAdaptiveControl.update!","text":"update!(mac,x, u)\n\nUpdate the internal states of the controller based  on current state x and control signal u. ...\n\nArguments:\n\nmac::MAController       MinimaxAdaptiveControl controller object\nx::AbstractArray{T,1}   The current state\nu::Abstractarray{T,1}   The previous control signal\n\n...\n\n\n\n\n\n","category":"method"},{"location":"#MinimaxAdaptiveControl.Candidate","page":"-","title":"MinimaxAdaptiveControl.Candidate","text":"Candidate{T<:Number}\n\nFields:\n\nA::AbstractMatrix{T} System matrix\n\nB::AbstractMatrix{T} Input Gain matrix\n\nK::AbstractMatrix{T} H_infty` feedback gain\n\nP::AbstractMatrix{T} Stationary solution to the Riccati equation\n\nhist::Base.RefValue{<:Real} History, sum _k=0^N lambda^N-kx_k+1 - Ax_k -Bu_k^2\n\nlam::T Forgetting factor lambda\n\n\n\n\n\n","category":"type"},{"location":"#MinimaxAdaptiveControl.MAController-Union{Tuple{T}, Tuple{AbstractArray{var\"#s21\",1} where var\"#s21\"<:AbstractArray{T,2},AbstractArray{var\"#s22\",1} where var\"#s22\"<:AbstractArray{T,2},AbstractArray{T,2} where T,AbstractArray{T,2} where T,Real,AbstractArray{var\"#s23\",1} where var\"#s23\"<:Number}, Tuple{AbstractArray{var\"#s24\",1} where var\"#s24\"<:AbstractArray{T,2},AbstractArray{var\"#s25\",1} where var\"#s25\"<:AbstractArray{T,2},AbstractArray{T,2} where T,AbstractArray{T,2} where T,Real,AbstractArray{var\"#s26\",1} where var\"#s26\"<:Number,Any}} where T<:Number","page":"-","title":"MinimaxAdaptiveControl.MAController","text":"mac = MAController(As, Bs, Q, R, γ, x0, lam)\n\nCreate a MAController (minimax adaptive controller) mac::MAController{T} with matrices containing elements of type T, solving \\min_u\\max_w,i\\sum (|xt|^2\\Q + |ut|^2\\R) - \\gamma^2 \\sum |w_t|^2 ...\n\nArguments\n\nAs::AbstractVector{<:AbstractMatrix{T}} A vector of state matrices\nBs::AbstractVector{<:AbstractMatrix{T}} A vector of input gains\nQ::AbstractMatrix,                      State penalty matrix |x|_Q^2\nR::AbstractMatrix,                      Control penalty matrix |u|_R^2\nγ::Real,                                Disturbance penalty, \\H_\\infty gain\nx0::AbstractVector{<:Number}            Initial state\nlam::Real=1                             Forgetting factor\n\n...\n\n\n\n\n\n","category":"method"}]
}
