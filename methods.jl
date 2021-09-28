using MAT
using LinearAlgebra
using SparseArrays

struct LinearL1Problem
    A::SparseMatrixCSC{Float64}
    b::Vector{Float64}
    λ::Float64
    ftarget::Float64
end

#incluir funções de teste

function readlasso(filename)
    vars = matread(filename)
    return LinearL1Problem(vars["A"], vec(vars["b"]), vars["lambda"], vars["ftarget"])
end

function readlogreg(filename)
    vars = matread(filename)
    return LinearL1Problem(vars["A"], vec(vars["b"]), vars["lambdalog"], vars["flogtarget"])
end

#Busca exata
function cauchystepsize(x, d, fval, ∇f, f, gradf)
    α = norm(d)^2
    α /= dot(d, data.A' * (data.A * d) + λ*d)
    return α
end
#Armijo

#Secção áurea

#Inexata com interpolação

#Método do gradiente descendente 
function grad_descent(x0, f, gradf, stepsize; ϵ=1.0e-5, ftarget=-1.0e20, max_iter=2000)
    x = copy(x0)
    fval, ∇f = gradf(x)
    histf = [fval]
    hist∇f = [norm(∇f, Inf)]
    iter = 0
    while hist∇f[end] > ϵ && fval > ftarget && iter < max_iter
        d = -∇f
        α = stepsize(x, d, fval, ∇f, f, gradf)
        @. x = x + α * d
        fval, ∇f = gradf(x)
        iter += 1
        append!(histf, fval)
        append!(hist∇f, norm(∇f, Inf))
    end
    return x, fval, ∇f, histf, hist∇f
end
