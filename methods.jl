using MAT
using LinearAlgebra
using SparseArrays
using ForwardDiff

struct problems
    f::Function
    ∇f::Function
    x₀::Vector{Float64}
    Ε::Float64
    max_iter::Int64
    minimizers::Dict{Float64,Vector}
    parameters::Dict{String,Float64}
end

function delta(i,k)
    if i == k
        return 1
    else
        return 0
    end
end

#incluir funções de teste
function teste1()
    f(x) = 0.5*(x[1]-2)^2+(x[2]-1)^2
    ∇f(x) = [(x[1]-2),2*(x[2]-1)]
    x₀ = [2.0, 2.0]
    Ε = 1.0e-5
    max_iter = 2000
    minimizers = Dict()
    minimizers[0.0] = [2.0, 1.0]
    parameters = Dict()
    return problems(f, ∇f, x₀, Ε, max_iter, minimizers, parameters)
end

#Função #01
function Rosenbrock()
    f(x) = (10*(x[2]-x[1]^2))^2 + (1-x[1])^2
    ∇f(x) = [-40*(x[2]-x[1]^2)*x[1]-2(1-x[1]),20*(x[2]-x[1]^2)]
    x₀ = [-1.2,1.0]
    Ε = 1.0e-5
    max_iter = 2000
    minimizers = Dict()
    minimizers[0.0] = [1.0, 1.0] 
    parameters = Dict()
    parameters["η"] = 0.25
    parameters["γ"] = 0.7
    parameters["ϵ"] = 10^(-3)
    parameters["ρ"] = 1
    parameters["β"] = 0.25
    parameters["α"] = 1    
    return problems(f, ∇f, x₀, Ε, max_iter, minimizers, parameters)
end

#função #09
function Gaussian()
    y = [0.0009, 0.0044, 0.0175, 0.0540, 0.1295, 0.2420, 0.3521, 0.3989, 0.3521, 0.2420, 0.1295, 0.0540, 0.0175, 0.0044, 0.0009]
    t(i) = (8-i)/2
    p(x,i) = (x[1]*exp((-x[2]*(t(i)-x[3])^2)/2)-y[i])^2
    f(x) = sum(i -> p(x,i),1:15)
    dp₁(x,i) = 2*exp((-x[2]*(t(i)-x[3])^2)/2)*p(x,i)
    dp₂(x,i) = (-x[1]*(t(i)-x[3])^2)*exp((-x[2]*(t(i)-x[3])^2)/2)*p(x,i)
    dp₃(x,i) = (2*x[1]*x[2]*(t(i)-x[3]))*exp((-x[2]*(t(i)-x[3])^2)/2)*p(x,i)
    ∇f(x) = [sum(i -> dp₁(x,i),1:15), sum(i -> dp₂(x,i),1:15), sum(i -> dp₃(x,i),1:15)]
    x₀ = [0.4, 1.0, 0.0]
    Ε = 1.0e-10
    max_iter = 10000
    minimizers = Dict()
    minimizers[1.12793*(10^(-8))] = [nothing]
    parameters = Dict()
    parameters["η"] = 0.4
    parameters["γ"] = 0.8
    parameters["ϵ"] = 10^(-5)
    parameters["ρ"] = 1
    parameters["β"] = 0.4
    parameters["α"] = 1    
    return problems(f, ∇f, x₀, Ε, max_iter, minimizers, parameters)
end

#função #26
function trigonometric(n::Int64)
    m = n
    p(x,i) = (n - sum(j -> cos(x[j]), 1:m) + i*(1 - cos(x[i])) - sin(x[i]))
    f(x) = sum(i -> p(x,i)^2, 1:m)
    dq(x,i,k) = sin(x[k])+delta(i,k)*(i*sin(x[i])-cos(x[i]))
    dp(x,k) = 2*sum(i -> p(x,i)*dq(x,i,k), 1:n)
    ∇f(x) = [dp(x,k) for k in 1:n]
    x₀= (1/n)*ones(n)
    E = 1.0e-06
    max_iter = 3000
    minimizers = Dict()
    minimizers[0.0] = [nothing]
    parameters = Dict()
    parameters["η"] = 0.4
    parameters["γ"] = 0.8
    parameters["ϵ"] = 10^(-5)
    parameters["ρ"] = 1
    parameters["β"] = 1.0e-06
    parameters["α"] = 1
    return problems(f, ∇f, x₀, E, max_iter, minimizers, parameters)
end

#Busca exata
function cauchystepsize(x, d, f, ∇f)
    α = norm(d)^2
    α /= dot(d, ForwardDiff.hessian(f,x)*d)
    return α
end

#Calcula o intervalo unimodal de ϕ
function unimodal(ρ,ϕ)
    a = 0
    s = ρ
    b = 2*ρ
    while ϕ(b) < ϕ(s)
        a = s
        s = b
        b *= 2
    end
    return a,b    
end

#Secção áurea
function secao_aurea(x,d,f,∇f,ϵ,ρ)
    ϕ(t) = f(x+t*d)
    a,b = unimodal(ρ,ϕ)
    φ = MathConstants.golden
    θ = 1/φ
    v = a + θ*(b-a)
    u = a + (1-θ)*(b-a)
    while b-a > ϵ
        if ϕ(u) < ϕ(v)
            b = v
            v = u
            u = a + (1-θ)*(b-a)
        else
            a = u
            u = v
            v = a + θ*(b-a)
        end
    end
    return (u+v)/2
end

#Armijo
function armijo(x,d,f,∇f,η,γ)
    α = 1
    ϕ(t) = f(x+t*d)
    dϕ(t) = ∇f(x+t*d)
    while ϕ(α) > ϕ(0) + η*α*dot(dϕ(0),d)
        α = γ*α
    end
    return α
end

#Inexata com interpolacao
function interpolacao(x, d, f, ∇f, η, α)
    ϕ(t) = f(x + t*d)
    dϕ(t) = ∇f(x+t*d)
    α₀ = α
    if ϕ(α) <= ϕ(0) + η*α*dot(dϕ(0),d)
        return α
    end
    α = -(dot(dϕ(0),d)*α^2)/(2*(ϕ(α) - ϕ(0) - (dot(dϕ(0),d))*α))
    if ϕ(α) <= ϕ(0) + η*α*dot(dϕ(0),d)
        return α
    end
    while (ϕ(α) > ϕ(0) + η*α*dot(dϕ(0),d))
        β = 1/((α₀^2)*(α^2)*(α-α₀))
        M = [α₀^2 -α^2; -α₀^3 α^3]
        v = [ϕ(α) - ϕ(0) - dot(dϕ(0),d)*α; ϕ(α₀) - ϕ(0) - dot(dϕ(0),d)*α₀]
        v = β*(M*v)
        a = v[1]
        b = v[2]
        raiz = sqrt(b^2 - 3*a*dot(dϕ(0),d))
        α₀ = α
        α = (-b + raiz)/(3*a)
    end
    return α
end

#Método do gradiente descendente 
function grad_descent(x0, f, gradf, stepsize; ϵ=1.0e-5, ftarget=-1.0e20, max_iter=2000)
    x = copy(x0)
    fval = f(x)
    ∇f = gradf(x)
    histf = [fval]
    hist∇f = [norm(∇f, Inf)]
    iter = 0
    while hist∇f[end] > ϵ && fval > ftarget && iter < max_iter
        d = -gradf(x)
        α = stepsize(x, d, f, gradf)
        x = x + α * d
        fval = f(x)
        ∇f = gradf(x)
        iter += 1
        append!(histf, fval)
        append!(hist∇f, norm(∇f, Inf))
    end
    return x, fval, ∇f, histf, hist∇f, iter
end

#Método do gradiente descendente com ruído
function noise_grad(x0, f, gradf, stepsize; ϵ=1.0e-5, ftarget=-1.0e20, max_iter=2000)
    x = copy(x0)
    fval = f(x)
    ∇f = gradf(x)
    histf = [fval]
    hist∇f = [norm(∇f, Inf)]
    iter = 0
    while hist∇f[end] > ϵ && fval > ftarget && iter < max_iter
        d = -gradf(x)
        α = stepsize(x, d, f, gradf)
        if iter >3
            if abs(fval - histf[end-1]) < ϵ
                ξ = -1 .+ (2*rand(length(x)))
                ξ = norm(d,2)*ξ/norm(ξ,2)
                x = x + α*(d+ξ)
            else
                x = x + α * d
            end 
        else
            x = x + α * d
        end
        fval = f(x)
        ∇f = gradf(x)
        iter += 1
        append!(histf, fval)
        append!(hist∇f, norm(∇f, Inf))
    end
    return x, fval, ∇f, histf, hist∇f, iter
end