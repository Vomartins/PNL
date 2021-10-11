using MAT
using LinearAlgebra
using SparseArrays
using ForwardDiff
using IterTools

struct problems
    name::String
    f::Function
    ∇f::Function
    x₀::Vector{Float64}
    T::Float64
    max_iter::Int64
    minima::Float64
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
    name = "Teste 1"
    f(x) = 0.5*(x[1]-2)^2+(x[2]-1)^2
    
    ∇f(x) = [(x[1]-2),2*(x[2]-1)]
    
    x₀ = [2.0, 2.0]
    Ε = 1.0e-5
    max_iter = 2000
    minima = 0.0
    parameters = Dict()
    parameters["η"] = 0.25
    parameters["γ"] = 0.7
    parameters["ϵ"] = 10^(-3)
    parameters["ρ"] = 1
    parameters["β"] = 0.25
    parameters["α"] = 1    
    return problems(name, f, ∇f, x₀, Ε, max_iter, minima, parameters)
end

#Função #01
function Rosenbrock()
    name = "Rosenbrock"
    
    f(x) = (10*(x[2]-x[1]^2))^2 + (1-x[1])^2
    
    ∇f(x) = [-40*(x[2]-x[1]^2)*x[1]-2(1-x[1]),20*(x[2]-x[1]^2)]
    
    x₀ = [-1.2,1.0]
    T = 1.0e-5
    max_iter = 2000
    minima = 0.0 
    parameters = Dict()
    parameters["η"] = 0.25
    parameters["γ"] = 0.7
    parameters["ϵ"] = 10^(-3)
    parameters["ρ"] = 1
    parameters["β"] = 0.25
    parameters["α"] = 1    
    return problems(name, f, ∇f, x₀, T, max_iter, minima, parameters)
end

#função #09
function Gaussian()
    name = "Gaussian"
    
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
    minima = 1.12793*(10^(-8))
    parameters = Dict()
    parameters["η"] = 0.4
    parameters["γ"] = 0.8
    parameters["ϵ"] = 10^(-5)
    parameters["ρ"] = 1
    parameters["β"] = 0.4
    parameters["α"] = 1    
    return problems(name, f, ∇f, x₀, Ε, max_iter, minima, parameters)
end

#função #26
function trigonometric(n::Int64)
    name = "Trigonometric"
    
    m = n
    
    p(x,i) = (n - sum(j -> cos(x[j]), 1:m) + i*(1 - cos(x[i])) - sin(x[i]))
    f(x) = sum(i -> p(x,i)^2, 1:m)
    
    dq(x,i,k) = sin(x[k])+delta(i,k)*(i*sin(x[i])-cos(x[i]))
    dp(x,k) = 2*sum(i -> p(x,i)*dq(x,i,k), 1:n)
    ∇f(x) = [dp(x,k) for k in 1:n]
    
    x₀= (1/n)*ones(n)
    E = 1.0e-06
    max_iter = 3000
    minima = 0.0
    parameters = Dict()
    parameters["η"] = 0.4
    parameters["γ"] = 0.8
    parameters["ϵ"] = 10^(-5)
    parameters["ρ"] = 1
    parameters["β"] = 1.0e-06
    parameters["α"] = 1
    return problems(name, f, ∇f, x₀, E, max_iter, minima, parameters)
end

#função #29
function discrete_integral(n::Int64)
    name = "Discrete Integral"
    
    h = 1/(n+1)
    t(i) = h*i
    
    function p(x,i)
        if i < n
            return x[i] + (h/2)*((1-t(i))*sum(j -> t(j)*(x[j]+t(j)+1)^3, 1:i) +t(i)*sum(j -> (1-t(j))*(x[j]+t(j)+1)^3, (i+1):n))
        else
            return x[i] + (h/2)*((1-t(i))*sum(j -> t(j)*(x[j]+t(j)+1)^3, 1:i))
        end
    end
    
    f(x) = sum(i-> p(x,i)^2 ,1:n)
    
    function q(x,i,k)
        if k <= i
            return t(k)*(1-t(i))
        else
            return t(i)*(1-t(k))
        end
    end
        
    dq(x,i,k) = delta(i,k)+(3*h*(x[k]+t(k)+1)^2)*q(x,i,k)/2 
    dp(x,k) = 2*sum(i -> p(x,i)*dq(x,i,k), 1:n) 
    ∇f(x) = [dp(x,k) for k in 1:n]
        
    x₀= [t(i)*(t(i)-1) for i in 1:n]
    E = 1.0e-06
    max_iter = 3000
    minimizers = Dict()
    minima = 0.0
    parameters = Dict()
    parameters["η"] = 0.4
    parameters["γ"] = 0.8
    parameters["ϵ"] = 10^(-5)
    parameters["ρ"] = 1
    parameters["β"] = 1.0e-06
    parameters["α"] = 1
    return problems(name, f, ∇f, x₀, E, max_iter, minima, parameters)
end

#função #33
function linear_rank1(n::Int64, m::Int64)
    name = "Linear, rank 1"
    
    if m<n
        error("É necessário que m ≥ n")
        return nothing
    end
        
    p(x,i) = i*(sum(j->j*x[j], 1:n)) - 1
    f(x) = sum(i -> p(x,i)^2 ,1:m)
        
    dp(x,i) = sum(j-> 2*i*j*p(x,j), 1:m) 
    ∇f(x) = [dp(x,i) for i in 1:m]
        
    x₀= ones(m)
    E = 1.0e-06
    max_iter = 3000
    minima = (m*(m-1))/(2*(2*m+1))
    parameters = Dict()
    parameters["η"] = 0.4
    parameters["γ"] = 0.8
    parameters["ϵ"] = 10^(-5)
    parameters["ρ"] = 1
    parameters["β"] = 1.0e-06
    parameters["α"] = 1
    return problems(name, f, ∇f, x₀, E, max_iter, minima, parameters)
end

#Busca exata
function cauchystepsize(x, d, p::problems)
    α = norm(d)^2
    α /= dot(d, ForwardDiff.hessian(p.f,x)*d)
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
function secao_aurea(x,d,p::problems)
    ρ = p.parameters["ρ"]
    ϵ = p.parameters["ϵ"]
    ϕ(t) = p.f(x+t*d)
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
function armijo(x,d,p::problems)
    η = p.parameters["η"]
    γ = p.parameters["γ"]
    α = 1
    ϕ(t) = p.f(x+t*d)
    dϕ(t) = p.∇f(x+t*d) 
    while ϕ(α) > ϕ(0) + η*α*dot(dϕ(0),d)
        α = γ*α
    end
    return α
end

#Inexata com interpolacao
function interpolacao(x, d, p::problems)
    β = p.parameters["β"]
    α = p.parameters["α"]
    ϕ(t) = p.f(x + t*d)
    dϕ(t) = p.∇f(x+t*d)
    α₀ = α
    if ϕ(α) <= ϕ(0) + β*α*dot(dϕ(0),d)
        return α
    end
    α = -(dot(dϕ(0),d)*α^2)/(2*(ϕ(α) - ϕ(0) - (dot(dϕ(0),d))*α))
    if ϕ(α) <= ϕ(0) + β*α*dot(dϕ(0),d)
        return α
    end
    while (ϕ(α) > ϕ(0) + β*α*dot(dϕ(0),d))
        β = 1/((α₀^2)*(α^2)*(α-α₀))
        M = [α₀^2 -α^2; -α₀^3 α^3]
        v = [ϕ(α) - ϕ(0) - dot(dϕ(0),d)*α; ϕ(α₀) - ϕ(0) - dot(dϕ(0),d)*α₀]
        v = β*(M*v)
        a = v[1]
        b = v[2]
        raiz = real(sqrt(Complex(b^2 - 3*a*dot(dϕ(0),d))))
        α₀ = α
        α = (-b + raiz)/(3*a)
    end
    return α
end

#Método para avaliar as estratégias de stepsize
function grid_search(E, R, estrategia, p::problems)
    P = product(E,R) #função do IterTools
    Erro = Dict()
    if estrategia == armijo
        for ρ in P
            p.parameters["η"] = ρ[1]
            p.parameters["γ"] = ρ[2]
            solver = grad_descent(p,estrategia,ϵ=p.T,max_iter=p.max_iter)
            erro = abs((solver[2]-p.minima))
            if !isnan(erro)
                Erro[ρ] = (erro, solver[6])
            end
        end
        return Erro
    elseif estrategia == interpolacao
        for ρ in P
            p.parameters["β"] = ρ[1]
            p.parameters["α"] = ρ[2]
            solver = grad_descent(p,estrategia,ϵ=p.T,max_iter=p.max_iter)
            erro = abs((solver[2]-p.minima))
            if !isnan(erro)
                Erro[ρ] = (erro, solver[6])
            end
        end
        return Erro
    elseif estrategia == secao_aurea
        for ρ in P
            p.parameters["ϵ"] = ρ[1]
            p.parameters["ρ"] = ρ[2]
            solver = grad_descent(p,estrategia,ϵ=p.T,max_iter=p.max_iter)
            erro = abs((solver[2]-p.minima))
            if !isnan(erro)
                Erro[ρ] = (erro, solver[6])
            end
        end
        return Erro
    else
       return "Estratégia inválida!"
    end
end

#Encontra o valor mínimo dos valores do dicionário com tuplas de valores e retorna o minimo com a chave correspondente
function find_min_dict(d)
    
    K = collect(keys(d))
    if isempty(K)
        return (0.0,0.0),"Erro - dicionário vazio"
    else
        minval = d[K[1]][1]
        minkey = K[1]
        miniter = d[K[1]][2]

        for key in keys(d)
            if d[key][1] < minval
                minkey = key
                miniter = d[key][2]
                minval = d[key][1]
            end
        end

        return minkey, minval, miniter
    end
end

#Método do gradiente descendente 
function grad_descent(p::problems, stepsize; ϵ=1.0e-5, ftarget=-1.0e20, max_iter=2000)
    x = copy(p.x₀)
    fval = p.f(x)
    ∇f = p.∇f(x)
    histf = [fval]
    hist∇f = [norm(∇f, Inf)]
    iter = 0
    while hist∇f[end] > ϵ && fval > ftarget && iter < max_iter
        d = -p.∇f(x)
        α = stepsize(x,d,p)
        x = x + α * d
        fval = p.f(x)
        ∇f = p.∇f(x)
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
        if iter > 3
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