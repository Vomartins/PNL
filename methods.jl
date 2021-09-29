using MAT
using LinearAlgebra
using SparseArrays
using ForwardDiff

#incluir funções de teste
function teste1()
    f(x) = 0.5*(x[1]-2)^2+(x[2]-1)^2
    ∇f(x) = [(x[1]-2),2*(x[2]-1)]
    return f, ∇f
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
    if ϕ(α) <= ϕ(0) + η*α*dot(dϕ(0),d)
        return α
    end
    α = -(dot(dϕ(0),d)*α^2)/(2*(ϕ(α) - ϕ(0) - (dot(dϕ(0),d))*α))
    if ϕ(α) <= ϕ(0) + η*α*dot(dϕ(0),d)
        return α
    end
    α₀ = α
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
        fval, ∇f = gradf(x)
        iter += 1
        append!(histf, fval)
        append!(hist∇f, norm(∇f, Inf))
    end
    return x, fval, ∇f, histf, hist∇f
end
