# Relevant parts of SDESystem interface
MTK.get_iv(ce::AbstractEquations) = ce.iv
MTK.states(ce::AbstractEquations) = ce.states
MTK.get_iv(me::AbstractMeanfieldEquations) = me.iv
MTK.states(me::AbstractMeanfieldEquations) = me.states

function MTK.equations(ce::AbstractEquations)
    # Get the MTK variables
    varmap = ce.varmap
    vs = MTK.states(ce)
    vhash = map(hash, vs)

    # Substitute conjugate variables by explicit conj
    vs′ = map(_conj, vs)
    vs′hash = map(hash, vs′)
    i = 1
    while i <= length(vs′)
        if vs′hash[i] ∈ vhash
            deleteat!(vs′, i)
            deleteat!(vs′hash, i)
        else
            i += 1
        end
    end
    rhs = [substitute_conj(eq.rhs, vs′, vs′hash) for eq ∈ ce.equations]

    # Substitute to MTK variables on rhs
    subs = Dict(varmap)
    rhs = [substitute(r, subs) for r∈rhs]
    vs_mtk = getindex.(varmap, 2)

    # Return equations
    t = MTK.get_iv(ce)
    D = MTK.Differential(t)
    return [Symbolics.Equation(D(vs_mtk[i]), rhs[i]) for i=1:length(vs)]
end

# Substitute conjugate variables
function substitute_conj(t, vs′, vs′hash)
    if SymbolicUtils.istree(t)
        if t isa Average
            if hash(t) ∈ vs′hash
                t′ = _conj(t)
                return conj(t′)
            else
                return t
            end
        else
            _f = x->substitute_conj(x,vs′,vs′hash)
            args = map(_f, SymbolicUtils.arguments(t))
            return SymbolicUtils.similarterm(t, SymbolicUtils.operation(t), args)
        end
    else
        return t
    end
end

function MTK.SDESystem(me::QuantumCumulantsConditional.AbstractMeanfieldEquations, ce::QuantumCumulantsConditional.AbstractEquations, ps; kwargs...)
    eqs = MTK.equations(me)
    #noiseeqs = MTK.equations(ce)
    noiseeqs = [Symbolics.Num(eq.rhs) for eq in MTK.equations(ce)]
    vars = [Symbolics.Num(eq.lhs) for eq in noiseeqs.equations]
    ps = [Symbolics.Symbol(p) for p in ps]
    iv = Symbolics.Num(ce.iv)
    return MTK.SDESystem(eqs, noiseeqs, iv, vars, ps; kwargs...)
end
