"""
    ConditionalEquations <: AbstractMeanfieldEquations

Type defining a system of stochastic differential equations, where `lhs` is a vector of
derivatives and `drift` and 'diffusion' are vectors of expressions. In addition, it keeps track
of the Hamiltonian, the collapse operators and the corresponding decay rates of
the system, as well as the measurement operators and the corresponding meaurement rates of the system..

# Fields
*`equations`: Vector of the differential equations of averages.
*`operator_equations`: Vector of the operator differential equations.
*`states`: Vector containing the averages on the left-hand-side of the equations.
*`operators`: Vector containing the operators on the left-hand-side of the equations.
*`hamiltonian`: Operator defining the system Hamiltonian.
*`jumps`: Vector of operators specifying the decay processes.
*`jumps`: Vector of operators specifying the adjoint of the decay processes.
*`rates`: Decay rates corresponding to the `jumps`.
*`iv`: The independent variable (time parameter) of the system.
*`varmap`: Vector of pairs that map the averages to time-dependent variables.
    That format is necessary for ModelingToolkit functionality.
*`order`: The order at which the [`cumulant_expansion`](@ref) has been performed.
*` mops: Vector of measurment operators.
*` mops_dagger: Vector of adjoint measurment operators.
*` mrates: Vector of measurement rates.
"""
struct ConditionalEquations <: AbstractMeanfieldEquations
    equations::Vector{Symbolics.Equation}
    operator_equations::Vector{Symbolics.Equation}
    states::Vector
    operators::Vector{QNumber}
    hamiltonian::QNumber
    jumps::Vector
    jumps_dagger
    rates::Vector
    iv::SymbolicUtils.Sym
    varmap::Vector{Pair}
    order::Union{Int,Vector{<:Int},Nothing}
    mops::Vector
    mops_dagger
    mrates::Vector
end



##### TODO: Modify to also call meanfield() to generate the equations sans the measurement terms.
"""
    conditional(ops::Vector,H::QNumber)
    conditional(op::QNumber,H::QNumber)
    conditional(ops::Vector,mops::Vector;
            mopsdagger::Vector=adjoint.(J),rates=ones(length(J)))
    conditional(op::QNumber,H::QNumber,mops::Vector;
            mops_dagger::Vector=adjoint.(J),rates=ones(length(J)))
Compute the set of equations for conditional backaction on the operators in `ops` under the 
measurement operators 'mops'. 
The resulting equation is equivalent to the stochastic part of the conditional 
Quantum-Langevin equation where noise is neglected.

# Arguments
*`ops::Vector`: The operators of which the equations are to be computed.
*`mops::Vector{<:QNumber}`: A vector containing the measurement operators of
    the system.
# Optional argumentes
*`mops_dagger::Vector=adjoint.(mops)`: Vector containing the hermitian conjugates of
    the measurement operators.
*`rates=ones(length(mops))`: Decay rates corresponding to the collapse operators in `mops`.
*`multithread=false`: Specify whether the derivation of equations for all operators in `ops`
    should be multithreaded using `Threads.@threads`.
*`simplify=true`: Specify whether the derived equations should be simplified.
*`order=nothing`: Specify to which `order` a [`cumulant_expansion`](@ref) is performed.
    If `nothing`, this step is skipped.
*`mix_choice=maximum`: If the provided `order` is a `Vector`, `mix_choice` determines
    which `order` to prefer on terms that act on multiple Hilbert spaces.
*`iv=SymbolicUtils.Sym{Real}(:t)`: The independent variable (time parameter) of the system.
"""
function conditional(ops::Vector, mops; mops_dagger::Vector=adjoint.(mops), rates=ones(Int,length(mops)),
                    multithread=false,
                    simplify=true,
                    order=nothing,
                    mix_choice=maximum,
                    iv=SymbolicUtils.Sym{Real}(:t))

    if rates isa Matrix
        mops = [mops]; mops_dagger = [mops_dagger]; rates = [rates]
    end
    mops_, mops_dagger_, rates_ = _expand_clusters(mops, mops_dagger, rates) # no idea what this does
    
    # Derive operator equations
    rhs = Vector{Any}(undef, length(ops))
    if multithread
        Threads.@threads for i=1:length(ops)
            rhs[i] = sum([mop*ops[i] - average(mop)*ops[i] for mop in mops])
        end
    else
        for i=1:length(ops)
            rhs[i] = sum([mop*ops[i] - average(mop)*ops[i] for mop in mops])
        end
    end

    # Average
    vs = map(average, ops)                              # contains the lhs of the equations
    rhs_avg = map(average, rhs)                         # rhs - operator products are mapped to averages
    if simplify
        rhs_avg = map(SymbolicUtils.simplify, rhs_avg)
    end
    rhs = map(undo_average, rhs_avg)                    # not sure why you need to undo the average

    if order !== nothing
        rhs_avg = [cumulant_expansion(r, order; simplify=simplify, mix_choice=mix_choice) for r∈rhs_avg]
    end

    eqs_avg = [Symbolics.Equation(l,r) for (l,r)=zip(vs,rhs_avg)]
    eqs = [Symbolics.Equation(l,r) for (l,r)=zip(ops,rhs)]
    varmap = make_varmap(vs, iv)

    ce = ConditionalEquations(eqs_avg, eqs, vs, ops, mops_, mops_dagger_, rates_, iv, varmap, order)
    return ce
end
#conditional(ops::QNumber, args...; kwargs...) = conditional([ops], args...; kwargs...)
#conditional(ops::Vector, mops; kwargs...) = conditional(ops, mops; mops_dagger=[], kwargs...)
