"""
    FockSpace <: HilbertSpace

[`HilbertSpace`](@ref) defining a Fock space for bosonic operators.
See also: [`Destroy`](@ref), [`Create`](@ref)
"""
struct FockSpace{S} <: HilbertSpace
    name::S
    function FockSpace{S}(name::S) where {S}
        r = SymbolicUtils.@rule(*(~~x::has_consecutive(isdestroy,iscreate)) => commute_bosonic(*, ~~x))
        (r ∈ COMMUTATOR_RULES.rules) || push!(COMMUTATOR_RULES.rules, r)
        new(name)
    end
end
FockSpace(name::S) where {S} = FockSpace{S}(name)
Base.:(==)(h1::T,h2::T) where T<:FockSpace = (h1.name==h2.name && h1.name==h2.name)

"""
    Destroy <: BasicOperator

Bosonic operator on a [`FockSpace`](@ref) representing the quantum harmonic
oscillator annihilation operator.
"""
struct Destroy{H<:HilbertSpace,S,A,IND} <: BasicOperator
    hilbert::H
    name::S
    aon::A
    index::IND
    function Destroy{H,S,A,IND}(hilbert::H,name::S,aon::A,index::IND) where {H,S,A,IND}
        @assert has_hilbert(FockSpace,hilbert,aon)
        new(hilbert,name,aon,index)
    end
end
isdestroy(a) = false
isdestroy(a::SymbolicUtils.Term{T}) where {T<:Destroy} = true

"""
    Create <: BasicOperator

Bosonic operator on a [`FockSpace`](@ref) representing the quantum harmonic
oscillator creation operator.
"""
struct Create{H<:HilbertSpace,S,A,IND} <: BasicOperator
    hilbert::H
    name::S
    aon::A
    index::IND
    function Create{H,S,A,IND}(hilbert::H,name::S,aon::A,index::IND) where {H,S,A,IND}
        @assert has_hilbert(FockSpace,hilbert,aon)
        new(hilbert,name,aon,index)
    end
end
iscreate(a) = false
iscreate(a::SymbolicUtils.Term{T}) where {T<:Create} = true

for f in [:Destroy,:Create]
    @eval $(f)(hilbert::H,name::S,aon::A,index::IND=default_index()) where {H,S,A,IND} = $(f){H,S,A,IND}(hilbert,name,aon,index)
    @eval $(f)(hilbert::FockSpace,name) = $(f)(hilbert,name,1)
    @eval function $(f)(hilbert::ProductSpace,name)
        i = findall(x->isa(x,FockSpace),hilbert.spaces)
        if length(i)==1
            return $(f)(hilbert,name,i[1])
        else
            isempty(i) && error("Can only create $($(f)) on FockSpace! Not included in $(hilbert)")
            length(i)>1 && error("More than one FockSpace in $(hilbert)! Specify on which Hilbert space $($(f)) should be created with $($(f))(hilbert,name,i)!")
        end
    end
    @eval function embed(h::ProductSpace,op::T,aon::Int) where T<:($(f))
        check_hilbert(h.spaces[aon],op.hilbert)
        op_ = $(f)(h,op.name,aon)
        return op_
    end
    @eval function _to_symbolic(op::T) where T<:($(f))
        sym = SymbolicUtils.term($(f), op.hilbert, op.name, acts_on(op), get_index(op); type=$(f))
        return sym
    end
    @eval function Base.hash(op::T, h::UInt) where T<:($(f))
        hash(op.hilbert, hash(op.name, hash(op.aon, hash(op.index, h))))
    end
end

Base.adjoint(op::Destroy) = Create(op.hilbert,op.name,acts_on(op),get_index(op))
Base.adjoint(op::Create) = Destroy(op.hilbert,op.name,acts_on(op),get_index(op))

# Commutation relation in simplification
function commute_bosonic(f,args)
    commuted_args = []
    i = 1
    while i <= length(args)
        if isdestroy(args[i]) && i<length(args) && iscreate(args[i+1]) && (acts_on_index(args[i])==acts_on_index(args[i+1]))
            push!(commuted_args, args[i+1]*args[i] + 1)
            i += 2
        else
            push!(commuted_args, args[i])
            i += 1
        end
    end
    return f(commuted_args...)
end
