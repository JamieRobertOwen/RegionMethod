using JuMP, CPLEX, LinearAlgebra

#include("relax.jl")

#testing
A = [3 2 5; 2 1 1; 1 1 3; 5 2 4; -1 -1 -1]

b = [55; 26; 30; 57 ; 0]

c = [20, 10, 15]

tol = 0.1

A = rationalize.(float.(A))

b = float.(b)




function relax!(A::AbstractMatrix{<:Rational{Int64}},b::AbstractVector{<:AbstractFloat},tol::AbstractFloat)
    #takes a IP of form Ax<=b and relaxes b using rao cuts

    #rationalize.(float.(A)) to convert A from Int64 to rational
    size(A,1) == size(b,1) || throw(DimensionMismatch("A and b must have the same number of rows"))

    for (index, row) in enumerate(eachrow(A))
        rowgcd = gcd(row)
        if rowgcd != 0
            A[index,:] = A[index,:]/rowgcd
            b[index] = fld(b[index],rowgcd)+1.0-tol
        end
    end
end

function transformationMatrixBounds(A::AbstractMatrix)
    #need to check that there is no empty columns in A
    rowsum = sum(A,dims=2)
    Areciprical = 1 ./ A
    minoffdiagcoef= -ones(size(A,2))
    maxoffdiagcoef= -ones(size(A,2))

    for (index, col) in enumerate(eachcol(Areciprical.*rowsum))
        minoffdiagcoef[index] += minimum(col[isfinite.(col)])
        maxoffdiagcoef[index] += maximum(col[isfinite.(col)])
    end
    return minoffdiagcoef,maxoffdiagcoef
end

function transformationMatrixBoundsWithC(A::AbstractMatrix, c::AbstractVector)
    A= [A; c']
    rowsum = sum(A,dims=2)
    Areciprical = 1 ./ A
    minoffdiagcoef= -ones(size(A,2))
    maxoffdiagcoef= -ones(size(A,2))

    for (index, col) in enumerate(eachcol(Areciprical.*rowsum))
        minoffdiagcoef[index] += minimum(col[isfinite.(col)])
        maxoffdiagcoef[index] += maximum(col[isfinite.(col)])
    end
    return minoffdiagcoef,maxoffdiagcoef
end




function orientationMIP(A,b,c)
    n,m = size(A)
    orientationModel = Model(CPLEX.Optimizer)

    #orientationModel = Model()

    @variable(orientationModel, x[1:m] )

    @variable(orientationModel, u[1:m], Bin)

    @objective(orientationModel, Max, c' * x)

    @constraint(orientationModel, mainConstraint1, A * (x+u.-(1/2)) .<= b)

    @constraint(orientationModel, mainConstraint2, A * (x-u.+(1/2)) .<= b)

    #TT = stdout # save original STDOUT stream
    #redirect_stdout()

    optimize!(orientationModel)

    #redirect_stdout(TT)

    return value.(x), (2*round.(Int,value.(u)) .-1)
end


#u=[1,1,1]
function myMethodUV(A,b,c,mini,orientation)
    n,m = size(A)

    #re-orient the problem
    A = A.*orientation'
    c = c.*orientation
    #remeber to reorder for orientation

    #signCombs = unique(sign.(A), dims =2)
    #NosignCombs = size(signCombs,1)
    myMethodModel = Model(CPLEX.Optimizer)

    @variable(myMethodModel, x[1:m])

    @variable(myMethodModel, u[1:m] >=0)

    @variable(myMethodModel, v[1:m] >=0)

    transmatdiag = v
    transmatofdiag = u

    #transmat = reduce(hcat,fill(u,m))+diagm(v)

    transmat = reduce(hcat,fill(transmatofdiag,m)) -
    diagm(transmatofdiag) + diagm(transmatdiag)

    #@objective(myMethodModel, Max,
    #    c'*(x-(1/2)*(sign.(c).*u.+sum(y.*sign.(c))))
    #)

    @objective(myMethodModel, Max,
        c' * (x + 1/2 .* transmat*sign.(c))
    )

    #@constraint(myMethodModel, MainCon[i=1:n],
    #    A[i,:]'*(x+(1/2)*(sign.(A[i,:]).*u.+sum(u.*v.*sign.(A[i,:])))) <= b[i]
    #)

    #@constraint(myMethodModel, MainCon[i=1:n],
    #    A[i,:]'*(x+(1/2)*(sign.(A[i,:]).*u.+sum(y.*sign.(A[i,:])))) <= b[i]
    #)

    @constraint(myMethodModel, MainCon[i=1:n],
        A[i,:]' * (x + 1/2 .* transmat*sign.(A[i,:])) <= b[i]
    )

    @constraint(myMethodModel, validity,
        transmat * fill(1,m) .>= fill(1,m)
    )

    @constraint(myMethodModel, orientationMin,
        0 .<= transmatdiag + transmatofdiag.*mini
    )

    #shouldn't need if offdiag is non-negative
    #@constraint(myMethodModel, orientationMax,
    #    0 .<= transmatdiag + transmatofdiag.*maxi
    #)

    #TT = stdout # save original STDOUT stream
    #redirect_stdout()
    #set_optimizer_attribute(myMethodModel, "CPX_PARAM_EPRHS", 1e-4)

    optimize!(myMethodModel)

    #println(termination_status(myMethodModel))
    #redirect_stdout(TT)

    return objective_value(myMethodModel), value.(x) .*orientation, value.(u), value.(v)

    #if termination_status(myMethodModel) == MOI.OPTIMAL
    #    return objective_value(myMethodModel), value.(x), value.(v)
    #else
    #    return NaN, ones(m)*NaN, ones(m)*NaN
    #end
end




relax!(A,b,tol)

mini, maxi = transformationMatrixBounds([A; c']) #include c in orientation constriants

flotSol, orientation = orientationMIP(A,b,c)

~,~,uactual,vactual =myMethodUV(A,b,c,mini,orientation)

actualT = reduce(hcat,fill(uactual,m)) -diagm(uactual) + diagm(vactual)
