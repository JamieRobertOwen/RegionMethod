using JuMP, CPLEX, LinearAlgebra, NLopt

#include("relax.jl")

#testing
A = [3 2 5; -2 1 1; 1 -1 3; 5 2 -4; -1 0 0; 0 -1 0; 0 0 -1]

b = [55; 26; 30; 57; 0; 0; 0]

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

#this function is wrong
function transformationMatrixBoundsWRONG(A::AbstractMatrix)
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

#this function is wrong
function transformationMatrixBoundsWithCWRONG(A::AbstractMatrix, c::AbstractVector)
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

function transformationMatrixBoundsWithC(A::AbstractMatrix, c::AbstractVector)
    A= [A; c']
    rowsum = sum(A,dims=2)
    Areciprical = 1 ./ A
    umin= zeros(size(A,2))
    umax= zeros(size(A,2))

    for (index, col) in enumerate(eachcol(Areciprical.*rowsum))
        umin[index] = minimum(col[isfinite.(col)])
        umax[index] = maximum(col[isfinite.(col)])
    end
    return umin,umax
end




function orientationMIPOLD(A,b,c)
    n,m = size(A)
    orientationModel = Model(CPLEX.Optimizer)

    #orientationModel = Model()

    @variable(orientationModel, x[1:m] )

    @variable(orientationModel, u[1:m], Bin)

    @objective(orientationModel, Max, c' * x)

    @constraint(orientationModel, mainConstraint1, A * (x+u.-(1/2)) .<= b)

    @constraint(orientationModel, mainConstraint2, A * (x-u.+(1/2)) .<= b)

    #TT = stdout # save original STDOUT stream0
    #redirect_stdout()

    optimize!(orientationModel)

    #redirect_stdout(TT)

    return value.(x), (2*round.(Int,value.(u)) .-1)
end


function orientationMIP(A,b,c)
    n,m = size(A)
    orientationModel = Model(CPLEX.Optimizer)

    #orientationModel = Model()

    @variable(orientationModel, x[1:m] )

    @variable(orientationModel, u[1:m], Bin)

    fix(u[1], 1; force = true)

    v = u.-1/2

    @objective(orientationModel, Max, c' * x)

    @constraint(orientationModel, mainConstraint1,
        A * (x+v/2) .<= b-sum(abs.(A), dims=2)/4
    )

    @constraint(orientationModel, mainConstraint2,
        A * (x-v/2) .<= b-sum(abs.(A), dims=2)/4
    )

    #TT = stdout # save original STDOUT stream0
    #redirect_stdout()

    JuMP.optimize!(orientationModel)

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

    transmat = reduce(vcat,fill(transmatofdiag,m)') -
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


function finding_bounds(A,b,c,tol)
    A = rationalize.(float.(A))
    b = float.(b)
    relax!(A,b,tol)
    flotSol, orientation = orientationMIP(A,b,c)
    A = A.*orientation'
    c = c.*orientation
    uMin,uMax = transformationMatrixBoundsWithC(A,c)
    return uMin, uMax
end # functions

function finding_bounds_xor(n,tol)

    A = [ones(n)' ; -I]
    b = [1 ; zeros(n)]
    c = ones(n)
    A = rationalize.(float.(A))
    b = float.(b)
    relax!(A,b,tol)
    flotSol, orientation = orientationMIP(A,b,c)
    A = A.*orientation'
    c = c.*orientation
    uMin,uMax = transformationMatrixBoundsWithC(A,c)
    return A, b, c, uMin, uMax
end # functions

function housekeeping(A,b,c,tol)
    A = rationalize.(float.(A))
    b = float.(b)
    relax!(A,b,tol)
    flotSol, orientation = orientationMIP(A,b,c)
    A = A.*orientation'
    c = c.*orientation
    dMin,dMax = transformationMatrixBoundsWithC(A,c)
    return A, b, c, dMin, dMax
end


function qSolve(d,A,b,c)
    m,n = size(A)
    myMethodModel = Model(CPLEX.Optimizer)

    @variable(myMethodModel, x[1:n])

    @variable(myMethodModel, q[1:n] >=0)

    transmat = reduce(vcat,fill(d.*q,n)')+diagm(q)

    r = d .* (sign.(d).==1)
    s = -d .* (sign.(d).==-1)

    rSum = sum(r)
    sSum = sum(s)
    dSum = sum(d)

    @objective(myMethodModel, Max,
        #c' * (x + 1/2 .* transmat*sign.(c))
        c'*x
    )

    @constraint(myMethodModel, MainCon[i=1:m],
        A[i,:]' * (x + 1/2 .* transmat*sign.(A[i,:])) <= b[i]
    )

    @constraint(myMethodModel, rCon[i=1:n],
        1 + rSum -r[i]<=q[i]*(1+dSum)
    )

    @constraint(myMethodModel, sCon[i=1:n],
        sSum -s[i]<=q[i]*(1+dSum)
    )


    optimize!(myMethodModel)


    return objective_value(myMethodModel), value.(x), value.(q-
end

function NLsolveVersion(A,b,c,dMin)
    m,n = size(A)
    myMethodModel = Model(NLopt.Optimizer)

    set_optimizer_attribute(myMethodModel, "algorithm", :LD_MMA)

    @variable(myMethodModel, x[1:n])

    @variable(myMethodModel, q[1:n] >=0)

    @variable(myMethodModel, d[1:n] >=0)

    #@variable(myMethodModel, u[1:n])

    transmat = reduce(vcat,fill(d .* q,n)')+diagm(q)

    #r = d .* (sign.(d).==1)
    #s = -d .* (sign.(d).==-1)

    #rSum = sum(r)
    #sSum = sum(s)
    #dSum = sum(d)

    @objective(myMethodModel, Max,
        #c' * (x + 1/2 .* transmat*sign.(c))
        c'*x
    )

    @constraint(myMethodModel, MainCon[i=1:m],
        A[i,:]' * (x + 1/2 .* transmat*sign.(A[i,:])) <= b[i]
    )

    @constraint(myMethodModel, OrderCon[i=1:n],
        -d[i]<=(q[i]-1)*(1+sum(d))
    )

    @constraint(myMethodModel, dCon1,
        -1 .<= d .* dMin
    )

    #@constraint(myMethodModel, dDefineCon,
    #    d .* q .>= u
    #)

    JuMP.optimize!(myMethodModel)


    return objective_value(myMethodModel), value.(x), value.(q), value.(d)
end

function IntSolve(x,q,d,c)
    n=length(x)
    IntModel = Model(CPLEX.Optimizer)

    @variable(IntModel, gamma[1:n], Int)

    #@variable(IntModel, 0 <= s[1:n] <= 1)

    @objective(IntModel, Max, c' * gamma)

    transmat = reduce(vcat,fill(d .* q,n)')+diagm(q)

    #@constraint(IntModel, mainConstraint,
    #    gamma - x .<= transmat * (fill(1/2, n) - s)
    #)

    @constraint(IntModel, mainConstraint1,
        transmat^-1 *(gamma - x) .<= fill(1/2, n)
    )

    @constraint(IntModel, mainConstraint2,
        transmat^-1 *(gamma - x) .>= -fill(1/2, n)
    )

    JuMP.optimize!(IntModel)


    return objective_value(IntModel), value.(gamma)
end






function oldstuff

relax!(A,b,tol)

#this is wrong need to do orientation first as that can massively impact bounds on u
mini, maxi = transformationMatrixBounds([A; c']) #include c in orientation constriants

flotSol, orientation = orientationMIP(A,b,c)

~,~,uactual,vactual =myMethodUV(A,b,c,mini,orientation)

n,m = size(A)
actualT = reduce(vcat,fill(uactual,m)') -diagm(uactual) + diagm(vactual)
end
