using JuMP, CPLEX, LinearAlgebra, NLopt


#test cases
include("test_cases.jl")

#testing
A = [3 2 5; -2 1 1; 1 -1 3; 5 2 -4; -1 0 0; 0 -1 0; 0 0 -1]

b = [55; 26; 30; 57; 0; 0; 0]

c = [20, 10, 15]

tol = 0.1

A = rationalize.(float.(A))

b = float.(b)




function relax(A::AbstractMatrix{<:Rational{Int64}},b::AbstractVector{<:Rational{Int64}},tol::Rational{Int64})
    #takes a IP of form Ax<=b and relaxes b using rao cuts

    #rationalize.(float.(A)) to convert A from Int64 to rational
    size(A,1) == size(b,1) || throw(DimensionMismatch("A and b must have the same number of rows"))

    for (index, row) in enumerate(eachrow(A))
        rowgcd = gcd(row)
        if rowgcd != 0
            A[index,:] = A[index,:]/rowgcd
            b[index] = fld(b[index],rowgcd)+1//1-tol
        end
    end
    return A,b
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
    set_silent(orientationModel)
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
    b = rationalize.(float.(b))
    tol = rationalize.(tol)
    A,b = relax(A,b,tol)
    flotSol, orientation = orientationMIP(A,b,c)
    A = A.*orientation'
    c = c.*orientation
    dMin,dMax = transformationMatrixBoundsWithC(A,c)
    return A, b, c, dMin, dMax
end # functions

function housekeeping(A,b,c,tol)
    A = rationalize.(float.(A))
    b = rationalize.(float.(b))
    A,b = relax(A,b,rationalize.(tol))
    flotSol, orientation = orientationMIP(A,b,c)
    A = A.*orientation'
    c = c.*orientation
    dMin,dMax = transformationMatrixBoundsWithC(A,c)
    return A, b, c, dMin, dMax
end


function qSolve(d,A,b,c)
    m,n = size(A)
    myMethodModel = Model(CPLEX.Optimizer)
    set_silent(myMethodModel)
    @variable(myMethodModel, x[1:n])

    @variable(myMethodModel, q[1:n] >=0)

    transmat = reduce(vcat,fill(d.*q,n)')+diagm(q)
    r = zeros(n)
    s = zeros(n)

    r[sign.(d).==1] = d[sign.(d).==1]
    #r = d .* (sign.(d).==1)
    #s = -d .* (sign.(d).==-1)
    s[sign.(d).==-1] = -d[sign.(d).==-1]


    #rSum = sum(r)
    #sSum = sum(s)
    #dSum = sum(d)

    @objective(myMethodModel, Max,
        c' * (x + 1/2 .* transmat*sign.(c))
        #c'*x
    )

    @constraint(myMethodModel, MainCon[i=1:m],
        A[i,:]' * (x + 1/2 .* transmat*sign.(A[i,:])) <= b[i]
    )

    @constraint(myMethodModel, rCon[i=1:n],
        #1 + rSum -r[i]<=q[i]*(1+dSum)
        1+sum(r[j] for j in filter(filt -> filt!=i, 1:n))<= q[i]*(1+sum(d[j] for j in 1:n))
    )

    @constraint(myMethodModel, sCon[i=1:n],
        #sSum -s[i]<=q[i]*(1+dSum
        -sum(s[j] for j in filter(filt -> filt!=i, 1:n)) <= q[i]*(1+sum(d[j] for j in 1:n))
    )

    println(myMethodModel)
    JuMP.optimize!(myMethodModel)
    println(termination_status(myMethodModel))

    return objective_value(myMethodModel), value.(x), value.(q)
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
        c' * (x - 1/2 .* transmat*sign.(c))
        #c'*x
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


function NLsolveVersion2(A,b,c,dMin,dMax)
    m,n = size(A)
    myMethodModel = Model(NLopt.Optimizer)

    set_optimizer_attribute(myMethodModel, "algorithm", :LD_MMA)

    #set_optimizer_attribute(myMethodModel, "xtol_rel", 0.1)
    @variable(myMethodModel, x[1:n])

    @variable(myMethodModel, q[1:n] >=0)

    @variable(myMethodModel, d[1:n])

    @variable(myMethodModel, r[1:n] >=0)

    @variable(myMethodModel, s[1:n] >=0)

    #@variable(myMethodModel, u[1:n])

    transmat = reduce(vcat,fill(d .* q,n)')+diagm(q)

    #r = d .* (sign.(d).==1)
    #s = -d .* (sign.(d).==-1)

    #rSum = sum(r)
    #sSum = sum(s)
    #dSum = sum(d)


    @NLobjective(myMethodModel, Max,
        c' * (x + 1/2 .* transmat*sign.(c))
        #c'*x
    )

    @NLconstraint(myMethodModel, MainCon[i=1:m],
        A[i,:]' * (x + 1/2 .* transmat*sign.(A[i,:])) <= b[i]
    )

    @NLconstraint(myMethodModel, OrderCon1[i=1:n],
        1+sum(r[j] for j in 1:n)-r[i] <= q[i]*(1+sum(d[j] for j in 1:n))
    )

    @NLconstraint(myMethodModel, OrderCon2[i=1:n],
        sum(s[j] for j in 1:n)-s[i] <= q[i]*(1+sum(d[j] for j in 1:n))
    )

    @constraint(myMethodModel, dConMin,
        -1 .<= d .* dMin
    )

    @constraint(myMethodModel, dConMax,
        -1 .<= d .* dMax
    )

    @constraint(myMethodModel, determminantCon,
        sum(d)>=-1
    )

    @constraint(myMethodModel, rCon,
        r .>= d
    )

    @constraint(myMethodModel, sCon,
        s .>= -d
    )
    #@constraint(myMethodModel, dDefineCon,
    #    d .* q .>= u
    #)
    println(myMethodModel)
    JuMP.optimize!(myMethodModel)
    println(termination_status(myMethodModel))
    return objective_value(myMethodModel), value.(x), value.(q), value.(d)
end

#use uv
function NLsolveVersion3(A,b,c,dMin,dMax)
    m,n = size(A)
    myMethodModel = Model(NLopt.Optimizer)

    set_optimizer_attribute(myMethodModel, "algorithm", :LD_MMA)

    @variable(myMethodModel, x[1:n])

    @variable(myMethodModel, q[1:n] >=0)

    #@variable(myMethodModel, d[1:n])
    @variable(myMethodModel, u[1:n])


    @variable(myMethodModel, r[1:n] >=0)

    @variable(myMethodModel, s[1:n] >=0)

    #@variable(myMethodModel, u[1:n])
    #d = u./q

    transmat = reduce(vcat,fill(u,n)')+diagm(q)

    #r = d .* (sign.(d).==1)
    #s = -d .* (sign.(d).==-1)

    #rSum = sum(r)
    #sSum = sum(s)
    #dSum = sum(d)


    @objective(myMethodModel, Max,
        c' * (x - 1/2 .* transmat*sign.(c))
        #c'*x
    )

    @constraint(myMethodModel, MainCon[i=1:m],
        A[i,:]' * (x + 1/2 .* transmat*sign.(A[i,:])) <= b[i]
    )

    @NLconstraint(myMethodModel, OrderCon1[i=1:n],
        1+sum(r[j] for j in 1:n)-r[i] <= q[i]*(1+sum(u[j] / q[j] for j in 1:n))
    )

    @NLconstraint(myMethodModel, OrderCon2[i=1:n],
        sum(s[j] for j in 1:n)-s[i] <= q[i]*(1+sum(u[j] / q[j] for j in 1:n))
    )

    @constraint(myMethodModel, dConMin,
        0 .<=q+ u.* dMin
    )

    @constraint(myMethodModel, dConMax,
        0 .<=q+ u.* dMax
    )

    @NLconstraint(myMethodModel, determminantCon,
        sum(u[j] / q[j] for j in 1:n)>=-1
    )

    @constraint(myMethodModel, rCon,
        r .* q .>= u
    )

    @constraint(myMethodModel, sCon,
        s .* q .>= -u
    )
    #@constraint(myMethodModel, dDefineCon,
    #    d .* q .>= u
    #)

    JuMP.optimize!(myMethodModel)


    return objective_value(myMethodModel), value.(x), value.(q), value.(d)
end


function NLsolveVersion4(A,b,c,dMin,dMax)
    m,n = size(A)
    myMethodModel = Model(NLopt.Optimizer)

    set_optimizer_attribute(myMethodModel, "algorithm", :LD_MMA)

    #set_optimizer_attribute(myMethodModel, "xtol_rel", 0.1)
    @variable(myMethodModel, x[1:n])

    @variable(myMethodModel, q[1:n] >=0, start = 1)

    #@variable(myMethodModel, d[1:n])

    @variable(myMethodModel, r[1:n] >=0, start = 0)

    @variable(myMethodModel, s[1:n] >=0, start = 0)



    #@variable(myMethodModel, u[1:n])
    u = ones(n)
    d = r - s

    transmat = transpose((d.*q)*u') +diagm(q)

    objfun = c' * (x + 1/2 .* transmat*sign.(c))

    #mainfun = A[i,:]' * (x + 1/2 .* transmat*sign.(A[i,:]))
    mainfun = zeros(QuadExpr,m)
    for i in 1:m
        mainfun[i] = A[i,:]' * (x + 1/2 .* transmat*sign.(A[i,:]))
    end



    #r = d .* (sign.(d).==1)
    #s = -d .* (sign.(d).==-1)

    #rSum = sum(r)
    #sSum = sum(s)
    #dSum = sum(d)


    @NLobjective(myMethodModel, Max,
        objfun
        #c'*x
        #objfun
    )

    @NLconstraint(myMethodModel, MainCon[i=1:m],
        mainfun[i] <= b[i]
        #A[i,:]' * (x + 1/2 .* transmat*sign.(A[i,:])) <= b[i]
    )

    @NLconstraint(myMethodModel, OrderCon1[i=1:n],
        1+sum(r[j] for j in filter(filt -> filt!=i, 1:n))<= q[i]*(1+sum(d[j] for j in 1:n))
    )

    @NLconstraint(myMethodModel, OrderCon2[i=1:n],
        -sum(s[j] for j in filter(filt -> filt!=i, 1:n)) <= q[i]*(1+sum(d[j] for j in 1:n))
    )

    @NLconstraint(myMethodModel, dConMin[i=1:n],
        #-1 <= d .* dMin
        -1 <= d[i]*dMin[i]
    )

    @NLconstraint(myMethodModel, dConMax[i=1:n],
        #-1 <= d .* dMax
        -1 <= d[i]*dMax[i]
    )

    @NLconstraint(myMethodModel, determminantCon,
        sum(d[j] for j in 1:n)>=-1
    )

    #constraint(myMethodModel, rCon,
    #    r .>= d
    #)

    #@constraint(myMethodModel, sCon,
    #    s .>= -d
    #)
    #@constraint(myMethodModel, dDefineCon,
    #    d .* q .>= u
    #)
    println(myMethodModel)
    JuMP.optimize!(myMethodModel)
    println(termination_status(myMethodModel))


    return objective_value(myMethodModel), value.(x), value.(q), value.(r), value.(s)
end




function IntSolve(x,q,d,A,b,c)
    n=length(x)
    IntModel = Model(CPLEX.Optimizer)

    set_silent(IntModel)
    @variable(IntModel, gamma[1:n], Int)

    #@variable(IntModel, 0 <= s[1:n] <= 1)

    @objective(IntModel, Max, c' * gamma)

    transmat = reduce(vcat,fill(d .* q,n)')+diagm(q)

    v = d.*q
    u = ones(n)
    Ainv = diagm(1 ./q)

    mult = 1 + v'*Ainv*u

    #problem with this when q is zero
    #lhs = (mult * Ainv-Ainv*u*v'*Ainv)

    lhs = Ainv-(Ainv*u*v'*Ainv)/mult

    #lhs = Ainv-(Ainv*u*v'*Ainv)/(1 + v'*Ainv*u)

    #@constraint(IntModel, mainConstraint,
    #    gamma - x .<= transmat * (fill(1/2, n) - s)
    #)

    #@constraint(IntModel, mainConstraint1,
        #lhs *(gamma - x) .<= fill(mult/2, n)
    #    lhs *(gamma - x) .<= fill(1/2, n)
    #)

    x2= gamma - x

    @constraint(IntModel, mainConstraint1[i=1:n],
        x2[i]+sum(d[j]*(x2[i]-x2[j]) for j in 1:n) <= q[i]*(1+sum(d))/2
    )

    #@constraint(IntModel, mainConstraint2,
        #lhs *(gamma - x) .>= -fill(mult/2, n)
    #    lhs *(gamma - x) .>= -fill(1/2, n)
    #)

    @constraint(IntModel, mainConstraint2[i=1:n],
        x2[i]+sum(d[j]*(x2[i]-x2[j]) for j in 1:n) >= -q[i]*(1+sum(d))/2
    )

    @constraint(IntModel, originalConstraints,
        A*gamma .<=b
    )


    println(IntModel)
    JuMP.optimize!(IntModel)
    println(termination_status(IntModel))

    #println(primal_feasibility_report(IntModel, Dict(gamma .=> [2.0,0.0,0.0,1.0]) ))

    return objective_value(IntModel), value.(gamma)
end


function testing(n,tol)
    A, b, c, dMin, dMax = finding_bounds_xor(n,tol)
    #b=rationalize.(b)
    #obj, x, q, d = NLsolveVersion(A,b,c,dMin)
    #obj, x, q, d = NLsolveVersion2(A,b,c,dMin,dMax)
    #obj, x, q, d = NLsolveVersion3(A,b,c,dMin,dMax)

    obj, x, q, d = NLsolveVersion4(A,b,c,dMin,dMax)

    obj2,x2,q2 = qSolve(d,A,b,c)
    trueobj, truex = IntSolve(x2,q2,d,A,b,c)

    return A, b, c, q2, d, trueobj, truex
end

function testing2(AOrig,bOrig,cOrig,tol)
    A = rationalize.(float.(AOrig))
    b = rationalize.(float.(bOrig))
    A,b = relax(A,b,rationalize.(tol))
    flotSol, orientation = orientationMIP(A,b,cOrig)
    A = A.*orientation'
    c = cOrig.*orientation
    dMin,dMax = transformationMatrixBoundsWithC(A,c)

    obj, x, q, d = NLsolveVersion4(A,b,c,dMin,dMax)

    obj2,x2,q2 = qSolve(d,A,b,c)
    trueobj, truex = IntSolve(x2,q2,d,A,bOrig,c)

    truex = truex.*orientation
    return AOrig, bOrig, cOrig, q2, d, trueobj, truex, all(AOrig*truex .<= bOrig)
end


function oldstuff()

    relax!(A,b,tol)

    #this is wrong need to do orientation first as that can massively impact bounds on u
    mini, maxi = transformationMatrixBounds([A; c']) #include c in orientation constriants

    flotSol, orientation = orientationMIP(A,b,c)

    ~,~,uactual,vactual =myMethodUV(A,b,c,mini,orientation)

    n,m = size(A)
    actualT = reduce(vcat,fill(uactual,m)') -diagm(uactual) + diagm(vactual)
end
