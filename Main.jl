using JuMP, CPLEX, LinearAlgebra, NLopt, DataFrames, PlotlyJS


#test cases
include("test_cases.jl")

#testing
A = [3 2 5; -2 1 1; 1 -1 3; 5 2 -4; -1 0 0; 0 -1 0; 0 0 -1]

b = [55; 26; 30; 57; 0; 0; 0]

c = [20, 10, 15]

tol = 0.1

A = rationalize.(float.(A))

b = float.(b)




function relax(A::AbstractMatrix{<:Rational{Int64}},b::AbstractVector{<:Rational{Int64}})
    #takes a IP of form Ax<=b and relaxes b using rao cuts

    #rationalize.(float.(A)) to convert A from Int64 to rational
    size(A,1) == size(b,1) || throw(DimensionMismatch("A and b must have the same number of rows"))

    for (index, row) in enumerate(eachrow(A))
        rowgcd = gcd(row)
        if rowgcd != 0
            A[index,:] = A[index,:]/rowgcd
            b[index] = b[index]/rowgcd
        end
    end
    return A,b
end

function relaxMIP(A::AbstractMatrix{<:Rational{Int64}},G,b::AbstractVector{<:Rational{Int64}})
    #takes a IP of form Ax<=b and relaxes b using rao cuts

    #rationalize.(float.(A)) to convert A from Int64 to rational
    size(A,1) == size(b,1) || throw(DimensionMismatch("A and b must have the same number of rows"))

    for (index, row) in enumerate(eachrow(A))
        rowgcd = gcd(row)
        if rowgcd != 0
            A[index,:] = A[index,:]/rowgcd
            G[index,:] = G[index,:]/rowgcd
            b[index] = b[index]/rowgcd
        end
    end

    return A,G,b
end


function transformationMatrixBoundsWithC(A::AbstractMatrix, c::AbstractVector)
    A= [A; c']
    rowsum = sum(A,dims=2)
    Areciprical = 1 ./ A
    umin= zeros(Rational,size(A,2))
    umax= zeros(Rational,size(A,2))

    for (index, col) in enumerate(eachcol(A))
        umin[index] = minimum(rowsum[col.!=0] ./ col[col.!=0])
        umax[index] = maximum(rowsum[col.!=0] ./ col[col.!=0])
    end
    return umin,umax
end

function findOrientation(A,b,c)
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


function findOrientationMIP(A,G,b,c,h)
    n,m = size(A)
    n2,m2 = size(G)
    orientationModel = Model(CPLEX.Optimizer)
    set_silent(orientationModel)


    @variable(orientationModel, x[1:m])

    @variable(orientationModel, y[1:m2])

    @variable(orientationModel, u[1:m], Bin)

    fix(u[1], 1; force = true)

    v = u.-1/2

    @objective(orientationModel, Max, c' * x + h' * y)

    @constraint(orientationModel, mainConstraint1,
        A * (x+v/2) + G*y .<= b-sum(abs.(A), dims=2)/4
    )

    @constraint(orientationModel, mainConstraint2,
        A * (x-v/2) + G*y .<= b-sum(abs.(A), dims=2)/4
    )

    JuMP.optimize!(orientationModel)

    return value.(x), value.(y), (2*round.(Int,value.(u)) .-1)
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
        #c' * (x - 1/2 .* transmat*sign.(c))
        c'*x
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
        sum(s[j] for j in filter(filt -> filt!=i, 1:n)) <= q[i]*(1+sum(d[j] for j in 1:n))
    )

    println(myMethodModel)
    JuMP.optimize!(myMethodModel)
    println(termination_status(myMethodModel))

    if termination_status(myMethodModel) == OPTIMAL
        return objective_value(myMethodModel), value.(x), value.(q)
    else
        return NaN,NaN,NaN
    end
end


function qSolve2(d,A,b,c,solPercentile)
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
        c' * (x + solPercentile*(1/2 .* transmat*sign.(c)))
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
        sum(s[j] for j in filter(filt -> filt!=i, 1:n)) <= q[i]*(1+sum(d[j] for j in 1:n))
    )

    #println(myMethodModel)
    JuMP.optimize!(myMethodModel)
    #println(termination_status(myMethodModel))

    if termination_status(myMethodModel) == OPTIMAL
        return objective_value(myMethodModel), value.(x), value.(q)
    else
        return NaN,NaN,NaN
    end
end


function qSolveMIP(d,A,G,b,c,h, ystart)
    m,n = size(A)
    m2,n2 = size(G)
    myMethodModel = Model(CPLEX.Optimizer)
    set_silent(myMethodModel)
    @variable(myMethodModel, x[1:n])

    @variable(myMethodModel, y[i=1:n2], start = ystart[i])

    @variable(myMethodModel, q[1:n] >=0)

    transmat = reduce(vcat,fill(d.*q,n)')+diagm(q)
    r = zeros(n)
    s = zeros(n)

    r[sign.(d).==1] = d[sign.(d).==1]
    s[sign.(d).==-1] = -d[sign.(d).==-1]

    @objective(myMethodModel, Max,
        #c' * (x + 1/2 .* transmat*sign.(c)) + h' * y
        c'*x+h' * y
    )

    @constraint(myMethodModel, MainCon[i=1:m],
        A[i,:]' * (x + 1/2 .* transmat*sign.(A[i,:])) + G[i,:]'*y <= b[i]
    )

    @constraint(myMethodModel, rCon[i=1:n],
        #1 + rSum -r[i]<=q[i]*(1+dSum)
        1+sum(r[j] for j in filter(filt -> filt!=i, 1:n))<= q[i]*(1+sum(d[j] for j in 1:n))
    )

    @constraint(myMethodModel, sCon[i=1:n],
        #sSum -s[i]<=q[i]*(1+dSum
        sum(s[j] for j in filter(filt -> filt!=i, 1:n)) <= q[i]*(1+sum(d[j] for j in 1:n))
    )

    println(myMethodModel)
    JuMP.optimize!(myMethodModel)
    println(termination_status(myMethodModel))

    return objective_value(myMethodModel), value.(x), value.(y), value.(q)
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

    #transmat = transpose((d.*q)*u') +diagm(q)
    transmat = reduce(vcat,fill(d.*q,n)')+diagm(q)

    #objfun = c' * (x + 1/2 .* transmat*sign.(c))

    objfun = c' * x

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
        sum(s[j] for j in filter(filt -> filt!=i, 1:n)) <= q[i]*(1+sum(d[j] for j in 1:n))
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


    return objective_value(myMethodModel), value.(x), value.(q), value.(r)-value.(s)
end

function NLsolveMIP(A,G,b,c,h,dMin,dMax, ystart)
    m,n = size(A)
    m2,n2 = size(G)
    myMethodModel = Model(NLopt.Optimizer)

    set_optimizer_attribute(myMethodModel, "algorithm", :LD_MMA)

    @variable(myMethodModel, x[1:n])

    @variable(myMethodModel, q[1:n] >=0, start = 1)

    @variable(myMethodModel, r[1:n] >=0, start = 0)

    @variable(myMethodModel, s[1:n] >=0, start = 0)

    @variable(myMethodModel, y[i=1:n2], start = ystart[i])

    u = ones(n)
    d = r - s

    transmat = transpose((d.*q)*u') +diagm(q)

    #objfun = c' * (x + 1/2 .* transmat*sign.(c)) + h' * y
    objfun = c' * x  + h' * y

    mainfun = zeros(QuadExpr,m)
    for i in 1:m
        mainfun[i] = A[i,:]' * (x + 1/2 .* transmat*sign.(A[i,:]))
    end

    mainfun = mainfun + G * y

    @NLobjective(myMethodModel, Max,
        objfun
    )

    @NLconstraint(myMethodModel, MainCon[i=1:m],
        mainfun[i] <= b[i]
    )

    @NLconstraint(myMethodModel, OrderCon1[i=1:n],
        1+sum(r[j] for j in filter(filt -> filt!=i, 1:n))<= q[i]*(1+sum(d[j] for j in 1:n))
    )

    @NLconstraint(myMethodModel, OrderCon2[i=1:n],
        sum(s[j] for j in filter(filt -> filt!=i, 1:n)) <= q[i]*(1+sum(d[j] for j in 1:n))
    )

    @NLconstraint(myMethodModel, dConMin[i=1:n],
        -1 <= d[i]*dMin[i]
    )

    @NLconstraint(myMethodModel, dConMax[i=1:n],
        -1 <= d[i]*dMax[i]
    )

    @NLconstraint(myMethodModel, determminantCon,
        sum(d[j] for j in 1:n)>=-1
    )

    println(myMethodModel)
    JuMP.optimize!(myMethodModel)
    println(termination_status(myMethodModel))


    return objective_value(myMethodModel), value.(x), value.(y), value.(q), value.(r)-value.(s)
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


    #println(IntModel)
    JuMP.optimize!(IntModel)
    #println(termination_status(IntModel))

    #println(primal_feasibility_report(IntModel, Dict(gamma .=> [2.0,0.0,0.0,1.0]) ))

    return objective_value(IntModel), value.(gamma)
end


function IntSolve2(x,q,d,A,b,c)
    n=length(x)
    IntModel = Model(CPLEX.Optimizer)

    set_silent(IntModel)
    @variable(IntModel, gamma[1:n], Int)

    @variable(IntModel, -1/2 <= s[1:n] <= 1/2)

    @objective(IntModel, Max, c' * gamma)

    transmat = reduce(vcat,fill(d .* q,n)')+diagm(q)

    #v = d.*q
    #u = ones(n)
    #Ainv = diagm(1 ./q)

    #mult = 1 + v'*Ainv*u

    #lhs = Ainv-(Ainv*u*v'*Ainv)/mult


    x2= gamma - x

    #@constraint(IntModel, mainConstraint1[i=1:n],
    #    x2[i]+sum(d[j]*(x2[i]-x2[j]) for j in 1:n) <= q[i]*(1+sum(d))/2
    #)

    @constraint(IntModel, mainConstraint,
        x2.==transmat * s
    )

    #@constraint(IntModel, mainConstraint2[i=1:n],
    #    x2[i]+sum(d[j]*(x2[i]-x2[j]) for j in 1:n) >= -q[i]*(1+sum(d))/2
    #)

    @constraint(IntModel, originalConstraints,
        A*gamma .<=b
    )


    #println(IntModel)
    JuMP.optimize!(IntModel)
    #println(termination_status(IntModel))

    #println(primal_feasibility_report(IntModel, Dict(gamma .=> [2.0,0.0,0.0,1.0]) ))

    return objective_value(IntModel), value.(gamma)
end


function IntSolveMIP(x,y,q,d,A,G,b,c,h)
    n=length(x)
    n2 = length(y)
    IntModel = Model(CPLEX.Optimizer)

    set_silent(IntModel)
    @variable(IntModel, gamma[1:n], Int)

    @variable(IntModel, theta[i=1:n2], start = y[i])

    @objective(IntModel, Max, c' * gamma + h' * theta)

    x2= gamma - x


    @constraint(IntModel, mainConstraint1[i=1:n],
        x2[i]+sum(d[j]*(x2[i]-x2[j]) for j in 1:n) <= q[i]*(1+sum(d))/2
    )

    @constraint(IntModel, mainConstraint2[i=1:n],
        x2[i]+sum(d[j]*(x2[i]-x2[j]) for j in 1:n) >= -q[i]*(1+sum(d))/2
    )

    @constraint(IntModel, originalConstraints,
        A*gamma + G*theta .<=b
    )


    println(IntModel)
    JuMP.optimize!(IntModel)
    println(termination_status(IntModel))
    return objective_value(IntModel), value.(gamma), value.(theta)
end

function presolve(AOrig,bOrig,cOrig,tol)
    A = rationalize.(float.(AOrig))
    b = rationalize.(float.(bOrig))
    c = rationalize.(float.(cOrig))
    A,btight = relax(A,b)
    b = floor.(btight) .+ (1//1 - rationalize(tol))
    flotSol, orientation = findOrientation(A,b,c)
    A = A.*orientation'
    c = c.*orientation
    dMin,dMax = transformationMatrixBoundsWithC(A,c)

    return A,b,c,dMin,dMax,btight,orientation
end

function presolveMIP(AOrig,GOrig,bOrig,cOrig,h,tol)
    A = rationalize.(float.(AOrig))
    b = rationalize.(float.(bOrig))
    G = rationalize.(float.(GOrig))
    c = rationalize.(float.(cOrig))
    A,G,btight = relaxMIP(A,G,b)
    b = floor.(btight) .+ (1//1 - rationalize(tol))
    flotSol,yStart, orientation = findOrientationMIP(A,G,b,c,h)
    A = A.*orientation'
    c = c.*orientation
    dMin,dMax = transformationMatrixBoundsWithC(A,c)

    return A,G,b,c,dMin,dMax,yStart,btight,orientation
end

function testing2(AOrig,bOrig,cOrig,tol)
    A,b,c,dMin,dMax,btight,orientation = presolve(AOrig,bOrig,cOrig,tol)

    obj, x, q, d = NLsolveVersion4(A,b,c,dMin,dMax)

    obj2,x2,q2 = qSolve(d,A,b,c)
    #trueobj, truex = IntSolve(x2,q2,d,A,bOrig,c)

    #slack variable version - may be more temperamental
    trueobj, truex = IntSolve2(x2,q2,d,A,btight,c)

    truex = truex.*orientation
    return q2, d, trueobj, truex, all(AOrig*truex .<= bOrig)
end

function testingMIP(AOrig,GOrig,bOrig,cOrig,h,tol)

    A,G,b,c,dMin,dMax,yStart,btight,orientation = presolveMIP(AOrig,GOrig,bOrig,cOrig,h,tol)


    obj, x, y, q, d = NLsolveMIP(A,G,b,c,h,dMin,dMax, yStart)

    obj2,x2,y2,q2 = qSolveMIP(d,A,G,b,c,h, y)
    trueobj, truex, truey = IntSolveMIP(x2,y2,q2,d,A,G,btight,c,h)

    truex = truex.*orientation
    return q2, d, trueobj, truex,truey, all(AOrig*truex+G*truey .<= bOrig)
end

function SolveMIP(A,G,b,c,h)
    m,n = size(A)
    m2,n2 = size(G)
    IntModel = Model(CPLEX.Optimizer)
    @variable(IntModel, x[1:n], Int)
    @variable(IntModel, y[1:n2])
    @objective(IntModel, Max, c' * x + h' * y)
    @constraint(IntModel, MainCon, A*x + G*y .<= b)
    JuMP.optimize!(IntModel)
    return objective_value(IntModel), value.(x), value.(y)
end

function SolveINT(A,b,c)
    m,n = size(A)
    IntModel = Model(CPLEX.Optimizer)
    @variable(IntModel, x[1:n], Int)
    @objective(IntModel, Max, c' * x)
    @constraint(IntModel, MainCon, A*x.<= b)
    JuMP.optimize!(IntModel)
    return objective_value(IntModel), value.(x)
end

function plot2Dexample(AOrig,bOrig,cOrig,tol,resolution,solPercentile)

    A,b,c,dMin,dMax,btight,orientation = presolve(AOrig,bOrig,cOrig,tol)
    all([sign.(dMin).==-1 ; sign.(dMax).==1]) || throw("lack of bounds on d")
    Results = DataFrames.DataFrame(
    d1 = Float64[],
    d2 = Float64[],
    q1 = Float64[],
    q2 = Float64[],
    preobj = Float64[],
    actualobj = Float64[],
    )
    #generator = ModelGenerator(zeros(size(c)), A,b,c

    #resolution = 20

    # need to add error checking here
    for i = range(-1/dMax[1],-1/dMin[1], length =resolution)
        for j = range(-1/dMax[2],-1/dMin[2], length =resolution)

            preobj,x,q = qSolve2([i,j],A,b,c,solPercentile)
            if isnan(preobj) == false
                trueobj, truex = IntSolve2(x,q,[i,j],A,btight,c)
                push!(Results, (float(i),float(j), q[1],q[2], preobj, trueobj))
            else
                push!(Results,(float(i),float(j), NaN, NaN, NaN, NaN))
            end
        end
    end

    #d1range = float.(collect(range(-1/dMax[1],-1/dMin[1], length =resolution)))
    #d2range = float.(collect(range(-1/dMax[2],-1/dMin[2], length =resolution)))


    #q1range = collect(range(extrema(), length =resolution))
    #d1results = Results.d1
    #d2results = Results.d2
    #preobjresults = Results.preobj

    #d1resultsfilt = d1results[isnan.(preobjresults).==false]
    #d2resultsfilt = d2results[isnan.(preobjresults).==false]
    #preobjresultsfilt = preobjresults[isnan.(preobjresults).==false]

    #this is the wrong way round for normal things but plotting is weird
    #resultsReshape = reshape(Results.preobj, resolution,:)'
    #actualresultsReshape = reshape(Results.actualobj, resolution,:)'

    #data = contour(x=d1range,y=d2range,z=resultsReshape',contours_coloring="heatmap")
    #data2 = contour(x=d1range,y=d2range,z=actualresultsReshape',contours_coloring="heatmap")

    #layout = Layout(xaxis_range=[-1/dMax[1],-1/dMin[1]], yaxis_range=[-1/dMax[2],-1/dMin[2]])
    #plot(data)
    #plot(data2)

    #surfaceplotfigd = make_subplots(rows = 2, cols =1,specs = fill(Spec(kind="scene"),2,1) ,row_titles=["predicted output" ; "actual output"])
    #add_trace!(surfaceplotfigd,surface(x=d1range,y=d2range,z=resultsReshape),row=1, col=1)
    #add_trace!(surfaceplotfigd,surface(x=d1range,y=d2range,z=actualresultsReshape),row=2, col=1)

    #scatterplot = scatter(x=Results.preobj, y=Results.actualobj, mode="markers")
    return Results
end

function resultsplot(Results,resolution)


    d1range = float.(collect(range(extrema(Results.d1)..., length =resolution)))
    d2range = float.(collect(range(extrema(Results.d2)..., length =resolution)))

    resultsReshape = reshape(Results.preobj, resolution,:)'
    actualresultsReshape = reshape(Results.actualobj, resolution,:)'

    surfaceplotfigd = make_subplots(rows = 2, cols =1,specs = fill(Spec(kind="scene"),2,1) ,row_titles=["predicted output" ; "actual output"])
    add_trace!(surfaceplotfigd,surface(x=d1range,y=d2range,z=resultsReshape),row=1, col=1)
    add_trace!(surfaceplotfigd,surface(x=d1range,y=d2range,z=actualresultsReshape),row=2, col=1)

    heatmapq = make_subplots(rows = 2, cols =1,specs = fill(Spec(kind="xy"),2,1) ,row_titles=["predicted output" ; "actual output"])
    add_trace!(heatmapq,heatmap(x = vec(Results.q1), y = vec(Results.q2), z= vec(Results.preobj)),row=1, col=1)
    add_trace!(heatmapq,heatmap(x = vec(Results.q1), y = vec(Results.q2), z= vec(Results.actualobj)),row=2, col=1)

    heatmapd = make_subplots(rows = 2, cols =1,specs = fill(Spec(kind="xy"),2,1) ,row_titles=["predicted output" ; "actual output"])
    add_trace!(heatmapd,heatmap(x = vec(Results.d1), y = vec(Results.d2), z= vec(Results.preobj)),row=1, col=1)
    add_trace!(heatmapd,heatmap(x = vec(Results.d1), y = vec(Results.d2), z= vec(Results.actualobj)),row=2, col=1)

    scatterobj = plot(scatter(x=Results.preobj, y=Results.actualobj, mode="markers"))

    display.([surfaceplotfigd,heatmapq,heatmapd,scatterobj])
    return surfaceplotfigd, heatmapq, heatmapd, scatterobj
end
