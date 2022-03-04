
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


function oldstuff()

    relax!(A,b,tol)

    #this is wrong need to do orientation first as that can massively impact bounds on u
    mini, maxi = transformationMatrixBounds([A; c']) #include c in orientation constriants

    flotSol, orientation = orientationMIP(A,b,c)

    ~,~,uactual,vactual =myMethodUV(A,b,c,mini,orientation)

    n,m = size(A)
    actualT = reduce(vcat,fill(uactual,m)') -diagm(uactual) + diagm(vactual)
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



conflict_constraint_list = ConstraintRef[]
for (F, S) in list_of_constraint_types(orientationModel)
    for con in all_constraints(orientationModel, F, S)
        if MOI.get(orientationModel, MOI.ConstraintConflictStatus(), con) == MOI.IN_CONFLICT
            push!(conflict_constraint_list, con)
            println(con)
        end
    end
end
