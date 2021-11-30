function orientationMIP(A,b,c)
    n,m = size(A)
    orientationModel = Model(CPLEX.Optimizer)

    @variable(orientationModel, x[1:m] )

    @variable(orientationModel, u[1:m], Bin)

    @objective(orientationModel, Max, c' * x)

    @constraint(orientationModel, mainConstraint1, A * (x+u.-(1/2)) .<= b)

    @constraint(orientationModel, mainConstraint2, A * (x-u.+(1/2)) .<= b)

    TT = stdout # save original STDOUT stream
    redirect_stdout()

    optimize!(orientationModel)

    redirect_stdout(TT)

    return value.(x), 2*(value.(u) .-1/2)
end
