
function finding_bounds(A,b,c,tol)
    A = rationalize.(float.(A))
    b = float.(b)
    relax!(A,b,tol)
    flotSol, orientation = findOrientation(A,b,c)
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
    flotSol, orientation = findOrientation(A,b,c)
    A = A.*orientation'
    c = c.*orientation
    dMin,dMax = transformationMatrixBoundsWithC(A,c)
    return A, b, c, dMin, dMax
end # functions

function housekeeping(A,b,c,tol)
    A = rationalize.(float.(A))
    b = rationalize.(float.(b))
    A,b = relax(A,b,rationalize.(tol))
    flotSol, orientation = findOrientation(A,b,c)
    A = A.*orientation'
    c = c.*orientation
    dMin,dMax = transformationMatrixBoundsWithC(A,c)
    return A, b, c, dMin, dMax
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
