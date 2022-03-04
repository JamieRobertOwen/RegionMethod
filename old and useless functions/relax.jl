function relax!(A::AbstractMatrix{<:Rational{Int64}},b::AbstractVector{<:AbstractFloat},tol::AbstractFloat)
    #rationalize.(float.(A)) to convert A from Int64 to rational
    size(A,1) == size(b,1) || throw(DimensionMismatch("A and b must have the same number of rows"))

    for (index, row) in enumerate(eachrow(A))
        rowgcd = gcd(row)
        if rowgcd != 0
            A[index,:] = A[index,:]/rowgcd
            b[index] = fld(b[index],rowgcd)+1.0-tol
        end
    end

    #return A,b
end
