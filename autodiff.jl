# Automatic differentiation using dual numbers
struct Dual <: Number
  x::Real
  dx::Real
end

# Convert a variable into a dual number
Dual(x) = Dual(x, one(x))

# Dual number unit
const ϵ = Dual(0, 1)

# Convert real numbers into dual numbers
Base.convert(::Type{Dual}, x::Real) = Dual(x, zero(x))
Base.promote_rule(::Type{Dual}, ::Type{<:Number}) = Dual

# Fancy visualization of dual numbers
Base.show(io::IO, f::Dual) = print(io, f.x, " + ", f.dx, "ϵ")

# Basic algebraic properties
Base.:+(f::Dual, g::Dual) = Dual(f.x + g.x, f.dx + g.dx)
Base.:-(f::Dual, g::Dual) = Dual(f.x - g.x, f.dx - g.dx)
Base.:*(f::Dual, g::Dual) = Dual(f.x * g.x, f.dx * g.x + f.x * g.dx)
Base.:/(f::Dual, g::Dual) = Dual(f.x / g.x, (f.dx * g.x - f.x * g.dx) / g.x^2)

# Extend basic functions to work with dual numbers
Base.sin(f::Dual) = sin(f.x) + cos(f.x) * f.dx * ϵ
Base.cos(f::Dual) = cos(f.x) - sin(f.x) * f.dx * ϵ
Base.exp(f::Dual) = exp(f.x) + exp(f.x) * f.dx * ϵ
Base.log(f::Dual) = log(f.x) + f.dx / f.x * ϵ

# TEST ============
@debug "Testing the autodifferentiation method" let
  f(x) = x^2 + 2x - cos(x)
  f(Dual(π))
end
