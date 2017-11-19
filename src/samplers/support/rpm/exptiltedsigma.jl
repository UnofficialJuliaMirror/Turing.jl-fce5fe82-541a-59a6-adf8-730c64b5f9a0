immutable ExpTiltedSigma <: ContinuousUnivariateDistribution
  a::Real
  l::Real
end

function Distributions.rand(d::ExpTiltedSigma)
  a = d.a; l = d.l;
  g = (cos(a*pi/2))^(1/a)
  m = Int(max(1, round(l^a)))
  S = zeros(Real, m)

  for k in 1:m
    Sk = 0.0
    U = 2.0
    while U > exp(-l * Sk)
      Sk = stblrnd(a, 1, g/m^(1/a), floor(a)/m)[1]
      U = rand()
    end

    S[k] = Sk
  end

  sum(S)
end

function stblrnd(alpha, beta, gamma, delta)
  sizeOut = 1

  if alpha == 2 # Gaussian distribution
    r = sqrt(2) * randn(sizeOut)

  elseif alpha == 1 && beta == 0  # Cauchy distribution
    r = tan( pi/2 * (2*rand(sizeOut) - 1) )

  elseif alpha == .5 && abs(beta) == 1 # Levy distribution (a.k.a. Pearson V)
    r = beta ./ randn(sizeOut).^2

  elseif beta == 0 # Symmetric alpha-stable
    V = pi/2 * (2*rand(sizeOut) - 1)
    W = -log(rand(sizeOut))
    r = sin(alpha * V) ./ ( cos(V).^(1/alpha) ) .* ( cos( V.*(1-alpha) ) ./ W ).^( (1-alpha)/alpha )

  elseif alpha != 1 # General case, alpha not 1
    V = pi/2 * (2*rand(sizeOut) - 1)
    W = - log( rand(sizeOut) )
    const_tmp = beta * tan(pi*alpha/2)
    B = atan( const_tmp )
    S = (1 + const_tmp * const_tmp).^(1/(2*alpha))
    r = S * sin( alpha*V + B ) ./ ( cos(V) ).^(1/alpha) .* ( cos( (1-alpha) * V - B ) ./ W ).^((1-alpha)/alpha)

  else # General case, alpha = 1
    V = pi/2 * (2*rand(sizeOut) - 1)
    W = - log( rand(sizeOut) )
    piover2 = pi/2
    sclshftV =  piover2 + beta * V
    r = 1/piover2 * ( sclshftV .* tan(V) - beta * log( (piover2 * W .* cos(V) ) ./ sclshftV ) )
  end

  # Scale and shift
  if alpha != 1
    r = gamma * r + delta
  else
    r = gamma * r + (2/pi) * beta * gamma * log(gamma) + delta
  end

  r
end
