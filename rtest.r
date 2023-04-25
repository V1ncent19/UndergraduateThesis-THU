
integrate(function(x){-qnorm(x)}, 1e-8,0.5)$value*2
integrate(function(x){qnorm(x)-qnorm(x,0,2)}, 1e-8,0.5)$value*2

cdf = function(x){pnorm(x,0,1)}
cdf2 = function(x){pnorm(x+1,0,1)}
cdf25 = function(x){pnorm(x-1,0,1)}


curve(cdf,from = -5, to = 5, n=500)
curve(cdf2,from = -5, to = 5, n=500, add = T)
curve(cdf25,from = -5, to = 5, n=500, add = T)

cdf3 = function(x){pnorm(x,0,3)}
cdf4 = function(x){pnorm(x+1,0,3)}
cdf45 = function(x){pnorm(x-1,0,3)}

curve(cdf3,from = -5, to = 5, n=500, add = T)
curve(cdf4,from = -5, to = 5, n=500, add = T)
curve(cdf45,from = -5, to = 5, n=500, add = T)

