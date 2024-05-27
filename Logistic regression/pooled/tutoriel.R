set.seed(1234) #clustered example:: just for sim
pp <- function(x) exp(x) / (1 + exp(x))
NG <- 20 #number of groups
GS <- 15 #group size
total <- NG * GS
tr <- rep(sample(c(0,1), NG, 1), each = GS)
w1 <- rep(rnorm(NG), each = GS)
e2 <- rep(rnorm(NG, 0, .5), each = GS)
school <- rep(1:NG, each = GS)
x1 <- rnorm(total)
x2 <- rbinom(total, 1, .5)
ystar <- 1 + tr * .5 + w1 * .3 + x1 * .6 + x2 * -.4 + e2
y <- rbinom(total, 1, pp(ystar))
dat <- data.frame(y, school, tr, w1, x1, x2)
sel <- sample(total, 100) #randomly select data to remove
dat <- dat[-sel, ] #remove to create unbalanced data

X <- model.matrix(~tr + w1 + x1 + x2, data = dat) #just to quickly get this
#this is not the same as the weighted one
tol <- 1e-6 #tolerance: if change is low, then stop
y <- dat$y
mu <- (y + mean(y))/2 #for convergence
#eta <- log(mu) ## Poisson link
eta <- log(mu / (1 - mu)) #logit

dev <- 0 #deviance
delta.dev <- 1
i <- 1 #iteration number

## run the iterative procedure to get the weights

tmp <- mu * (1 - mu)
tmp
1 / (tmp * (1/tmp)^2)
W <- diag(1 / (tmp * (1/tmp)^2))
W
ncol(W)
nrow(W)

while(i < 50 & abs(delta.dev) > tol) { #repeat until deviance change is minimal 
  # or number of iterations is exceeded (e.g., 50 here)
  
  tmp <- mu * (1 - mu) #since I use this over and over again
  W <- diag(1 / (tmp * (1/tmp)^2))
  z <- eta + (y - mu) * (1 / tmp) 
  b <- solve(t(X) %*% W %*% X) %*% t(X) %*% W %*% z #these are the cofficients
  eta <- X %*% b #update eta
  mu <- as.vector( 1 / (1 + exp(-eta)))
  dev0 <- dev
  #dev <- -2 * sum(y * log(y/mu) + (1 - y) * (1 -log(y/mu)), na.rm = T)
  dev <- -2 * sum((y * log(mu)) + ((1 - y) * log(1 - mu))) #deviance
  delta.dev <- dev - dev0 #assess change in deviance for logistic reg
  cat(i, dev, "::")
  i <- i + 1 #increment by 1
}

serror <- sqrt(diag(solve(t(X) %*% W %*% X)))
# compare results
data.frame(B.manual = as.numeric(b), se.manual = serror)

rm(list = ls())

