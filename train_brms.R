# Loading required libraries
library(brms)
library(cmdstanr)

# Ensuring that GPU is visible 
print(OpenCL::oclDevices())

# Hyper paramters
n <- 300000
k <- 10

# Constructing X and y matrices
X <- matrix(rnorm(n * k), ncol = k)
y <- rbinom(n, size = 1, prob = plogis(10 * X[,1] + 9 * X[,2] + 8 * X[,3] + 7 * X[,4] + 6 * X[,5] + 5 * X[,6] + 4 * X[,7] + 3 * X[,8] + 2 * X[,9] + 2 * X[,10] + 1))
 
# Making stan objects
code <- make_stancode(y ~ X1 + X2 + X3 + X4 + X5 + X6 + X7 + X8 + X9 + X10, data = data.frame(y, X), family = bernoulli(), refresh = 0)
data <- make_standata(y ~ X1 + X2 + X3 + X4 + X5 + X6 + X7 + X8 + X9 + X10, data = data.frame(y, X), family = bernoulli(), refresh = 0)
class(data) <- NULL

file_cpu <- write_stan_file(code)
file_cl <- write_stan_file(code)

# Fitting on CPU and GPU
mod_cpu <- cmdstan_model(file_cpu)
mod_cl <- cmdstan_model(file_cl, cpp_options = list(stan_opencl=TRUE))

fit_cl <- mod_cl$sample(data = data, seed = 123, chains = 4, parallel_chains = 4, opencl_ids = c(0,0))
fit_cpu <- mod_cpu$sample(data = data, seed = 123, chains = 4, parallel_chains = 4)