require(mvtnorm)
require(nleqslv)
require(minpack.lm)
fit_owpl <- function(
  manifest,        # manifest dataset (n x p)
  constraintMat,   # constraints for loadings matrix (p x q)
  corrFLAG = 0,    # estimate latent covariance matrix
  silent = F,      # silent mode
  ncores = 1,
  solver = 'nleqslv',
  maxiter = 1000,
  init.method = 'nleqslv'
){
  start_time <- Sys.time()
  #### PREPARING MODEL INPUT
  n <- nrow(manifest) ### number of subjects
  p <- ncol(manifest) ### number of items
  q <- ncol(constraintMat) ### number of latent variables
  
  categories <- apply(manifest, 2, max) + 1 ### number of categories in each item
  
  lambda0_init <- c()
  s <- 0
  
  for (i in 1:length(categories)) {
    vec <- 1:(categories[i]-1)
    vec <- (vec -min(vec))/(max(vec)-min(vec))*(2)-1
    lambda0_init[(s + 1):(s + categories[i] - 1)] <- vec
    s <- s + categories[i] - 1
  }

  lambda_init = rep(0.5, sum(constraintMat))
  transformed_rhos_init = rep(0.5493061, q*(q-1)/2)
  
  #### PREPARING FITTING FUNCTIONS
  freq_tab <- pairs_freq(manifest, categories, ncores) 
  
  weightsVec <- rep(1, p*(p-1)/2)
  
  # function for nll
  pair_nll <- function(par_vec){
    lambda0_ <- par_vec[1:length(lambda0_init)]
    lambda_ <- par_vec[(length(lambda0_init)+1):(length(lambda0_init)+length(lambda_init))]
    transformed_rhos_ <- par_vec[(length(lambda0_init)+1+length(lambda_init)):length(par_vec)]
    mod <- pairwise_gllvm(y = manifest,
                          c_vec = categories,
                          A = constraintMat,
                          freq = freq_tab,
                          tau = lambda0_,
                          lambda = lambda_,
                          transformed_rhos = transformed_rhos_,
                          corrFLAG = corrFLAG,
                          grFLAG = 0,
                          ncores = 1,
                          silentFLAG = 1,
                          weights = weightsVec
    )
    
    out <- mod$nll
    return(out)
  }
  
  # function for gradient
  pair_gr <- function(par_vec){
    lambda0_ <- par_vec[1:length(lambda0_init)]
    lambda_ <- par_vec[(length(lambda0_init)+1):(length(lambda0_init)+length(lambda_init))]
    transformed_rhos_ <- par_vec[(length(lambda0_init)+1+length(lambda_init)):length(par_vec)]
    mod <- pairwise_gllvm(y = manifest,
                          c_vec = categories,
                          A = constraintMat,
                          freq = freq_tab,
                          tau = lambda0_,
                          lambda = lambda_,
                          transformed_rhos = transformed_rhos_,
                          corrFLAG = corrFLAG,
                          grFLAG = 1,
                          ncores = 1,
                          silentFLAG = 1,
                          weights = weightsVec
    )
    
    out <- mod$gradient
    return(out)
  }
  
  # parameter vector
  init_par <- c(lambda0_init, lambda_init, transformed_rhos_init)
  
  #### FIT
  if(init.method == 'nleqslv'){
    require(nleqslv)
    sol <- nleqslv(init_par, pair_gr, method = "Broyden", global= "dbldog", xscalm = "auto", control = list(allowSingular = T, maxit = maxiter, trace = 0))
    par <- sol$x
    message <- sol$termcd
    obj <- NULL
  }else if(init.method == 'nlminb'){
    sol <- nlminb(start = init_par, objective = pair_nll, gradient = pair_gr)
    par <- sol$par
  }
  if(silent == F) cat('Optimizing the model...\n')
  opt_par <- par
  
  # compute weights
  weights_wrap <- function(par_vec){
    lambda0_ <- par_vec[1:length(lambda0_init)]
    lambda_ <- par_vec[(length(lambda0_init)+1):(length(lambda0_init)+length(lambda_init))]
    transformed_rhos_ <- par_vec[(length(lambda0_init)+1+length(lambda_init)):length(par_vec)]
    mod <- owps_gllvm(y = manifest,
                      c_vec = categories,
                      A = constraintMat,
                      freq = freq_tab,
                      tau = lambda0_,
                      lambda = lambda_,
                      transformed_rhos = transformed_rhos_,
                      corrFLAG = corrFLAG,
                      ncores = 1,
                      silentFLAG = 1,
                      weightsFLAG = 0,
                      pre_weights = matrix(0,1,1)
    )
    
    out <- mod$weights
    return(out)
  }
  weights <- weights_wrap(opt_par)
  
  # weighted scores function
  owps_wrap <- function(par_vec){
    lambda0_ <- par_vec[1:length(lambda0_init)]
    lambda_ <- par_vec[(length(lambda0_init)+1):(length(lambda0_init)+length(lambda_init))]
    transformed_rhos_ <- par_vec[(length(lambda0_init)+1+length(lambda_init)):length(par_vec)]
    mod <- owps_gllvm(y = manifest,
                      c_vec = categories,
                      A = constraintMat,
                      freq = freq_tab,
                      tau = lambda0_,
                      lambda = lambda_,
                      transformed_rhos = transformed_rhos_,
                      corrFLAG = corrFLAG,
                      ncores = ncores,
                      silentFLAG = 1,
                      weightsFLAG = 1,
                      pre_weights = weights
    )
    
    out <- mod$ows
    return(out)
  }
  
  # solve the system
  if(solver == "nleqslv"){
    sol <- nleqslv(opt_par, owps_wrap, method = "Broyden", global= "dbldog", xscalm = "auto", control = list(allowSingular = T, maxit = maxiter, trace = 0))
    par <- sol$x
    message <- sol$termcd
  } else if(solver == 'nls.lm'){
    sol <- nls.lm(par = opt_par, fn = owps_wrap, control = list(nprint = 0, maxiter = maxiter, maxfev = 10000))
    par <- sol$par
    message <- sol$info
  }
  
  #### REARRANGE RESULTS
  if(silent == F) cat('Storing results...\n')
  fit = list()
  
  # Fit message
  fit$convergence <- message
  
  # intercept
  # intercepts
  intercepts <- par[1:length(lambda0_init)]
  fit$intercepts<- list()
  s <- 1
  for(i in 1:p){
    fit$intercepts[[i]] <- intercepts[s:(s + categories[i] - 2)]
    s <- s + categories[i] -1
  }
  
  # loadings
  lambda <- par[(length(lambda0_init)+1):(length(lambda0_init)+length(lambda_init))]
  fit$loadings <- constraintMat
  s = 1
  for(h in 1:q){
    for(j in 1:p){
      if(fit$loadings[j, h] != 0.0)
      {
        fit$loadings[j, h] = lambda[s]
        s = s+1
      }
    }
  }
  
  
  # correlations
  trhos <- par[(length(lambda0_init)+length(lambda_init)+1):length(par)]
  rhos <- (exp(2*trhos)-1)/(exp(2*trhos)+1)
  
  fit$factor_cov <- matrix(1, q, q)
  s = 1
  for( h in 1:q){
    for(j in 1:q){
      if(j > h)
      {
        fit$factor_cov[j, h] = rhos[s]
        fit$factor_cov[h, j] = rhos[s]
        s = s + 1
      }
    }
  }
  
  # nll
  fit$nll <- NULL
  
  
  # time
  end_time <- Sys.time()
  fit$time <- difftime(end_time, start_time, units = ("secs"))[[1]]
  if(silent == F) cat('Completed!\n')
  
  return(fit)
  
  
}
pmvnormM <- function(lower, upper, sigma, seed = 123){
  require(mvtnorm)
  set.seed(seed)
  out <- pmvnorm(lower = lower, upper = upper, sigma = sigma, keepAttr = F)
  return(out)
}