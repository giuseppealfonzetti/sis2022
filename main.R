rm(list = ls())
library(tidyverse)
library(Rcpp)
library(RcppEigen)
library(RcppThread)
library(pbmcapply)
library(pbv)
library(nleqslv)
library(mirt)
source('R/fit_pl.R')
source('R/fit_owpl.R')
sourceCpp('src/weightsFuns.cpp')
nseed <- 123
set.seed(nseed)

##### functions for model estimation
fit_mirtEM <- function(manifest, constraintMat, corrFLAG =0, met = 'EM', itemType = NULL, true_load){
  start_time <- Sys.time()
  p <- nrow(constraintMat); q <- ncol(constraintMat)
  # build mirt model from constraint matrix and corrFLAG
  mirt_mod <- ''
  for (lat in 1:ncol(constraintMat)) {
    plusFLAG <- 0
    mirt_mod <- paste0(mirt_mod, 'F', lat, ' = ')
    for (item in 1:nrow(constraintMat)) {
      if(constraintMat[item,lat]!=0){
        if(plusFLAG == 1){
          mirt_mod <- paste0(mirt_mod, ',', item)
        }else{
          mirt_mod <- paste0(mirt_mod, item)
        }
        plusFLAG <- 1
      }
    }
    mirt_mod <- paste0(mirt_mod, '\n   ')
  }
  if(corrFLAG==1){
    mirt_mod <- paste0(mirt_mod, 'COV = F1')
    for (lat in 2:ncol(constraintMat)) {
      mirt_mod <- paste0(mirt_mod, '*F', lat)
    }
  }
  
  # fit
  fit.mirt <- mirt(as.data.frame(manifest), mirt_mod, method = met, verbose = F)
  
  # store
  loadingsMIRT <- extract.mirt(fit.mirt, 'F')
  thresholdsMIRT <- as.matrix(as_tibble(coef(fit.mirt, simplify = T)$items) %>% select(starts_with('d')) )
  thresholdsMIRT <- thresholdsMIRT[, ncol(thresholdsMIRT):1]
  thresholdsMIRT <- split(thresholdsMIRT, row(thresholdsMIRT)[,1])
  names(thresholdsMIRT) <- NULL
  latent_covMIRT <- coef(fit.mirt, simplify = T)$cov
  
  # check sign switching
  for (lat in 1:ncol(constraintMat)) {
    if(sum(sign(loadingsMIRT[,lat])==-sign(true_load[,lat]))==p){
      loadingsMIRT[,lat] <- -loadingsMIRT[,lat]
      latent_covMIRT[, lat] <- -latent_covMIRT[, lat]
      latent_covMIRT[lat, ] <- -latent_covMIRT[lat, ]
    }
  }

  end_time <- Sys.time()
  
  # get complete parameter vector
  theta_vector <-  c(
    unlist(thresholdsMIRT), 
    unlist(loadingsMIRT[constraintMat!=0]), 
    unlist(latent_covMIRT[latent_covMIRT!=1])
    )
  
  
  # return output
  out <- list(
    theta = theta_vector,
    time = difftime(end_time, start_time, units = ("secs"))[[1]]
  )
  
  return(out)
}
URV_to_IRT <- function(thresholds, loadings, latent_cov){
  p <- nrow(loadings); q <- ncol(loadings)
  loadingsIRT <- loadings
  thresholdsIRT <- thresholds
  ErrSd <- sqrt(diag(diag(1, p, p) - loadings%*%latent_cov%*%t(loadings)))
  
  for (j in 1:p) {
    loadingsIRT[j,] <- loadings[j,]/ErrSd[j]
    thresholdsIRT[[j]] <- thresholds[[j]]/ErrSd[j]
  }
  
  out <- list(
    loadings = loadingsIRT,
    thresholds = thresholdsIRT,
    latent_cov = latent_cov
  )
  
  return(out)
}
fit_pl_IRT <- function(manifest, constraintMat, corrFLAG = 0, true_load){
  
  start_time <- Sys.time()
  p <- nrow(constraintMat);
  
  # URV estimation
  fit_PL <- fit_pl(  manifest,        
                     constraintMat,   
                     corrFLAG = corrFLAG,  
                     silent = T,    
                     ncores = 1,
                     method = 'nlminb',
                     maxiter = 1000,
                     init = 'none',
                     control =list())
  
  # Transform to IRT
  PL_IRT <- URV_to_IRT(fit_PL$intercepts, fit_PL$loadings, fit_PL$factor_cov)
  
  # check sign switching
  for (lat in 1:ncol(constraintMat)) {
    if(sum(sign(PL_IRT$loadings[,lat])==-sign(true_load[,lat]))==p){
      PL_IRT$loadings[,lat] <- -PL_IRT$loadings[,lat]
      PL_IRT$latent_cov[, lat] <- -PL_IRT$latent_cov[, lat]
      PL_IRT$latent_cov[lat, ] <- -PL_IRT$latent_cov[lat, ]
    }
  }
  
  # get complete parameter vector
  theta_vector <- c(
    unlist(PL_IRT$thresholds)*1.7, 
    unlist(PL_IRT$loadings[constraintMat!=0])*1.7, 
    unlist(PL_IRT$latent_cov[PL_IRT$latent_cov!=1])
    )
  
  end_time <- Sys.time()
  
  # return output
  out <- list(
    theta = theta_vector,
    time = difftime(end_time, start_time, units = ("secs"))[[1]]
  )
  
  return(out)
}
fit_owpl_IRT <- function(manifest, constraintMat, corrFLAG = 0, true_load){
  
  start_time <- Sys.time()
  p <- nrow(constraintMat)
  
  # URV estimation
  fit_ow <- fit_owpl(
    manifest, 
    constraintMat, 
    corrFLAG = corrFLAG,    
    silent = T,      
    ncores = 1,
    solver = 'nleqslv',
    maxiter = 1000,
    init.method = 'nlminb'
  )
  
  # transform to IRT
  OWPL_IRT <- URV_to_IRT(fit_ow$intercepts, fit_ow$loadings, fit_ow$factor_cov)
  
  # check sign switching
  for (lat in 1:ncol(constraintMat)) {
    if(sum(sign(OWPL_IRT$loadings[,lat])==-sign(true_load[,lat]))==p){
      OWPL_IRT$loadings[,lat] <- -OWPL_IRT$loadings[,lat]
      OWPL_IRT$latent_cov[, lat] <- -OWPL_IRT$latent_cov[, lat]
      OWPL_IRT$latent_cov[lat, ] <- -OWPL_IRT$latent_cov[lat, ]
    }
  }
  
  # get complete parameter vector
  theta_vector <- c(
    unlist(OWPL_IRT$thresholds)*1.7, 
    unlist(OWPL_IRT$loadings[constraintMat!=0])*1.7, 
    unlist(OWPL_IRT$latent_cov[OWPL_IRT$latent_cov!=1])
  )
  
  end_time <- Sys.time()
  
  # return output
  out <- list(
    theta = theta_vector,
    time = difftime(end_time, start_time, units = ("secs"))[[1]]
  )
  
  return(out)
}

##### simulated setting ######
load('data/sim_data_SIS2022.RData')
nthr <- length(true_tau)*nrow(constrMat); nload <- sum(constrMat); ncorr <- ncol(constrMat)*( ncol(constrMat)-1); d <- nthr + nload + ncorr

##### Estiamtion #####
setup <- simulated_data %>% 
  expand_grid(mod = c('ML', 'PL', 'OWPL')) %>% 
  mutate(
    n_units = map_dbl(manifest, ~nrow(.x))
    )

results <- setup 

est <- pbmclapply(setup %>% purrr::transpose(), function(x){
                          switch(x$mod, 
                                 'ML' = fit_mirtEM(x$manifest, constrMat, corrFLAG = 1, true_load = true_load),
                                 'PL'= fit_pl_IRT(x$manifest, constraintMat = constrMat, corrFLAG = 1, true_load = true_load),
                                 'OWPL' = fit_owpl_IRT(x$manifest, constraintMat = constrMat, corrFLAG = 1, true_load = true_load)
                          )
                          }, mc.cores = 8
)

results$mod_obj <- est
perf <- results %>%
  mutate(
    par = map(mod_obj, ~.x$theta),
    time = map_dbl(mod_obj, ~.x$time)
  ) %>% 
  mutate(
    theta = map_dbl(par, ~mean((.x-true_theta)^2, na.rm = T )),
    thresholds = map_dbl(par, ~mean((.x[1:nthr]-true_theta[1:nthr])^2, na.rm = T)),
    loadings = map_dbl(par, ~mean((.x[(nthr+1):(nthr+nload)]-true_theta[(nthr+1):(nthr+nload)])^2, na.rm = T)),
    correlations = map_dbl(par, ~mean((.x[(nthr+nload+1):d]-true_theta[(nthr+nload+1):d])^2, na.rm = T))
  ) %>% 
  gather(key = 'par_type', val = 'mse', theta, thresholds, loadings, correlations) %>% 
  left_join(
    results %>% 
      mutate(
        par = map(mod_obj, ~.x$theta),
        time = map_dbl(mod_obj, ~.x$time)
      ) %>% 
      mutate(
        theta = map_dbl(par, ~mean(abs(.x-true_theta), na.rm = T)),
        thresholds = map_dbl(par, ~mean(abs(.x[1:nthr]-true_theta[1:nthr]), na.rm = T)),
        loadings = map_dbl(par, ~mean(abs(.x[(nthr+1):(nthr+nload)]-true_theta[(nthr+1):(nthr+nload)]), na.rm = T)),
        correlations = map_dbl(par, ~mean(abs(.x[(nthr+nload+1):d]-true_theta[(nthr+nload+1):d]), na.rm = T))
      ) %>% 
      gather(key = 'par_type', val = 'mab', theta, thresholds, loadings, correlations),
    by = c('id', 'n_units', 'manifest', 'mod', 'mod_obj', 'par', 'par_type', 'time')
  )
perf

mse_box <- perf %>% 
  ggplot(aes(x = mod, y = mse)) +
  geom_boxplot() +
  geom_point(alpha = .5) +
  facet_grid(rows = vars(par_type), cols = vars(n_units), scales = 'free')

tab <- perf %>% 
  ungroup() %>% 
  select(id, n_units, mod, time, par_type, mse, mab) %>% 
  mutate(par_type = factor(par_type, levels = c('thresholds', 'loadings', 'correlations', 'theta'))) %>% 
  group_by(n_units, mod, par_type) %>% 
  summarise(
    mse = median(mse),
    mab = median(mab)
  ) %>% 
  pivot_wider(
    names_from = c(mod),
    values_from = c(mse, mab),
    names_glue = '{mod}_{.value}'
  ) %>% 
  arrange(n_units, par_type) %>% 
  print(n = 100)


Latex_tab <- tab %>% 
  mutate(
    par_type = factor(par_type, levels = c('correlations', 'loadings','thresholds','theta'), labels = c('$ \\rho $', '$ B $', '$ \\alpha $', '$\\theta$'))
  ) %>%
  select(n = n_units, parameter = par_type, starts_with('ML'), starts_with('PL'), starts_with('OWPL')) %>% 
  arrange(n, parameter)

model_header <- '&& \\multicolumn{3}{c|}{ML}& \\multicolumn{3}{c|}{PL}& \\multicolumn{3}{c}{OWPL}\\\\'
addToRow <- list(pos = list(-1), command = model_header)
names(Latex_tab) <- c('n', 'parameter', rep(c('mse', 'mab'), 3))
Latex_tab %>%
  xtable::xtable(digits = c(0,0, 0,rep(4, 6))) %>%
  print(include.rownames = FALSE,
        add.to.row = addToRow,
        booktabs =T,
        hline.after = F,
        sanitize.text.function=function(x){x})







