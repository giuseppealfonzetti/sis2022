#define EIGEN_PERMANENTLY_DISABLE_STUPID_WARNINGS
#include <Rcpp.h>
#include <RcppEigen.h>
#include <RcppThread.h>
#include <pbv.h>
#include <random>

#include "bivariateNormal.h"
#include "utils.h"
#include "pi.h"
#include "pair.h"

// [[Rcpp::depends(RcppEigen)]]
// [[Rcpp::depends(RcppThread)]]
// [[Rcpp::depends(pbv)]]



/* COMPUTE FREQUENCIES */
// It returns a 5-rows matrix with each combination of items and categories as columns.
// Row0: item k, Row1: item l, Row2; category item k, Row3: category item l, Row4: freq
// It is computed just once, before the optimization
// [[Rcpp::export]]
Eigen::MatrixXd pairs_freq(Eigen::Map<Eigen::MatrixXd> y,
                           Eigen::Map<Eigen::VectorXd> c_vec,
                           int ncores){
  
  int n = y.rows(); // number of units
  int p = y.cols(); // number of items
  
  int R = p*(p-1)/2; // number of pairs
  int c = c_vec.sum();
  int iter; // helper iterator
  
  Eigen::MatrixXd freq(5,1);
  
  // Serial loop:
  // Find how many possible pairs and setup freq matrix
  iter = 0;
  for(int k = 1; k < p; k++){
    int ck = c_vec(k);
    for(int l = 0; l < k; l ++){
      int cl = c_vec(l);
      for(int sk = 0; sk < ck; sk++){
        for(int sl = 0; sl < cl; sl ++){
          freq.conservativeResize(5, iter + 1);
          Eigen::VectorXd freq_coord(5);
          freq_coord << k, l, sk, sl, 0;
          freq.col(iter) = freq_coord;
          iter++; 
        }
      }
    }
  }
  
  // Spawning threads
  RcppThread::ThreadPool pool(ncores);
  
  auto parallel_task = [&freq, &n, &y](unsigned int r){
    
    // read data
    int k  = freq(0,r);
    int l  = freq(1,r);
    int sk = freq(2,r);
    int sl = freq(3,r);
    Eigen::MatrixXd obs_resp(n,2);
    obs_resp.col(0) = y.col(k); obs_resp.col(1) = y.col(l);
    //if(silentFLAG == 0)Rcpp::Rcout << "k: " << k << " , l: "<< l << " ,sk: " << sk << " , sl: " << sl << "\n";
    //if(silentFLAG == 0)Rcpp::Rcout << "obs responses : \n" << obs_resp << "\n";
    // compute frequency
    int n_sksl = 0;
    for(int i = 0; i < n; i++){
      if(sk == obs_resp(i,0) & sl == obs_resp(i,1)) {
        //if(silentFLAG == 0)Rcpp::Rcout << "Match on unit " << i <<"!\n";
        n_sksl++;
      }
    }
    
    // update
    freq(4, r) = n_sksl;
  };
  
  // Parallel loop to compute frequencies
  for(int r = 0; r < freq.cols(); r++){
    pool.push(parallel_task, r);
  }
  pool.join();
  
  
  return freq;
}

double compute_EIskslIbrbt(
    Rcpp::Function pmvnorm,
    const unsigned int k,
    const unsigned int l,
    const unsigned int r,
    const unsigned int t,
    const unsigned int sk,
    const unsigned int sl,
    const unsigned int br,
    const unsigned int bt,
    const double pi_sksl,
    const double rho_kl,
    const double rho_rt,
    const double rho_kt,
    const double rho_lt,
    const double rho_kr,
    const double rho_lr,
    const Eigen::VectorXd pisksl_thresholds,
    const Eigen::VectorXd pibrbt_thresholds
){
  
  // read pi related thresholds
  const double t_sk = pisksl_thresholds(0); 
  const double t_sl = pisksl_thresholds(1);                                               
  const double t_sk_prev = pisksl_thresholds(2);                                             
  const double t_sl_prev = pisksl_thresholds(3);
  const double t_br = pibrbt_thresholds(0); 
  const double t_bt = pibrbt_thresholds(1);                                               
  const double t_br_prev = pibrbt_thresholds(2);                                             
  const double t_bt_prev = pibrbt_thresholds(3);
  
  // compute E[Isksl * Ibrbt]
  double EIskslIbrbt = 0;
  if(k == r & l == t){
    // 2 items shared
    if(sk == br & sl == bt){
      EIskslIbrbt = pi_sksl;
      //Rcpp::Rcout << ". 2 items, EIskslIbrbt:"<<EIskslIbrbt ;
    } 
  } else if(r == k | r == l){
    
    // r-th item shared
    if( r == k & br == sk){
      // -> pi_skslbt
      Eigen::VectorXd upper(3); upper << t_sk, t_sl, t_bt;
      Eigen::VectorXd lower(3); lower << t_sk_prev, t_sl_prev, t_bt_prev;
      Eigen::MatrixXd sig = Eigen::MatrixXd::Identity(3,3);
      sig(0,1) = rho_kl; sig(1,0) = rho_kl;
      sig(0,2) = rho_kt; sig(2,0) = rho_kt;
      sig(1,2) = rho_lt; sig(2,1) = rho_lt;
      EIskslIbrbt = Rcpp::as<double>(pmvnorm(Rcpp::Named("lower")=lower, Rcpp::Named("upper")=upper, Rcpp::Named("sigma")=sig));
      
      //EIskslIbrbt = 0;
    } else if(r == l & br == sl){
      // -> pi_skslbt
      
      Eigen::VectorXd upper(3); upper << t_sk, t_sl, t_bt;
      Eigen::VectorXd lower(3); lower << t_sk_prev, t_sl_prev, t_bt_prev;
      Eigen::MatrixXd sig = Eigen::MatrixXd::Identity(3,3);
      sig(0,1) = rho_kl; sig(1,0) = rho_kl;
      sig(0,2) = rho_kt; sig(2,0) = rho_kt;
      sig(1,2) = rho_lt; sig(2,1) = rho_lt;
      EIskslIbrbt = Rcpp::as<double>(pmvnorm(Rcpp::Named("lower")=lower, Rcpp::Named("upper")=upper, Rcpp::Named("sigma")=sig));
      
    }
  } else if(t == k | t == l){
    // t-th item shared
    if(t == k & bt == sk){
      //-> pi_skslbr
      Eigen::VectorXd upper(3); upper << t_sk, t_sl, t_br;
      Eigen::VectorXd lower(3); lower << t_sk_prev, t_sl_prev, t_br_prev;
      Eigen::MatrixXd sig = Eigen::MatrixXd::Identity(3,3);
      sig(0,1) = rho_kl; sig(1,0) = rho_kl;
      sig(0,2) = rho_kr; sig(2,0) = rho_kr;
      sig(1,2) = rho_lr; sig(2,1) = rho_lr;
      EIskslIbrbt = Rcpp::as<double>(pmvnorm(Rcpp::Named("lower")=lower, Rcpp::Named("upper")=upper, Rcpp::Named("sigma")=sig));
      
    }else if(t == l & bt == sl){
      //-> pi_skslbr
      Eigen::VectorXd upper(3); upper << t_sk, t_sl, t_br;
      Eigen::VectorXd lower(3); lower << t_sk_prev, t_sl_prev, t_br_prev;
      Eigen::MatrixXd sig = Eigen::MatrixXd::Identity(3,3);
      sig(0,1) = rho_kl; sig(1,0) = rho_kl;
      sig(0,2) = rho_kr; sig(2,0) = rho_kr;
      sig(1,2) = rho_lr; sig(2,1) = rho_lr;
      EIskslIbrbt = Rcpp::as<double>(pmvnorm(Rcpp::Named("lower")=lower, Rcpp::Named("upper")=upper, Rcpp::Named("sigma")=sig));
      
    }
  } else {
    // no item shared
    // -> pi_skslbrbt
    
    Eigen::VectorXd upper(4); upper << t_sk, t_sl, t_br, t_bt;
    Eigen::VectorXd lower(4); lower << t_sk_prev, t_sl_prev, t_br_prev, t_bt_prev;
    Eigen::MatrixXd sig = Eigen::MatrixXd::Identity(4, 4);
    sig(0,1) = rho_kl; sig(1,0) = rho_kl;
    sig(0,2) = rho_kr; sig(2,0) = rho_kr;
    sig(0,3) = rho_kt; sig(3,0) = rho_kt;
    sig(1,2) = rho_lr; sig(2,1) = rho_lr;
    sig(1,3) = rho_lt; sig(3,1) = rho_lt;
    sig(2,3) = rho_rt; sig(3,2) = rho_rt;
    
    EIskslIbrbt = Rcpp::as<double>(pmvnorm(Rcpp::Named("lower")=lower, Rcpp::Named("upper")=upper, Rcpp::Named("sigma")=sig));
    
  }
  return EIskslIbrbt;
}


/* LIKELIHOOD AND GRADIENT */
// Not parallelized yet
// [[Rcpp::export]]
Rcpp::List pairwise_gllvm(Eigen::Map<Eigen::MatrixXd> y,
                          Eigen::Map<Eigen::VectorXd> c_vec,
                          Eigen::Map<Eigen::MatrixXd> A,
                          Eigen::Map<Eigen::MatrixXd> freq,
                          
                          Eigen::Map<Eigen::VectorXd> tau,
                          Eigen::Map<Eigen::VectorXd> lambda,
                          Eigen::Map<Eigen::VectorXd> transformed_rhos,
                          Eigen::Map<Eigen::VectorXd> weights,
                          
                          int corrFLAG,
                          int grFLAG,
                          int ncores,
                          int silentFLAG
                            
){
  if(silentFLAG == 0)if(silentFLAG == 0)Rcpp::Rcout << "Started!\n";
  Eigen::VectorXd rhos = ((Eigen::VectorXd(2*transformed_rhos)).array().exp() - 1)/
    ((Eigen::VectorXd(2*transformed_rhos)).array().exp() + 1);  // reparametrize rhos
  int n = y.rows(); // number of units
  int p = A.rows(); // number of items
  int q = A.cols(); // number of latents
  int d = tau.size() + lambda.size() + transformed_rhos.size(); // number of parameters
  int R = p*(p-1)/2; // number of pairs of items
  int RR = freq.cols(); // number of total pairs of categories 
  int c = c_vec.sum();
  int iter; // helper iterator
  
  // copy and resize the freq matrix.
  // New row will be used to store probabilities
  Eigen::MatrixXd pairs_tab = freq; 
  pairs_tab.conservativeResize(freq.rows() + 1, Eigen::NoChange_t() );
  
  // Build the loadings matrix setting zero-constraints
  Eigen::MatrixXd Lam = A;
  iter = 0; 
  for(int h = 0; h < q; h++){
    for(int j = 0; j < p; j++){
      if(A(j, h) != 0.0)
      {
        Lam(j, h) = lambda(iter) ;
        iter ++;
      };
    };
  }
  
  // latent variable covariance matrix
  Eigen::MatrixXd Sigma_u(q,q); Sigma_u.setIdentity();
  if(corrFLAG == 1){
    iter = 0; 
    for(int j = 0; j < q; j++){
      for(int h = 0; h < q; h++){
        if(j > h)
        {
          Sigma_u(j, h) = rhos(iter);
          Sigma_u(h, j) = rhos(iter);
          iter ++;
        }
      }
    }
  }
  
  
  
  
  ///////////////////////////
  /* LIKELIHOOD COMPUTATION */
  //////////////////////////
  if(silentFLAG == 0)if(silentFLAG == 0)Rcpp::Rcout << "Computing likelihood... ";
  double nll = 0;
  
  unsigned int iter_pair_kl = 0;
  for(int k = 1; k < p; k++){
    int ck = c_vec(k);
    Eigen::VectorXd lambdak = Lam.row(k);
    
    // identify column index in freq table
    // i1: starting index item k
    int i1 = 0; 
    if(k > 1){
      for(int u = 1; u < k; u++){
        int cu = c_vec(u);
        //if(silentFLAG == 0)Rcpp::Rcout << "u: " << u << ", cu: "<< cu << "\n";
        i1 += cu * c_vec.segment(0,u).sum();
      }
    }
    
    for(int l = 0; l < k; l++){
      
      int cl = c_vec(l);
      Eigen::VectorXd lambdal = Lam.row(l);
      double rho_kl = lambdak.transpose() * Sigma_u * lambdal;
      double nllr = 0;
      
      Eigen::MatrixXd Sigma_kl(2,2); Sigma_kl.setIdentity(); 
      Sigma_kl(1,0) = rho_kl; Sigma_kl(0,1) = rho_kl;
      
      // i2 starting index from i1 dor item l
      int i2 = 0;
      if(l > 0){
        i2 = c_vec.segment(0,l).sum() * c_vec(k);
      }
      //Rcpp::Rcout << "\nPair ("<< k <<","<< l<<"), corr:" << rho_kl;
      
      for(int sk = 0; sk < ck; sk ++){
        
        // i3: starting index from i2 for cat sk
        int i3 = sk * cl;
        
        for(int sl = 0; sl < cl; sl ++){
          //Rcpp::Rcout << "\nsk:"<< sk <<", sl:"<< sl<<"";
          
          
          // final column index for pairs_tab. Print to check 
          int r = i1 + i2 + i3 + sl;
          //if(silentFLAG == 0)Rcpp::Rcout << "r: " << r << "\n";
          int n_sksl = pairs_tab(4, r);
          
          int sk_tau_index = c_vec.segment(0, k).sum() - (k) + sk;   // index sk in tau vector
          int sl_tau_index = c_vec.segment(0, l).sum() - (l) + sl;   // index sl in tau vector
          
          double t_sk;                                               // tau_sk
          if(sk == ck-1){
            t_sk = 1000; 
          } else {
            t_sk = tau(sk_tau_index);
          }                            
          double t_sk_prev;                                           // tau_sk-1  
          if(sk == 0){
            t_sk_prev = -1000;
          }else{
            t_sk_prev = tau(sk_tau_index-1);
          }
          
          double t_sl;                                               // tau_sl
          if(sl == cl-1){
            t_sl = 1000; 
          } else {
            t_sl = tau(sl_tau_index);
          }                            
          double t_sl_prev;                                           // tau_sl-1  
          if(sl == 0){
            t_sl_prev = -1000;
          }else{
            t_sl_prev = tau(sl_tau_index-1);
          }          
          
          // compute likelihood
          //if(silentFLAG == 0)Rcpp::Rcout << "t_sk: " << t_sk << " , t_sl: "<< t_sl <<"\n";
          
          // Phi(t_sk, t_sl; rho_kl)
          double cum1;
          if(sk == (ck-1) & sl == (cl-1)){
            cum1 = 1;
          } else if(sk == (ck-1)){
            cum1 = R::pnorm(t_sl, 0, 1, 1, 0);
          } else if(sl == (cl-1)){
            cum1 = R::pnorm(t_sk, 0, 1, 1, 0);
          } else {
            cum1 = pbv::pbv_rcpp_pbvnorm0( t_sk, t_sl, rho_kl);
          }
          // Phi(t_sk, t_sl-1; rho_kl)
          double cum2;
          if(sl == 0){
            cum2 = 0;
          } else {
            Eigen::VectorXd upper(2); upper << t_sk,t_sl_prev;
            cum2 = pbv::pbv_rcpp_pbvnorm0( t_sk, t_sl_prev, rho_kl);
          }
          // Phi(t_sk-1, t_sl; rho_kl)
          double cum3;
          if(sk == 0){
            cum3 = 0;
          } else{
            Eigen::VectorXd upper(2); upper << t_sk_prev,t_sl;
            cum3 = pbv::pbv_rcpp_pbvnorm0( t_sk_prev, t_sl, rho_kl);
          }
          // Phi(t_sk-1, t_sl-1; rho_kl)
          double cum4;
          if(sl == 0 | sk == 0){
            cum4 = 0;
          }else{
            Eigen::VectorXd upper(2); upper << t_sk_prev,t_sl_prev;
            cum4 = pbv::pbv_rcpp_pbvnorm0( t_sk_prev, t_sl_prev, rho_kl);
          }
          //Rcpp::Rcout << "\nc1:"<< cum1 <<", c2:"<< cum2<<", c3:"<<cum3<<", c4:"<< cum4;
          
          double pi_sksl = cum1 - cum2 - cum3 + cum4;
          pairs_tab(5,r) = pi_sksl;
          nllr -= n_sksl * log(pi_sksl+1e-8);
          
          
        }
      }
      if(silentFLAG == 0)Rcpp::Rcout << "\nnll Pair ("<< k <<","<< l<<")="<<nllr;
      nll += weights(iter_pair_kl)*nllr; iter_pair_kl++;
      //Rcpp::Rcout << "\nnll Pair ("<< k <<","<< l<<")="<<nllr;
    }
  }
  if(silentFLAG == 0)Rcpp::Rcout << "Done!\n";
  
  //////////////////////////
  /* GRADIENT COMPUTATION */
  /////////////////////////
  if(silentFLAG == 0)Rcpp::Rcout << "Computing gradient...\n";
  Eigen::VectorXd gradient(d); gradient.fill(0.0);
  
  if(grFLAG == 1 ){
    iter_pair_kl = 0;
    for(int k = 1; k < p; k++){
      int ck = c_vec(k);
      Eigen::VectorXd lambdak = Lam.row(k);
      
      // identify column index in freq table
      // i1: starting index item k
      int i1 = 0; 
      if(k > 1){
        for(int u = 1; u < k; u++){
          int cu = c_vec(u);
          i1 += cu * c_vec.segment(0,u).sum();
        }
      }
      
      for(int l = 0; l < k; l++){
        if(silentFLAG == 0)Rcpp::Rcout << "\nPair (" << k << "," << l << "): \n";
        
        int cl = c_vec(l);
        Eigen::VectorXd lambdal = Lam.row(l);
        double rho_kl = lambdak.transpose() * Sigma_u * lambdal;
        double nllr = 0;
        
        // i2 starting index from i1 dor item l
        int i2 = 0;
        if(l > 0){
          i2 = c_vec.segment(0,l).sum() * c_vec(k);
        }
        
        int iter = 0;
        Eigen::VectorXd gradientr(d); gradientr.fill(0.0);
        
        /////////////////////////////////////////////////////
        // (k,l)-pair likelihood derivative wrt thresholds //
        /////////////////////////////////////////////////////
        if(silentFLAG == 0)Rcpp::Rcout << "- Gradient wrt thresholds: \n";
        
        // loop: terate over elements of thresholds vector
        for(int s = 0; s < tau.size(); s++){
          double grs = 0; // temporary location for gradient related to s-th element of tau
          if(silentFLAG == 0)Rcpp::Rcout << "  |_ gradient("<< s<< ")\n";
          
          // Elicit three cases: 1. threshold related to item k, 2. threshold related to item l, 3. threshold non relevant to items couple (k,l)
          if(s >= (c_vec.segment(0, k).sum()) - (k) & s < c_vec.segment(0, k + 1).sum() - (k + 1)){
            // [CASE 1]: threshold related to item k
            
            if(silentFLAG == 0)Rcpp::Rcout << "  |    |_ tau item k:\n";
            int sk = s - (c_vec.segment(0, k).sum()) + (k);
            
            // i3: starting index from i2 for cat sk and sk+1
            int i3 = sk * cl;
            int i3suc = (sk+1) * cl; 
            if(silentFLAG == 0)Rcpp::Rcout << "  |    |_ sk: " << sk << ". Summing over categories item l: ";
            
            // iterate over categories of item l
            for(int sl = 0; sl < cl; sl ++){
              if(silentFLAG == 0)Rcpp::Rcout << " ... cat" << sl ;
              
              // identify pairs_tab column for (sk,sl) and (sk+1, sl)
              int r = i1 + i2 + i3 + sl; 
              int rsuc = i1 + i2 + i3suc + sl;
              
              // read frequences
              int n_sksl = pairs_tab(4, r);
              int n_sksucsl = pairs_tab(4, rsuc);
              
              // read probabilities
              double pi_sksl = pairs_tab(5, r);
              double pi_sksucsl = pairs_tab(5, rsuc);
              
              // identify tau_sk, tau_sl, tau_sl-1
              int sk_tau_index = c_vec.segment(0, k).sum() - (k) + sk;   // index tau_sk in tau vector
              int sl_tau_index = c_vec.segment(0, l).sum() - (l) + sl;   // index tau_sl in tau vector
              double t_sk;                                               // tau_sk
              if(sk == ck-1){
                t_sk = 100; 
              } else {
                t_sk = tau(sk_tau_index);
              }                            
              
              double t_sl;                                               // tau_sl
              if(sl == cl-1){
                t_sl = 100; 
              } else {
                t_sl = tau(sl_tau_index);
              }                            
              double t_sl_prev;                                           // tau_sl-1  
              if(sl == 0){
                t_sl_prev = -100;
              }else{
                t_sl_prev = tau(sl_tau_index-1);
              }
              
              // compute gradient
              double tmp1 = ((n_sksl/(pi_sksl+1e-8))-(n_sksucsl/(pi_sksucsl+1e-8)));
              double tmp2 = R::dnorm(t_sk, 0, 1, 0);
              double tmp3 = R::pnorm((t_sl-rho_kl*t_sk)/(pow(1-pow(rho_kl,2), .5)), 0, 1, 1, 0);
              double tmp4 = R::pnorm((t_sl_prev-rho_kl*t_sk)/(pow(1-pow(rho_kl,2), .5)), 0, 1, 1, 0);
              grs += tmp1 * tmp2 * (tmp3 - tmp4);
            }
            if(silentFLAG == 0)Rcpp::Rcout << "\n";
            
          }else if(s >= (c_vec.segment(0, l).sum())-(l) & s<c_vec.segment(0, l + 1).sum()-(l + 1)){
            // [CASE 2]: threshold related to item l
            
            if(silentFLAG == 0)Rcpp::Rcout << "  |    |_ tau item l\n";
            int sl = s - (c_vec.segment(0, l).sum()) + (l);
            
            if(silentFLAG == 0)Rcpp::Rcout << "  |    |_  sl: " << sl << ". Summing over categories item k: ";
            
            // iterate over categories item k
            for(int sk = 0; sk < ck; sk ++){
              
              // i3: starting index from i2 for cat sk 
              int i3 = sk * cl;
              
              // identify pairs_tab column for (sk,sl) and (sk, sl + 1)
              int r = i1 + i2 + i3 + sl; 
              int rsuc = i1 + i2 + i3 + sl + 1;
              
              // read frequences
              int n_sksl = pairs_tab(4, r);
              int n_skslsuc = pairs_tab(4, rsuc);
              
              // read probabilities
              double pi_sksl = pairs_tab(5, r);
              double pi_skslsuc = pairs_tab(5, rsuc);
              
              // identify tau_sk, tau_sl, tau_sk-1
              int sk_tau_index = c_vec.segment(0, k).sum() - (k) + sk;   // index tau_sk in tau vector
              int sl_tau_index = c_vec.segment(0, l).sum() - (l) + sl;   // index tau_sl in tau vector
              
              double t_sk;                                               // tau_sk
              if(sk == ck-1){
                t_sk = 100; 
              } else {
                t_sk = tau(sk_tau_index);
              }                            
              double t_sk_prev;                                           // tau_sk-1  
              if(sk == 0){
                t_sk_prev = -100;
              }else{
                t_sk_prev = tau(sk_tau_index-1);
              }
              
              double t_sl;                                               // tau_sl
              if(sl == cl-1){
                t_sl = 100; 
              } else {
                t_sl = tau(sl_tau_index);
              }                            
              
              if(silentFLAG == 0)Rcpp::Rcout<<"\n  |    |   |_ sk:"<< sk << ", r: "<< r<<", n_sksl:"
                                            << n_sksl<< ", n_sksl+1:" << n_skslsuc << ", pi_sksl:" 
                                            << pi_sksl << ", pi_sksl+1:"<< pi_skslsuc << ", t_sk:"
                                            << t_sk<< ", t_sl:" << t_sl << "t_sk-1:"<< t_sk_prev;
              // compute gradient
              double tmp1 = ((n_sksl/(pi_sksl+1e-8))-(n_skslsuc/(pi_skslsuc+1e-8)));
              double tmp2 = R::dnorm(t_sl, 0, 1, 0);
              double tmp3 = R::pnorm((t_sk-rho_kl*t_sl)/(pow(1-pow(rho_kl,2), .5)), 0, 1, 1, 0);
              double tmp4 = R::pnorm((t_sk_prev-rho_kl*t_sl)/(pow(1-pow(rho_kl,2), .5)), 0, 1, 1, 0);
              if(silentFLAG == 0)Rcpp::Rcout<<" => out" << sk << ":" << tmp1 * tmp2 * (tmp3 - tmp4);
              grs += tmp1 * tmp2 * (tmp3 - tmp4);
            }
            if(silentFLAG == 0)Rcpp::Rcout << "\n";
            
          }else{
            if(silentFLAG == 0)Rcpp::Rcout << "  |    |_  tau of other item\n";
          }
          if(silentFLAG == 0)Rcpp::Rcout << "  |    |_  Thresholds:: " << iter<< "/"<< tau.size() 
                                         <<". Tot:: "<< iter << "/"<< d <<". Val ="<< grs <<"\n";
          gradientr(iter) += grs;
          iter ++;
        }
        if(silentFLAG == 0)Rcpp::Rcout << "  |_ Done. \n";
        
        ///////////////////////////////////////////////////////////
        // (k,l)-pair likelihood derivative wrt URV correlation: //
        // intermediate step for derivatives wrt                 //
        // loadings and factor correlations                      //
        ///////////////////////////////////////////////////////////
        if(silentFLAG == 0)Rcpp::Rcout << "\n- Intermediate derivative for loadings and correlation: \n";
        double tmp_kl = 0; // temporary location of the gradient
        
        // double loop: iterate over each combination of categories of items k and l
        for(int sk = 0; sk < ck; sk ++){
          for(int sl = 0; sl < cl; sl ++){
            if(silentFLAG == 0)Rcpp::Rcout << "  |_ sk: "<< sk << ", sl: " << sl << ": \n";
            
            // identify pairs_tab column for (sk,sl)
            int i3 = sk * cl;
            int r = i1 + i2 + i3 + sl; 
            
            // read freq
            int n_sksl = pairs_tab(4, r);
            
            // read prob
            double pi_sksl = pairs_tab(5, r);
            
            // identify tau_sk, tau_sl, tau_sk-1, tau_sl-1
            int sk_tau_index = c_vec.segment(0, k).sum() - (k) + sk;   // index tau_sk in tau vector
            int sl_tau_index = c_vec.segment(0, l).sum() - (l) + sl;   // index tau_sl in tau vector
            double t_sk;                                               // tau_sk
            if(sk == ck-1){
              t_sk = 100; 
            } else {
              t_sk = tau(sk_tau_index);
            }                            
            double t_sk_prev;                                           // tau_sk-1  
            if(sk == 0){
              t_sk_prev = -100;
            }else{
              t_sk_prev = tau(sk_tau_index-1);
            }
            
            double t_sl;                                               // tau_sl
            if(sl == cl-1){
              t_sl = 100; 
            } else {
              t_sl = tau(sl_tau_index);
            }                            
            double t_sl_prev;                                           // tau_sl-1  
            if(sl == 0){
              t_sl_prev = -100;
            }else{
              t_sl_prev = tau(sl_tau_index-1);
            }
            
            // phi(t_sk, t_sl; rho_kl)
            double d1 = pbv::pbv_rcpp_dbvnorm0( t_sk, t_sl, rho_kl, 0);                  
            
            // phi(t_sk, t_sl-1; rho_kl)
            double d2 = pbv::pbv_rcpp_dbvnorm0( t_sk, t_sl_prev, rho_kl, 0);
            
            // phi(t_sk-1, t_sl; rho_kl)
            double d3 = pbv::pbv_rcpp_dbvnorm0( t_sk_prev, t_sl, rho_kl, 0);
            
            // phi(t_sk-1, t_sl-1; rho_kl)
            double d4 = pbv::pbv_rcpp_dbvnorm0( t_sk_prev, t_sl_prev, rho_kl, 0);
            
            tmp_kl += (n_sksl/(pi_sksl+1e-8)) * ( d1 - d2 - d3 + d4);
          }
        }
        if(silentFLAG == 0)Rcpp::Rcout << "  |_ tmp_kl:" << tmp_kl << "\n";
        
        ///////////////////////////////////////////////////
        // (k,l)-pair likelihood derivative wrt loadings //
        ///////////////////////////////////////////////////
        if(silentFLAG == 0)Rcpp::Rcout << "\n- Gradient wrt loadings: \n";
        
        // double loop: iterate over elements of loadings matrix
        for(int j = 0; j < p; j++){
          for(int v = 0; v < q; v++){
            if(silentFLAG == 0)Rcpp::Rcout << "  |_ visiting lambda_"<< j << v <<":\n";
            
            // elicit three cases: 1. free loading item k, 2. free loading l, 3. other
            if(j == k){
              if(A(j,v)!=0 ){
                if(silentFLAG == 0)Rcpp::Rcout << "  |   |_ item k, free loading:\n";
                Eigen::VectorXd ev(q); ev.fill(0.0); ev(v) = 1;
                double d_rho_kl = ev.transpose() * Sigma_u * lambdal;
                if(silentFLAG == 0)Rcpp::Rcout << "  |   |_ d_rho_kl:" << d_rho_kl << "\n";
                gradientr(iter) += tmp_kl * d_rho_kl;
                if(silentFLAG == 0)Rcpp::Rcout << "  |   |_ Loadings:: " << iter - tau.size() << "/"<< lambda.size() <<". Tot:: "<< iter << "/"<< d << "\n";
                
                iter ++;
              }
            }else if (j == l){
              if(A(j,v)!=0 ){
                if(silentFLAG == 0)Rcpp::Rcout << "  |   |_ item l, free loading:\n";
                Eigen::VectorXd ev(q); ev.fill(0.0); ev(v) = 1;
                double d_rho_kl = lambdak.transpose() * Sigma_u * ev;
                if(silentFLAG == 0)Rcpp::Rcout << "  |   |_ d_rho_kl:" << d_rho_kl << "\n";
                if(silentFLAG == 0)Rcpp::Rcout << "  |   |_ Loadings:: " << iter - tau.size() << "/"<< lambda.size() <<". Tot:: "<< iter << "/"<< d << "\n";
                gradientr(iter) += tmp_kl * d_rho_kl;
                
                iter ++;
              }
            }else if(A(j,v)!=0){
              
              if(silentFLAG == 0)Rcpp::Rcout << "  |  |_ Loadings:: " << iter - tau.size() << "/"<< lambda.size() <<". Tot:: "<< iter << "/"<< d << " [not included]\n";
              iter ++;
            }
          }
        }
        if(silentFLAG == 0)Rcpp::Rcout << "  |_ Done. \n";
        
        //////////////////////////////////////////////////////////////
        // (k,l)-pair likelihood derivative wrt latent correlations //
        //////////////////////////////////////////////////////////////     
        if(silentFLAG == 0)Rcpp::Rcout << "\n- Gradient wrt correlations: \n";
        
        if(corrFLAG == 1){
          // double loop: iterate over each non-redundant latent correlation
          for(int v = 1; v < q; v++){
            for(int  t = 0; t < v; t++){
              Eigen::VectorXd ev(q); ev.fill(0.0); ev(v) = 1;
              Eigen::VectorXd et(q); et.fill(0.0); et(t) = 1;
              double trho = transformed_rhos(iter - tau.size() - lambda.size());
              double drho = 2*exp(2*trho) * pow((exp(2*trho) + 1),-1) * ( 1 - ( exp(2*trho) - 1) * pow((exp(2*trho) + 1),-1) );
              
              // impose symmetric structure
              Eigen::MatrixXd Jvt = ev * et.transpose();
              Eigen::MatrixXd Jtv = et * ev.transpose();
              Eigen::MatrixXd Svt = Jvt + Jtv - Jvt*Jvt;
              
              double d_rho_kl = lambdak.transpose() * (Svt * drho) * lambdal;
              
              //if(silentFLAG == 0) 
              gradientr(iter) += tmp_kl * d_rho_kl;
              iter ++;
            }
          }
        }
        
        if(silentFLAG == 0)Rcpp::Rcout << "\n=====> gradient r-th pair:\n" << gradientr << "\n";
        gradient -= weights(iter_pair_kl)*gradientr; iter_pair_kl ++;
        if(silentFLAG == 0)Rcpp::Rcout << "\n";
      }    
      
    }
    
  }
  
  
  // output list
  Rcpp::List output =
    Rcpp::List::create(
      Rcpp::Named("nll") = nll,
      //Rcpp::Named("freq") = freq,
      Rcpp::Named("pairs_tab") = pairs_tab,
      Rcpp::Named("gradient") = gradient
    );
  return(output);
}




/* OPTIMALLY WEIGHTED PAIRWISE SCORES */
// [[Rcpp::export]]
Rcpp::List owps_gllvm(Eigen::Map<Eigen::MatrixXd> y,
                      Eigen::Map<Eigen::VectorXd> c_vec,
                      Eigen::Map<Eigen::MatrixXd> A,
                      Eigen::Map<Eigen::MatrixXd> freq,
                      
                      Eigen::Map<Eigen::VectorXd> tau,
                      Eigen::Map<Eigen::VectorXd> lambda,
                      Eigen::Map<Eigen::VectorXd> transformed_rhos,
                      Eigen::Map<Eigen::MatrixXd> pre_weights,
                      
                      int weightsFLAG,
                      int corrFLAG,
                      int ncores,
                      int silentFLAG,
                      int item1 = 1,
                      int item2 = 0,
                      int cat1 = 0,
                      int cat2 = 0
                        
){
  if(silentFLAG == 0)if(silentFLAG == 0)Rcpp::Rcout << "Started!\n";
  Eigen::VectorXd rhos = ((Eigen::VectorXd(2*transformed_rhos)).array().exp() - 1)/
    ((Eigen::VectorXd(2*transformed_rhos)).array().exp() + 1);  // reparametrize rhos
  int n = y.rows(); // number of units
  int p = A.rows(); // number of items
  int q = A.cols(); // number of latents
  int d = tau.size() + lambda.size() + transformed_rhos.size(); // number of parameters
  int R = p*(p-1)/2; // number of pairs of items
  int RR = freq.cols(); // number of total pairs of categories 
  int c = c_vec.sum();
  int iter; // helper iterator
  
  // copy and resize the freq matrix.
  // New row will be used to store probabilities
  Eigen::MatrixXd pairs_tab = freq; 
  pairs_tab.conservativeResize(freq.rows() + 1, Eigen::NoChange_t() );
  
  // Build the loadings matrix setting zero-constraints
  Eigen::MatrixXd Lam = A;
  iter = 0; 
  for(int h = 0; h < q; h++){
    for(int j = 0; j < p; j++){
      if(A(j, h) != 0.0)
      {
        Lam(j, h) = lambda(iter) ;
        iter ++;
      };
    };
  }
  
  // latent variable covariance matrix
  Eigen::MatrixXd Sigma_u(q,q); Sigma_u.setIdentity();
  if(corrFLAG == 1){
    iter = 0; 
    for(int j = 0; j < q; j++){
      for(int h = 0; h < q; h++){
        if(j > h)
        {
          Sigma_u(j, h) = rhos(iter);
          Sigma_u(h, j) = rhos(iter);
          iter ++;
        }
      }
    }
  }
  
  Rcpp::Function pmvnorm("pmvnormM");
  
  
  ///////////////////////
  /* COMPUTING PI_SKSL */
  ///////////////////////
  if(silentFLAG == 0)Rcpp::Rcout << "Computing pis... ";
  double nll = 0;
  for(int k = 1; k < p; k++){
    int ck = c_vec(k);
    Eigen::VectorXd lambdak = Lam.row(k);
    
    // identify column index in freq table
    // i1: starting index item k
    int i1 = 0; 
    if(k > 1){
      for(int u = 1; u < k; u++){
        int cu = c_vec(u);
        //if(silentFLAG == 0)Rcpp::Rcout << "u: " << u << ", cu: "<< cu << "\n";
        i1 += cu * c_vec.segment(0,u).sum();
      }
    }
    
    for(int l = 0; l < k; l++){
      //if(k==3 & l == 0){silentFLAG=0;}else{silentFLAG=1;}
      int cl = c_vec(l);
      Eigen::VectorXd lambdal = Lam.row(l);
      double rho_kl = lambdak.transpose() * Sigma_u * lambdal;
      double nllr = 0;
      
      Eigen::MatrixXd Sigma_kl(2,2); Sigma_kl.setIdentity(); 
      Sigma_kl(1,0) = rho_kl; Sigma_kl(0,1) = rho_kl;
      
      // i2 starting index from i1 dor item l
      int i2 = 0;
      if(l > 0){
        i2 = c_vec.segment(0,l).sum() * c_vec(k);
      }
      if(silentFLAG == 0)Rcpp::Rcout << "\nPair ("<< k <<","<< l<<"), corr:" << rho_kl;
      
      for(int sk = 0; sk < ck; sk ++){
        
        // i3: starting index from i2 for cat sk
        int i3 = sk * cl;
        
        for(int sl = 0; sl < cl; sl ++){
          if(silentFLAG == 0)Rcpp::Rcout << "\nsk:"<< sk <<", sl:"<< sl<<"";
          
          
          // final column index for pairs_tab. Print to check 
          int r = i1 + i2 + i3 + sl;
          //if(silentFLAG == 0)Rcpp::Rcout << "r: " << r << "\n";
          int n_sksl = pairs_tab(4, r);
          
          int sk_tau_index = c_vec.segment(0, k).sum() - (k) + sk;   // index sk in tau vector
          int sl_tau_index = c_vec.segment(0, l).sum() - (l) + sl;   // index sl in tau vector
          
          double t_sk;                                               // tau_sk
          if(sk == ck-1){
            t_sk = 100; 
          } else {
            t_sk = tau(sk_tau_index);
          }                            
          double t_sk_prev;                                           // tau_sk-1  
          if(sk == 0){
            t_sk_prev = -100;
          }else{
            t_sk_prev = tau(sk_tau_index-1);
          }
          
          double t_sl;                                               // tau_sl
          if(sl == cl-1){
            t_sl = 100; 
          } else {
            t_sl = tau(sl_tau_index);
          }                            
          double t_sl_prev;                                           // tau_sl-1  
          if(sl == 0){
            t_sl_prev = -100;
          }else{
            t_sl_prev = tau(sl_tau_index-1);
          }          
          
          // compute likelihood
          //if(silentFLAG == 0)Rcpp::Rcout << "t_sk: " << t_sk << " , t_sl: "<< t_sl <<"\n";
          
          // Phi(t_sk, t_sl; rho_kl)
          double cum1;
          if(sk == (ck-1) & sl == (cl-1)){
            cum1 = 1;
          } else if(sk == (ck-1)){
            cum1 = R::pnorm(t_sl, 0, 1, 1, 0);
          } else if(sl == (cl-1)){
            cum1 = R::pnorm(t_sk, 0, 1, 1, 0);
          } else {
            cum1 = pbv::pbv_rcpp_pbvnorm0( t_sk, t_sl, rho_kl);
          }
          // Phi(t_sk, t_sl-1; rho_kl)
          double cum2;
          if(sl == 0){
            cum2 = 0;
          } else {
            if(silentFLAG == 0)Rcpp::Rcout << "t_sk: " << t_sk << " , t_sl_prev: "<< t_sl_prev <<", rho:"<< rho_kl<<"\n";
            cum2 = pbv::pbv_rcpp_pbvnorm0( t_sk, t_sl_prev, rho_kl);
          }
          // Phi(t_sk-1, t_sl; rho_kl)
          double cum3;
          if(sk == 0){
            cum3 = 0;
          } else{
            cum3 = pbv::pbv_rcpp_pbvnorm0( t_sk_prev, t_sl, rho_kl);
          }
          // Phi(t_sk-1, t_sl-1; rho_kl)
          double cum4;
          if(sl == 0 | sk == 0){
            cum4 = 0;
          }else{
            cum4 = pbv::pbv_rcpp_pbvnorm0( t_sk_prev, t_sl_prev, rho_kl);
          }
          if(silentFLAG == 0)Rcpp::Rcout << "\nc1:"<< cum1 <<", c2:"<< cum2<<", c3:"<<cum3<<", c4:"<< cum4;
          
          double pi_sksl = cum1 - cum2 - cum3 + cum4;
          pairs_tab(5,r) = pi_sksl;
          
          
        }
      }
    }
  }
  if(silentFLAG == 0)Rcpp::Rcout << "Done!\n";
  
  //////////////////////
  /* COMPUTING SCORES */
  //////////////////////
  if(silentFLAG == 0)Rcpp::Rcout << "Computing scores...\n";
  Eigen::VectorXd scores(d*p*(p-1)/2); scores.fill(0.0);
  
  int iter_pair_kl = 0;
  for(int k = 1; k < p; k++){
    int ck = c_vec(k);
    Eigen::VectorXd lambdak = Lam.row(k);
    
    // identify column index in freq table
    // i1: starting index item k
    int i1 = 0; 
    if(k > 1){
      for(int u = 1; u < k; u++){
        int cu = c_vec(u);
        i1 += cu * c_vec.segment(0,u).sum();
      }
    }
    
    for(int l = 0; l < k; l++){
      //if(k==3 & l == 0){silentFLAG=0;}else{silentFLAG=1;}
      if(silentFLAG == 0)Rcpp::Rcout << "\nPair (" << k << "," << l << "): \n";
      
      int cl = c_vec(l);
      Eigen::VectorXd lambdal = Lam.row(l);
      double rho_kl = lambdak.transpose() * Sigma_u * lambdal;
      double nllr = 0;
      
      // i2 starting index from i1 dor item l
      int i2 = 0;
      if(l > 0){
        i2 = c_vec.segment(0,l).sum() * c_vec(k);
      }
      
      int iter = 0;
      Eigen::VectorXd gradientr(d); gradientr.fill(0.0);
      
      /////////////////////////////////////////////////////
      // (k,l)-pair likelihood derivative wrt thresholds //
      /////////////////////////////////////////////////////
      if(silentFLAG == 0)Rcpp::Rcout << "- Gradient wrt thresholds: \n";
      
      // loop: terate over elements of thresholds vector
      for(int s = 0; s < tau.size(); s++){
        double grs = 0; // temporary location for gradient related to s-th element of tau
        if(silentFLAG == 0)Rcpp::Rcout << "  |_ gradient("<< s<< ")\n";
        
        // Elicit three cases: 1. threshold related to item k, 2. threshold related to item l, 3. threshold non relevant to items couple (k,l)
        if(s >= (c_vec.segment(0, k).sum()) - (k) & s < c_vec.segment(0, k + 1).sum() - (k + 1)){
          // [CASE 1]: threshold related to item k
          
          if(silentFLAG == 0)Rcpp::Rcout << "  |    |_ tau item k:\n";
          int sk = s - (c_vec.segment(0, k).sum()) + (k);
          
          // i3: starting index from i2 for cat sk and sk+1
          int i3 = sk * cl;
          int i3suc = (sk+1) * cl; 
          if(silentFLAG == 0)Rcpp::Rcout << "  |    |_ sk: " << sk << ". Summing over categories item l: ";
          
          // iterate over categories of item l
          for(int sl = 0; sl < cl; sl ++){
            if(silentFLAG == 0)Rcpp::Rcout << " ... cat" << sl ;
            
            // identify pairs_tab column for (sk,sl) and (sk+1, sl)
            int r = i1 + i2 + i3 + sl; 
            int rsuc = i1 + i2 + i3suc + sl;
            
            // read frequences
            int n_sksl = pairs_tab(4, r);
            int n_sksucsl = pairs_tab(4, rsuc);
            
            // read probabilities
            double pi_sksl = pairs_tab(5, r);
            double pi_sksucsl = pairs_tab(5, rsuc);
            
            // identify tau_sk, tau_sl, tau_sl-1
            int sk_tau_index = c_vec.segment(0, k).sum() - (k) + sk;   // index tau_sk in tau vector
            int sl_tau_index = c_vec.segment(0, l).sum() - (l) + sl;   // index tau_sl in tau vector
            double t_sk;                                               // tau_sk
            if(sk == ck-1){
              t_sk = 100; 
            } else {
              t_sk = tau(sk_tau_index);
            }                            
            
            double t_sl;                                               // tau_sl
            if(sl == cl-1){
              t_sl = 100; 
            } else {
              t_sl = tau(sl_tau_index);
            }                            
            double t_sl_prev;                                           // tau_sl-1  
            if(sl == 0){
              t_sl_prev = -100;
            }else{
              t_sl_prev = tau(sl_tau_index-1);
            }
            
            // compute gradient
            double tmp1 = ((n_sksl/(pi_sksl+1e-6))-(n_sksucsl/(pi_sksucsl+1e-6)));
            double tmp2 = R::dnorm(t_sk, 0, 1, 0);
            double tmp3 = R::pnorm((t_sl-rho_kl*t_sk)/(pow(1-pow(rho_kl,2), .5)), 0, 1, 1, 0);
            double tmp4 = R::pnorm((t_sl_prev-rho_kl*t_sk)/(pow(1-pow(rho_kl,2), .5)), 0, 1, 1, 0);
            grs += tmp1 * tmp2 * (tmp3 - tmp4);
          }
          if(silentFLAG == 0)Rcpp::Rcout << "\n";
          
        }else if(s >= (c_vec.segment(0, l).sum())-(l) & s<c_vec.segment(0, l + 1).sum()-(l + 1)){
          // [CASE 2]: threshold related to item l
          
          if(silentFLAG == 0)Rcpp::Rcout << "  |    |_ tau item l\n";
          int sl = s - (c_vec.segment(0, l).sum()) + (l);
          
          if(silentFLAG == 0)Rcpp::Rcout << "  |    |_  sl: " << sl << ". Summing over categories item k: ";
          
          // iterate over categories item k
          for(int sk = 0; sk < ck; sk ++){
            
            // i3: starting index from i2 for cat sk 
            int i3 = sk * cl;
            
            // identify pairs_tab column for (sk,sl) and (sk, sl + 1)
            int r = i1 + i2 + i3 + sl; 
            int rsuc = i1 + i2 + i3 + sl + 1;
            
            // read frequences
            int n_sksl = pairs_tab(4, r);
            int n_skslsuc = pairs_tab(4, rsuc);
            
            // read probabilities
            double pi_sksl = pairs_tab(5, r);
            double pi_skslsuc = pairs_tab(5, rsuc);
            
            // identify tau_sk, tau_sl, tau_sk-1
            int sk_tau_index = c_vec.segment(0, k).sum() - (k) + sk;   // index tau_sk in tau vector
            int sl_tau_index = c_vec.segment(0, l).sum() - (l) + sl;   // index tau_sl in tau vector
            
            double t_sk;                                               // tau_sk
            if(sk == ck-1){
              t_sk = 100; 
            } else {
              t_sk = tau(sk_tau_index);
            }                            
            double t_sk_prev;                                           // tau_sk-1  
            if(sk == 0){
              t_sk_prev = -100;
            }else{
              t_sk_prev = tau(sk_tau_index-1);
            }
            
            double t_sl;                                               // tau_sl
            if(sl == cl-1){
              t_sl = 100; 
            } else {
              t_sl = tau(sl_tau_index);
            }                            
            
            if(silentFLAG == 0)Rcpp::Rcout<<"\n  |    |   |_ sk:"<< sk << ", r: "<< r<<", n_sksl:"
                                          << n_sksl<< ", n_sksl+1:" << n_skslsuc << ", pi_sksl:" 
                                          << pi_sksl << ", pi_sksl+1:"<< pi_skslsuc << ", t_sk:"
                                          << t_sk<< ", t_sl:" << t_sl << "t_sk-1:"<< t_sk_prev;
            // compute gradient
            double tmp1 = ((n_sksl/(pi_sksl+1e-6))-(n_skslsuc/(pi_skslsuc+1e-6)));
            double tmp2 = R::dnorm(t_sl, 0, 1, 0);
            double tmp3 = R::pnorm((t_sk-rho_kl*t_sl)/(pow(1-pow(rho_kl,2), .5)), 0, 1, 1, 0);
            double tmp4 = R::pnorm((t_sk_prev-rho_kl*t_sl)/(pow(1-pow(rho_kl,2), .5)), 0, 1, 1, 0);
            if(silentFLAG == 0)Rcpp::Rcout<<" => out" << sk << ":" << tmp1 * tmp2 * (tmp3 - tmp4);
            grs += tmp1 * tmp2 * (tmp3 - tmp4);
          }
          if(silentFLAG == 0)Rcpp::Rcout << "\n";
          
        }else{
          if(silentFLAG == 0)Rcpp::Rcout << "  |    |_  tau of other item\n";
        }
        if(silentFLAG == 0)Rcpp::Rcout << "  |    |_  Thresholds:: " << iter<< "/"<< tau.size() 
                                       <<". Tot:: "<< iter << "/"<< d <<". Val ="<< grs <<"\n";
        gradientr(iter) += grs;
        iter ++;
      }
      if(silentFLAG == 0)Rcpp::Rcout << "  |_ Done. \n";
      
      ///////////////////////////////////////////////////////////
      // (k,l)-pair likelihood derivative wrt URV correlation: //
      // intermediate step for derivatives wrt                 //
      // loadings and factor correlations                      //
      ///////////////////////////////////////////////////////////
      if(silentFLAG == 0)Rcpp::Rcout << "\n- Intermediate derivative for loadings and correlation: \n";
      double tmp_kl = 0; // temporary location of the gradient
      
      // double loop: iterate over each combination of categories of items k and l
      for(int sk = 0; sk < ck; sk ++){
        for(int sl = 0; sl < cl; sl ++){
          if(silentFLAG == 0)Rcpp::Rcout << "  |_ sk: "<< sk << ", sl: " << sl << ": \n";
          
          // identify pairs_tab column for (sk,sl)
          int i3 = sk * cl;
          int r = i1 + i2 + i3 + sl; 
          
          // read freq
          int n_sksl = pairs_tab(4, r);
          
          // read prob
          double pi_sksl = pairs_tab(5, r);
          
          // identify tau_sk, tau_sl, tau_sk-1, tau_sl-1
          int sk_tau_index = c_vec.segment(0, k).sum() - (k) + sk;   // index tau_sk in tau vector
          int sl_tau_index = c_vec.segment(0, l).sum() - (l) + sl;   // index tau_sl in tau vector
          double t_sk;                                               // tau_sk
          if(sk == ck-1){
            t_sk = 100; 
          } else {
            t_sk = tau(sk_tau_index);
          }                            
          double t_sk_prev;                                           // tau_sk-1  
          if(sk == 0){
            t_sk_prev = -100;
          }else{
            t_sk_prev = tau(sk_tau_index-1);
          }
          
          double t_sl;                                               // tau_sl
          if(sl == cl-1){
            t_sl = 100; 
          } else {
            t_sl = tau(sl_tau_index);
          }                            
          double t_sl_prev;                                           // tau_sl-1  
          if(sl == 0){
            t_sl_prev = -100;
          }else{
            t_sl_prev = tau(sl_tau_index-1);
          }
          
          // phi(t_sk, t_sl; rho_kl)
          double d1 = pbv::pbv_rcpp_dbvnorm0( t_sk, t_sl, rho_kl, 0);                  
          
          // phi(t_sk, t_sl-1; rho_kl)
          double d2 = pbv::pbv_rcpp_dbvnorm0( t_sk, t_sl_prev, rho_kl, 0);
          
          // phi(t_sk-1, t_sl; rho_kl)
          double d3 = pbv::pbv_rcpp_dbvnorm0( t_sk_prev, t_sl, rho_kl, 0);
          
          // phi(t_sk-1, t_sl-1; rho_kl)
          double d4 = pbv::pbv_rcpp_dbvnorm0( t_sk_prev, t_sl_prev, rho_kl, 0);
          
          tmp_kl += (n_sksl/(pi_sksl+1e-6)) * ( d1 - d2 - d3 + d4);
        }
      }
      if(silentFLAG == 0)Rcpp::Rcout << "  |_ tmp_kl:" << tmp_kl << "\n";
      
      ///////////////////////////////////////////////////
      // (k,l)-pair likelihood derivative wrt loadings //
      ///////////////////////////////////////////////////
      if(silentFLAG == 0)Rcpp::Rcout << "\n- Gradient wrt loadings: \n";
      
      // double loop: iterate over elements of loadings matrix
      for(int j = 0; j < p; j++){
        for(int v = 0; v < q; v++){
          if(silentFLAG == 0)Rcpp::Rcout << "  |_ visiting lambda_"<< j << v <<":\n";
          
          // elicit three cases: 1. free loading item k, 2. free loading l, 3. other
          if(j == k){
            if(A(j,v)!=0 ){
              if(silentFLAG == 0)Rcpp::Rcout << "  |   |_ item k, free loading:\n";
              Eigen::VectorXd ev(q); ev.fill(0.0); ev(v) = 1;
              double d_rho_kl = ev.transpose() * Sigma_u * lambdal;
              if(silentFLAG == 0)Rcpp::Rcout << "  |   |_ d_rho_kl:" << d_rho_kl << "\n";
              gradientr(iter) += tmp_kl * d_rho_kl;
              if(silentFLAG == 0)Rcpp::Rcout << "  |   |_ Loadings:: " << iter - tau.size() << "/"<< lambda.size() <<". Tot:: "<< iter << "/"<< d << "\n";
              
              iter ++;
            }
          }else if (j == l){
            if(A(j,v)!=0 ){
              if(silentFLAG == 0)Rcpp::Rcout << "  |   |_ item l, free loading:\n";
              Eigen::VectorXd ev(q); ev.fill(0.0); ev(v) = 1;
              double d_rho_kl = lambdak.transpose() * Sigma_u * ev;
              if(silentFLAG == 0)Rcpp::Rcout << "  |   |_ d_rho_kl:" << d_rho_kl << "\n";
              if(silentFLAG == 0)Rcpp::Rcout << "  |   |_ Loadings:: " << iter - tau.size() << "/"<< lambda.size() <<". Tot:: "<< iter << "/"<< d << "\n";
              gradientr(iter) += tmp_kl * d_rho_kl;
              
              iter ++;
            }
          }else if(A(j,v)!=0){
            
            if(silentFLAG == 0)Rcpp::Rcout << "  |  |_ Loadings:: " << iter - tau.size() << "/"<< lambda.size() <<". Tot:: "<< iter << "/"<< d << " [not included]\n";
            iter ++;
          }
        }
      }
      if(silentFLAG == 0)Rcpp::Rcout << "  |_ Done. \n";
      
      //////////////////////////////////////////////////////////////
      // (k,l)-pair likelihood derivative wrt latent correlations //
      //////////////////////////////////////////////////////////////     
      if(silentFLAG == 0)Rcpp::Rcout << "\n- Gradient wrt correlations: \n";
      
      if(corrFLAG == 1){
        // double loop: iterate over each non-redundant latent correlation
        for(int v = 1; v < q; v++){
          for(int  t = 0; t < v; t++){
            Eigen::VectorXd ev(q); ev.fill(0.0); ev(v) = 1;
            Eigen::VectorXd et(q); et.fill(0.0); et(t) = 1;
            double trho = transformed_rhos(iter - tau.size() - lambda.size());
            double drho = 2*exp(2*trho) * pow((exp(2*trho) + 1),-1) * ( 1 - ( exp(2*trho) - 1) * pow((exp(2*trho) + 1),-1) );
            
            // impose symmetric structure
            Eigen::MatrixXd Jvt = ev * et.transpose();
            Eigen::MatrixXd Jtv = et * ev.transpose();
            Eigen::MatrixXd Svt = Jvt + Jtv - Jvt*Jvt;
            
            double d_rho_kl = lambdak.transpose() * (Svt * drho) * lambdal;
            
            //if(silentFLAG == 0) 
            gradientr(iter) += tmp_kl * d_rho_kl;
            iter ++;
          }
        }
      }
      
      if(silentFLAG == 0)Rcpp::Rcout << "\n=====> gradient r-th pair:\n" << gradientr << "\n";
      scores.segment(d*iter_pair_kl,d) = gradientr;
      if(silentFLAG == 0)Rcpp::Rcout << "\n";
      iter_pair_kl ++;
    }    
    
  }  
  
  //////////////////////
  // OPTIMAL WEIGHTS  //
  //////////////////////
  Eigen::MatrixXd varkl(d,d); varkl.fill(0.0);
  Eigen::MatrixXd scores_cov(d*p*(p-1)/2,d*p*(p-1)/2); scores_cov.fill(0.0);
  Eigen::MatrixXd scores_stackvar(d*p*(p-1)/2,d); scores_stackvar.fill(0.0);
  Eigen::MatrixXd weights(d*p*(p-1)/2,d*p*(p-1)/2); weights.fill(0.0);
  Eigen::MatrixXd ows;
  Eigen::VectorXd Dpisksl(d); Dpisksl.fill(0.0);
  double Pisksl;
  
  if(weightsFLAG == 0){
    
    // COMPUTE GRADIENT OF EACH pi_sksl
    pairs_tab.conservativeResize(pairs_tab.rows() + d, Eigen::NoChange_t() );
    
    // double loop: pairs (k,l)
    for(int k = 1; k < p; k++){
      int ck = c_vec(k);
      Eigen::VectorXd lambdak = Lam.row(k);
      // identify column index in freq table
      // i1: starting index item k
      int i1 = 0; 
      if(k > 1){
        for(int u = 1; u < k; u++){
          int cu = c_vec(u);
          i1 += cu * c_vec.segment(0,u).sum();
        }
      }
      
      for(int l = 0; l < k; l++){
        if(silentFLAG == 0)Rcpp::Rcout << "\nPair (" << k << "," << l << "): \n";
        //Rcpp::Rcout << "\nPair (" << k << "," << l << "): \n";
        int cl = c_vec(l);
        Eigen::VectorXd lambdal = Lam.row(l);
        double rho_kl = lambdak.transpose() * Sigma_u * lambdal;
        
        // i2 starting index from i1 dor item l
        int i2 = 0;
        if(l > 0){
          i2 = c_vec.segment(0,l).sum() * c_vec(k);
        }
        
        Eigen::VectorXd exp_score(d); exp_score.fill(0.0);
        // double loop categories combinations (sk,sl)
        for(int sk = 0; sk < ck; sk ++){
          // i3: starting index from i2 for cat sk
          int i3 = sk * cl;
          
          for(int sl = 0; sl < cl; sl ++){
            if(silentFLAG == 0)Rcpp::Rcout << "  |_ sk: "<< sk << ", sl: " << sl << ": \n";
            //Rcpp::Rcout << "  |_ sk: "<< sk << ", sl: " << sl << ": \n";
            // identify pairs_tab column for (sk,sl)
            int r = i1 + i2 + i3 + sl; 
            
            // read freq
            int n_sksl = pairs_tab(4, r);
            
            // read prob
            double pi_sksl = pairs_tab(5, r);
            int sk_tau_index = c_vec.segment(0, k).sum() - (k) + sk;   // index sk in tau vector
            int sl_tau_index = c_vec.segment(0, l).sum() - (l) + sl;   // index sl in tau vector
            
            double t_sk;                                               // tau_sk
            if(sk == ck-1){
              t_sk = 100; 
            } else {
              t_sk = tau(sk_tau_index);
            }     
            
            double t_sk_prev;                                           // tau_sk-1  
            if(sk == 0){
              t_sk_prev = -100;
            }else{
              t_sk_prev = tau(sk_tau_index-1);
            }
            
            double t_sl;                                               // tau_sl
            if(sl == cl-1){
              t_sl = 100; 
            } else {
              t_sl = tau(sl_tau_index);
            }         
            
            double t_sl_prev;                                           // tau_sl-1  
            if(sl == 0){
              t_sl_prev = -100;
            }else{
              t_sl_prev = tau(sl_tau_index-1);
            }
            
            //Rcpp::Rcout << "  |    |_ n:"<<n_sksl<<", pi:"<<pi_sksl<<", t_sk:"<<t_sk<< ", t_sk_prev:"<<t_sk_prev<<", t_sl: "<<t_sl<<", t_sl_prev:"<<t_sl_prev<< "\n";
            //Intermediate derivative pi wrt to kl correlation
            // phi(t_sk, t_sl; rho_kl)
            double d1 = pbv::pbv_rcpp_dbvnorm0( t_sk, t_sl, rho_kl, 0);                  
            
            // phi(t_sk, t_sl-1; rho_kl)
            double d2 = pbv::pbv_rcpp_dbvnorm0( t_sk, t_sl_prev, rho_kl, 0);
            
            // phi(t_sk-1, t_sl; rho_kl)
            double d3 = pbv::pbv_rcpp_dbvnorm0( t_sk_prev, t_sl, rho_kl, 0);
            
            // phi(t_sk-1, t_sl-1; rho_kl)
            double d4 = pbv::pbv_rcpp_dbvnorm0( t_sk_prev, t_sl_prev, rho_kl, 0);
            
            double tmp_sksl = d1 - d2 - d3 + d4;
            
            //Rcpp::Rcout << "  |    |_ d1:"<<d1<<", d2:"<<d2<<", d3:"<<d3<< ", d4:"<<d4<<", tmp_sksl: "<< tmp_sksl<< "\n";
            // for each (sk, sl) compute the gradient of pi wrt the parameter vector
            Eigen::VectorXd dpi(d); dpi.fill(0.0);
            
            // iterator over parameter vector
            int iter_th = 0; 
            
            ////////////////////////////////////////////////
            // gradient pi_sksl wrt to thresholds 
            ///////////////////////////////////////////////
            
            // loop: iterate over elements of thresholds vector
            for(int s = 0; s < tau.size(); s++){
              
              // Elicit three cases: 1. threshold related to item k, 2. threshold related to item l, 3. threshold non relevant to items couple (k,l)
              if(s >= (c_vec.segment(0, k).sum()) - (k) & s < c_vec.segment(0, k + 1).sum() - (k + 1)){
                // [CASE 1]: threshold related to item k
                int sk_a = s - (c_vec.segment(0, k).sum()) + (k);
                //Rcpp::Rcout << "  |    |_ tau item k. sk_a:"<< sk_a <<"\n";
                if(sk_a == sk){
                  double tmp1 = R::dnorm(t_sk, 0, 1, 0);
                  double tmp2 = R::pnorm((t_sl-rho_kl*t_sk)/(pow(1-pow(rho_kl,2), .5)), 0, 1, 1, 0);
                  double tmp3 = R::pnorm((t_sl_prev-rho_kl*t_sk)/(pow(1-pow(rho_kl,2), .5)), 0, 1, 1, 0);;
                  dpi(iter_th) = tmp1*tmp2-tmp1*tmp3;
                }else if(sk_a == (sk-1)){
                  double tmp1 = R::dnorm(t_sk_prev, 0, 1, 0);
                  double tmp2 = R::pnorm((t_sl-rho_kl*t_sk_prev)/(pow(1-pow(rho_kl,2), .5)), 0, 1, 1, 0);
                  double tmp3 = R::pnorm((t_sl_prev-rho_kl*t_sk_prev)/(pow(1-pow(rho_kl,2), .5)), 0, 1, 1, 0);;
                  dpi(iter_th) = -tmp1*tmp2+tmp1*tmp3;
                }else{
                  dpi(iter_th) = 0;
                }
                
              }else if(s >= (c_vec.segment(0, l).sum())-(l) & s<c_vec.segment(0, l + 1).sum()-(l + 1)){
                // [CASE 2]: threshold related to item l
                int sl_a = s - (c_vec.segment(0, l).sum()) + (l);
                //Rcpp::Rcout << "  |    |_ tau item l. sl_a:"<< sl_a<<"\n";
                if(sl_a == sl){
                  double tmp1 = R::dnorm(t_sl, 0, 1, 0);
                  double tmp2 = R::pnorm((t_sk-rho_kl*t_sl)/(pow(1-pow(rho_kl,2), .5)), 0, 1, 1, 0);
                  double tmp3 = R::pnorm((t_sk_prev-rho_kl*t_sl)/(pow(1-pow(rho_kl,2), .5)), 0, 1, 1, 0);;
                  dpi(iter_th) = tmp1*tmp2-tmp1*tmp3;
                }else if(sl_a == (sl-1)){
                  double tmp1 = R::dnorm(t_sl_prev, 0, 1, 0);
                  double tmp2 = R::pnorm((t_sk-rho_kl*t_sl_prev)/(pow(1-pow(rho_kl,2), .5)), 0, 1, 1, 0);
                  double tmp3 = R::pnorm((t_sk_prev-rho_kl*t_sl_prev)/(pow(1-pow(rho_kl,2), .5)), 0, 1, 1, 0);;
                  dpi(iter_th) = -tmp1*tmp2+tmp1*tmp3;
                }else{
                  dpi(iter_th) = 0;
                }
              } else {
                // [CASE 3]: threshold non related to (k,l)
                //Rcpp::Rcout << "  |    |_ tau non related to (k,l)\n";
                dpi(iter_th) = 0;
              }
              iter_th ++;
            }
            
            //////////////////////////////////////////////
            // gradient pi_sksl wrt to loadings
            //////////////////////////////////////
            
            // double loop: iterate over elements of loadings matrix
            for(int j = 0; j < p; j++){
              for(int v = 0; v < q; v++){
                if(silentFLAG == 0)Rcpp::Rcout << "  |_ visiting lambda_"<< j << v <<":\n";
                
                // elicit three cases: 1. free loading item k, 2. free loading l, 3. other
                if(j == k){
                  if(A(j,v)!=0 ){
                    if(silentFLAG == 0)Rcpp::Rcout << "  |   |_ item k, free loading:\n";
                    Eigen::VectorXd ev(q); ev.fill(0.0); ev(v) = 1;
                    double d_rho_kl = ev.transpose() * Sigma_u * lambdal;
                    if(silentFLAG == 0)Rcpp::Rcout << "  |   |_ d_rho_kl:" << d_rho_kl << "\n";
                    dpi(iter_th) = tmp_sksl * d_rho_kl;
                    if(silentFLAG == 0)Rcpp::Rcout << "  |   |_ Loadings:: " << iter_th - tau.size() << "/"<< lambda.size() <<". Tot:: "<< iter_th << "/"<< d << "\n";
                    
                    iter_th ++;
                  }
                }else if (j == l){
                  if(A(j,v)!=0 ){
                    if(silentFLAG == 0)Rcpp::Rcout << "  |   |_ item l, free loading:\n";
                    Eigen::VectorXd ev(q); ev.fill(0.0); ev(v) = 1;
                    double d_rho_kl = lambdak.transpose() * Sigma_u * ev;
                    if(silentFLAG == 0)Rcpp::Rcout << "  |   |_ d_rho_kl:" << d_rho_kl << "\n";
                    if(silentFLAG == 0)Rcpp::Rcout << "  |   |_ Loadings:: " << iter_th - tau.size() << "/"<< lambda.size() <<". Tot:: "<< iter_th << "/"<< d << "\n";
                    dpi(iter_th) = tmp_sksl * d_rho_kl;
                    
                    iter_th ++;
                  }
                }else if(A(j,v)!=0){
                  
                  if(silentFLAG == 0)Rcpp::Rcout << "  |  |_ Loadings:: " << iter_th - tau.size() << "/"<< lambda.size() <<". Tot:: "<< iter_th << "/"<< d << " [not included]\n";
                  iter_th ++;
                }
              }
            }
            
            /////////////////////////////////////////
            // gradient pi_sksl wrt correlations
            ///////////////////////////////////////
            if(corrFLAG == 1){
              // double loop: iterate over each non-redundant latent correlation
              for(int v = 1; v < q; v++){
                for(int  t = 0; t < v; t++){
                  Eigen::VectorXd ev(q); ev.fill(0.0); ev(v) = 1;
                  Eigen::VectorXd et(q); et.fill(0.0); et(t) = 1;
                  double trho = transformed_rhos(iter_th- tau.size() - lambda.size());
                  double drho = 2*exp(2*trho) * pow((exp(2*trho) + 1),-1) * ( 1 - ( exp(2*trho) - 1) * pow((exp(2*trho) + 1),-1) );
                  
                  // impose symmetric structure
                  Eigen::MatrixXd Jvt = ev * et.transpose();
                  Eigen::MatrixXd Jtv = et * ev.transpose();
                  Eigen::MatrixXd Svt = Jvt + Jtv - Jvt*Jvt;
                  
                  double d_rho_kl = lambdak.transpose() * (Svt * drho) * lambdal;
                  
                  //if(silentFLAG == 0) 
                  dpi(iter_th) = tmp_sksl * d_rho_kl;
                  iter_th ++;
                }
              }
            }
            
            pairs_tab.block(6,r,d,1) = dpi;
            exp_score += dpi;
            if(k == item1 & l == item2 & sk == cat1 & sl == cat2){Dpisksl = dpi; Pisksl = pi_sksl;}
          }
        }
        //Rcpp::Rcout << "\n Exp score:\n" << exp_score << " \n";
      }
    }
    
    // COMPUTE SCORES COVARIANCES
    iter_pair_kl = 0;
    for(int k = 1; k < p; k++){
      int ck = c_vec(k);
      // identify column index in freq table
      // i1: starting index item k
      int i1_k = 0; 
      if(k > 1){
        for(int u = 1; u < k; u++){
          int cu = c_vec(u);
          i1_k += cu * c_vec.segment(0,u).sum();
        }
      }
      
      Eigen::VectorXd lambdak = Lam.row(k);
      
      for(int l = 0; l < k; l++){
        if(silentFLAG == 0)Rcpp::Rcout << "\nPair (" << k << "," << l << "): \n";
        int cl = c_vec(l);
        
        // i2 starting index from i1 dor item l
        int i2_l = 0;
        if(l > 0){
          i2_l = c_vec.segment(0,l).sum() * c_vec(k);
        }
        
        Eigen::VectorXd lambdal = Lam.row(l);
        double rho_kl = lambdak.transpose() * Sigma_u * lambdal;
        
        int iter_pair_rt = 0;
        for(int r = 1; r < p; r++){
          int cr = c_vec(r);
          
          // identify column index in freq table
          // i1: starting index item k
          int i1_r = 0; 
          if(r > 1){
            for(int u = 1; u < r; u++){
              int cu = c_vec(u);
              i1_r += cu * c_vec.segment(0,u).sum();
            }
          }
          
          Eigen::VectorXd lambdar = Lam.row(r);
          
          double rho_kr = lambdak.transpose() * Sigma_u * lambdar;
          double rho_lr = lambdal.transpose() * Sigma_u * lambdar;
          
          for(int t = 0; t < r; t++){
            if(silentFLAG == 0)Rcpp::Rcout << "\nPair (" << r << "," << t << "): \n";
            
            int ct = c_vec(t);
            
            // i2 starting index from i1 dor item l
            int i2_t = 0;
            if(t > 0){
              i2_t = c_vec.segment(0,t).sum() * c_vec(r);
            }
            
            // initialize covariance matrix between (k,l) and (r,t)
            Eigen::MatrixXd klrt_cov(d,d); klrt_cov.fill(0.0);
            
            Eigen::VectorXd lambdat = Lam.row(t);
            double rho_kt = lambdak.transpose() * Sigma_u * lambdat;
            double rho_lt = lambdal.transpose() * Sigma_u * lambdat;
            double rho_rt = lambdar.transpose() * Sigma_u * lambdat;
            
            
            if(iter_pair_rt <= iter_pair_kl){
              // 4 loops over categories (sk, sl, br, bt)
              for(int sk = 0; sk < ck; sk ++){
                // i3: starting index from i2 for cat sk
                int i3_sk = sk * cl;
                
                for(int sl = 0; sl < cl; sl ++){
                  if(silentFLAG == 0)Rcpp::Rcout << "  |_ sk: "<< sk << ", sl: " << sl << ": \n";
                  // identify pairs_tab column for (sk,sl)
                  int r_sksl = i1_k + i2_l + i3_sk + sl; 
                  
                  // read freq
                  int n_sksl = pairs_tab(4, r_sksl);
                  
                  // read prob
                  double pi_sksl = pairs_tab(5, r_sksl);
                  
                  // read gradient pi
                  Eigen::VectorXd dpi_sksl = pairs_tab.block(6,r_sksl,d,1);
                  
                  int sk_tau_index = c_vec.segment(0, k).sum() - (k) + sk;   // index sk in tau vector
                  int sl_tau_index = c_vec.segment(0, l).sum() - (l) + sl;   // index sl in tau vector
                  
                  double t_sk;                                               // tau_sk
                  if(sk == ck-1){
                    t_sk = 100; 
                  } else {
                    t_sk = tau(sk_tau_index);
                  }                            
                  double t_sk_prev;                                           // tau_sk-1  
                  if(sk == 0){
                    t_sk_prev = -100;
                  }else{
                    t_sk_prev = tau(sk_tau_index-1);
                  }
                  
                  double t_sl;                                               // tau_sl
                  if(sl == cl-1){
                    t_sl = 100; 
                  } else {
                    t_sl = tau(sl_tau_index);
                  }                            
                  double t_sl_prev;                                           // tau_sl-1  
                  if(sl == 0){
                    t_sl_prev = -100;
                  }else{
                    t_sl_prev = tau(sl_tau_index-1);
                  }       
                  
                  for(int br = 0; br < cr; br ++){
                    // i3_br: starting index from i2_t for cat br
                    int i3_br = br * ct;
                    
                    for(int bt = 0; bt < ct; bt ++){
                      if(silentFLAG == 0)Rcpp::Rcout << "  |_ br: "<< br << ", bt: " << bt << ": \n";
                      // identify pairs_tab column for (br,bt)
                      int r_brbt = i1_r + i2_t + i3_br + bt; 
                      
                      // read freq
                      int n_brbt = pairs_tab(4, r_brbt);
                      
                      // read prob
                      double pi_brbt = pairs_tab(5, r_brbt);
                      
                      // read gradient pi
                      Eigen::VectorXd dpi_brbt = pairs_tab.block(6, r_brbt, d, 1);
                      
                      int br_tau_index = c_vec.segment(0, r).sum() - (r) + br;   // index br in tau vector
                      int bt_tau_index = c_vec.segment(0, t).sum() - (t) + bt;   // index bt in tau vector
                      
                      double t_br;                                               // tau_br
                      if(br == cr-1){
                        t_br = 100; 
                      } else {
                        t_br = tau(br_tau_index);
                      }                            
                      double t_br_prev;                                           // tau_br-1  
                      if(br == 0){
                        t_br_prev = -100;
                      }else{
                        t_br_prev = tau(br_tau_index-1);
                      }
                      
                      double t_bt;                                               // tau_bt
                      if(bt == ct-1){
                        t_bt = 100; 
                      } else {
                        t_bt = tau(bt_tau_index);
                      }                            
                      double t_bt_prev;                                           // tau_bt-1  
                      if(bt == 0){
                        t_bt_prev = -100;
                      }else{
                        t_bt_prev = tau(bt_tau_index-1);
                      }       
                      //Rcpp::Rcout << "\nPairs: (" << k << "," << l << "), ("<<r<<","<<t<<"). Cat: ("<<sk<<","<<sl<<"), ("<<br<<","<<bt<<")";
                      
                      // compute E[Isksl * Ibrbt]
                      double EIskslIbrbt = 0;
                      if(k == r & l == t){
                        // 2 items shared
                        if(sk == br & sl == bt){
                          EIskslIbrbt = pi_sksl;
                          //Rcpp::Rcout << ". 2 items, EIskslIbrbt:"<<EIskslIbrbt ;
                        } 
                      } else if(r == k | r == l){
                        // r-th item shared
                        if( r == k & br == sk){
                          // -> pi_skslbt
                          //int n_skslbt = 0;
                          //for(int i = 0; i < n; i++){
                          //if(sk == y(i,k) & sl == y(i,l) & bt == y(i,t)){
                          //n_skslbt++;
                          //}
                          //}
                          //EIskslIbrbt = double(n_skslbt)/double(n); 
                          //Rcpp::Rcout << ". r = k, n_skslbt:"<< n_skslbt <<", EIskslIbrbt:"<<EIskslIbrbt ;
                          Eigen::VectorXd upper(3); upper << t_sk, t_sl, t_bt;
                          Eigen::VectorXd lower(3); lower << t_sk_prev, t_sl_prev, t_bt_prev;
                          Eigen::MatrixXd sig(3,3); sig.setIdentity();
                          sig(0,1) = rho_kl; sig(1,0) = rho_kl;
                          sig(0,2) = rho_kt; sig(2,0) = rho_kt;
                          sig(1,2) = rho_lt; sig(2,1) = rho_lt;
                          if(t_sk< t_sk_prev | t_sl< t_sl_prev | t_bt< t_bt_prev){
                            Rcpp::Rcout << "\nPairs: (" << k << "," << l << "), ("<<r<<","<<t<<"). Cat: ("<<sk<<","<<sl<<"), ("<<br<<","<<bt<<")";
                            
                            Rcpp::Rcout << "\nlower:"<< t_sk_prev << ","<< t_sl_prev << ","<< t_bt_prev << "; upper:"<< t_sk<<","<<t_sl<<","<<t_bt<<"\n";
                            
                          }
                          EIskslIbrbt = Rcpp::as<double>(pmvnorm(Rcpp::Named("lower")=lower, Rcpp::Named("upper")=upper, Rcpp::Named("sigma")=sig));
                          
                          //EIskslIbrbt = 0;
                        } else if(r == l & br == sl){
                          // -> pi_skslbt
                          //int n_skslbt = 0;
                          //for(int i = 0; i < n; i++){
                          //if(sk == y(i,k) & sl == y(i,l) & bt == y(i,t)){
                          //n_skslbt++;
                          //}
                          //}
                          //EIskslIbrbt = double(n_skslbt)/double(n);
                          //Rcpp::Rcout << ". r = l, n_skslbt:"<< n_skslbt <<", EIskslIbrbt:"<<EIskslIbrbt ;
                          Eigen::VectorXd upper(3); upper << t_sk, t_sl, t_bt;
                          Eigen::VectorXd lower(3); lower << t_sk_prev, t_sl_prev, t_bt_prev;
                          Eigen::MatrixXd sig(3,3); sig.setIdentity();
                          sig(0,1) = rho_kl; sig(1,0) = rho_kl;
                          sig(0,2) = rho_kt; sig(2,0) = rho_kt;
                          sig(1,2) = rho_lt; sig(2,1) = rho_lt;
                          EIskslIbrbt = Rcpp::as<double>(pmvnorm(Rcpp::Named("lower")=lower, Rcpp::Named("upper")=upper, Rcpp::Named("sigma")=sig));
                          
                        }
                      } else if(t == k | t == l){
                        // t-th item shared
                        if(t == k & bt == sk){
                          //-> pi_skslbr
                          //int n_skslbr = 0;
                          //for(int i = 0; i < n; i++){
                          //if(sk == y(i,k) & sl == y(i,l) & br == y(i,r)) {
                          //n_skslbr++;
                          //}
                          //}
                          //EIskslIbrbt = double(n_skslbr)/double(n);
                          //Rcpp::Rcout << ". t = k, n_skslbr:"<<n_skslbr<<", EIskslIbrbt:"<<EIskslIbrbt ;
                          Eigen::VectorXd upper(3); upper << t_sk, t_sl, t_br;
                          Eigen::VectorXd lower(3); lower << t_sk_prev, t_sl_prev, t_br_prev;
                          Eigen::MatrixXd sig(3,3); sig.setIdentity();
                          sig(0,1) = rho_kl; sig(1,0) = rho_kl;
                          sig(0,2) = rho_kr; sig(2,0) = rho_kr;
                          sig(1,2) = rho_lr; sig(2,1) = rho_lr;
                          EIskslIbrbt = Rcpp::as<double>(pmvnorm(Rcpp::Named("lower")=lower, Rcpp::Named("upper")=upper, Rcpp::Named("sigma")=sig));
                          
                        }else if(t == l & bt == sl){
                          //-> pi_skslbr
                          //int n_skslbr = 0;
                          //for(int i = 0; i < n; i++){
                          //if(sk == y(i,k) & sl == y(i,l) & br == y(i,r)) {
                          //n_skslbr++;
                          //}
                          //}
                          //EIskslIbrbt = double(n_skslbr)/double(n);
                          //Rcpp::Rcout << ". t = l, n_skslbr:"<<n_skslbr<<", EIskslIbrbt:"<<EIskslIbrbt ;
                          Eigen::VectorXd upper(3); upper << t_sk, t_sl, t_br;
                          Eigen::VectorXd lower(3); lower << t_sk_prev, t_sl_prev, t_br_prev;
                          Eigen::MatrixXd sig(3,3); sig.setIdentity();
                          sig(0,1) = rho_kl; sig(1,0) = rho_kl;
                          sig(0,2) = rho_kr; sig(2,0) = rho_kr;
                          sig(1,2) = rho_lr; sig(2,1) = rho_lr;
                          EIskslIbrbt = Rcpp::as<double>(pmvnorm(Rcpp::Named("lower")=lower, Rcpp::Named("upper")=upper, Rcpp::Named("sigma")=sig));
                          
                        }
                      } else {
                        // no item shared
                        // -> pi_skslbrbt
                        //int n_skslbrbt = 0;
                        //for(int i = 0; i < n; i++){
                        //if(sk == y(i,k) & sl == y(i,l) & br == y(i,r) & bt == y(i,t)) {
                        //n_skslbrbt++;
                        //}
                        //}
                        //EIskslIbrbt = double(n_skslbrbt)/double(n);
                        //Rcpp::Rcout << ". no item, n_skslbrbt:"<<n_skslbrbt<<", EIskslIbrbt:"<<EIskslIbrbt ;
                        Eigen::VectorXd upper(4); upper << t_sk, t_sl, t_br, t_bt;
                        Eigen::VectorXd lower(4); lower << t_sk_prev, t_sl_prev, t_br_prev, t_bt_prev;
                        Eigen::MatrixXd sig(4,4); sig.setIdentity();
                        sig(0,1) = rho_kl; sig(1,0) = rho_kl;
                        sig(0,2) = rho_kr; sig(2,0) = rho_kr;
                        sig(0,3) = rho_kt; sig(3,0) = rho_kt;
                        sig(1,2) = rho_lr; sig(2,1) = rho_lr;
                        sig(1,3) = rho_lt; sig(3,1) = rho_lt;
                        sig(2,3) = rho_rt; sig(3,2) = rho_rt;
                        
                        EIskslIbrbt = Rcpp::as<double>(pmvnorm(Rcpp::Named("lower")=lower, Rcpp::Named("upper")=upper, Rcpp::Named("sigma")=sig));
                        
                      }
                      
                      // compute E[n_sksl n_brbt]
                      double Enskslnbrbt = n*EIskslIbrbt + n*(n-1)*pi_sksl*pi_brbt;
                      
                      
                      
                      Eigen::MatrixXd skslbrbt_contribution(d,d);
                      skslbrbt_contribution = (Enskslnbrbt/(pi_sksl*pi_brbt)) * (dpi_sksl * dpi_brbt.transpose());
                      klrt_cov += skslbrbt_contribution;
                      //Rcpp::Rcout << "\nPairs: (" << k << "," << l << "), ("<<r<<","<<t<<"). Cat: ("
                      //            <<sk<<","<<sl<<"), ("<<br<<","<<bt<<"), E[Isksl*Ibrbt]:"<<EIskslIbrbt
                      //           << ", E[n_sksl*n_brbt]:"<< Enskslnbrbt;
                    }
                  }
                }
              }
              
              //Rcpp::Rcout << "\nPairs: (" << k << "," << l << "), ("<<r<<","<<t<<"). n:"<< iter_pair_kl<<","<<iter_pair_rt<<", Cov:\n"<< klrt_cov;
              
              //Store covariances in scores_cov and scores_stackvar
              if(iter_pair_kl != iter_pair_rt){
                scores_cov.block(d*iter_pair_kl, d*iter_pair_rt, d, d) = klrt_cov;
                scores_cov.block(d*iter_pair_rt, d*iter_pair_kl, d, d) = klrt_cov.transpose();
                
              }else{
                if(k == item1 & l == item2){varkl = klrt_cov;}
                scores_cov.block(d*iter_pair_kl, d*iter_pair_rt, d, d) = klrt_cov;
                scores_stackvar.block(d*iter_pair_kl,0,d,d) = klrt_cov;
              }
            }
            
            
            
            
            
            iter_pair_rt ++;
          }
        }
        iter_pair_kl ++;
      }
    }
    
    Eigen::MatrixXd Imat(d*p*(p-1)/2,d*p*(p-1)/2); Imat.setIdentity();
    Eigen::LDLT<Eigen::MatrixXd> Sigma_ldlt(scores_cov);
    Eigen::MatrixXd Sigma_inv = Sigma_ldlt.solve(Imat);
    weights = scores_stackvar.transpose()*Sigma_inv;
    
    ows = weights * scores;
  } else {
    ows = pre_weights * scores;
  }
  
  // output list
  Rcpp::List output =
    Rcpp::List::create(
      //Rcpp::Named("freq") = freq,
      Rcpp::Named("pairs_tab") = pairs_tab,
      Rcpp::Named("scores_cov") = scores_cov,
      Rcpp::Named("scores_stackvar") = scores_stackvar,
      Rcpp::Named("weights") = weights,
      Rcpp::Named("scores") = scores,
      Rcpp::Named("ows") = ows,
      Rcpp::Named("Dpisksl") = Dpisksl,
      Rcpp::Named("varkl") = varkl,
      Rcpp::Named("Pisksl") = Pisksl
    );
  return(output);
}














































