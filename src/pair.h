#ifndef pair_H
#define pair_H

//Individual pair of item contribution to
// 1. Log-likelihood
// 2. Gradient
// 3. Diagonal of Hessian
void pair_contribution(
    // Parameters
    Eigen::Map<Eigen::MatrixXd> A,
    Eigen::Map<Eigen::VectorXd> c_vec,
    const Eigen::VectorXd &theta,
    const int corrFLAG,
    
    // Input:
    const unsigned int k,
    const unsigned int l,
    const Eigen::MatrixXd &pairs_table,
    
    // Options:
    const unsigned int silentFLAG,
    const unsigned int DFLAG,
    const unsigned int gradFLAG,
    
    // Output:
    double &ll,
    Eigen::VectorXd &gradient

){
  unsigned int p = A.rows();
  unsigned int q = A.cols();
  unsigned int d = theta.size();
  unsigned int c = c_vec.sum();
  unsigned int nthr = c-p;
  unsigned int ncorr = q*(q-1)/2;
  unsigned int nload = d-nthr-ncorr;
  
  double tmp_ll = 0;
  Eigen::VectorXd tmp_gradient(d); tmp_gradient.setZero();

  // rearrange parameters
  Eigen::MatrixXd Lam            = get_Lam(A, c_vec, theta);
  Eigen::MatrixXd Sigma_u        = get_Sigma_u(A, theta);
  Eigen::VectorXd tau            = get_tau(theta, c_vec);
  Eigen::VectorXd transformed_rhos=theta.segment(nthr+nload, d);
  
  // Identifies quantities related to pair (k,l)
  unsigned int ck = c_vec(k);
  unsigned int cl = c_vec(l);
  Eigen::VectorXd lambdak = Lam.row(k);
  Eigen::VectorXd lambdal = Lam.row(l);
  double rho_kl = lambdak.transpose() * Sigma_u * lambdal;
  Eigen::MatrixXd pairs_tab = pairs_table;
  // identify column index in freq table
  // i1: starting index item k
  unsigned int i1 = 0;
  if(k > 1){
    for(unsigned int u = 1; u < k; u++){
      unsigned int cu = c_vec(u);
      //if(silentFLAG == 0)Rcpp::Rcout << "u: " << u << ", cu: "<< cu << "\n";
      i1 += cu * c_vec.segment(0,u).sum();
    }
  }
  
  // i2 starting index from i1 for item l
  unsigned int i2 = 0;
  if(l > 0){
    i2 = c_vec.segment(0,l).sum() * c_vec(k);
  }
  
  if(DFLAG!=1){
    ////////////////////////////
    /* LIKELIHOOD COMPUTATION */
    ////////////////////////////
    for(unsigned int sk = 0; sk < ck; sk ++){
      
      // i3: starting index from i2 for cat sk
      unsigned int i3 = sk * cl;
      
      for(unsigned int sl = 0; sl < cl; sl ++){
        
        // final column index for pairs_tab. Print to check
        unsigned int r = i1 + i2 + i3 + sl;
        
        // read frequency
        unsigned int n_sksl = pairs_table(4, r);
        
        // identify thresholds
        Eigen::VectorXd pi_thresholds = extract_thresholds(tau, c_vec, k, l, sk, sl);
        
        // compute pi
        double pi_sksl = compute_pi(c_vec, pi_thresholds, rho_kl, k, l, sk, sl);
        if(silentFLAG == 0)Rcpp::Rcout << "("<<k<<","<<l<<","<<sk<<","<<sl<<"), rho_kl:"<<rho_kl<<", t_sk:"<< pi_thresholds(0)<<", t_sl:"<< pi_thresholds(1)<<", t_sk-1:"<< pi_thresholds(2)<<", t_sl-1:"<< pi_thresholds(3)<<", pi: "<< pi_sksl<< "\n";
        pairs_tab(5,r) = pi_sksl;
        
        // update ll
        tmp_ll += n_sksl * log(pi_sksl+1e-8);
      }
    }
    
    //////////////////////////
    /* GRADIENT COMPUTATION */
    /////////////////////////
    
    if(gradFLAG == 1){
      unsigned int iter = 0;
      
      /////////////////////////////////////////////////////
      // (k,l)-pair likelihood derivative wrt thresholds //
      /////////////////////////////////////////////////////
      if(silentFLAG == 0)Rcpp::Rcout << "- Gradient wrt thresholds: \n";
      
      // loop: terate over elements of thresholds vector
      for(unsigned int s = 0; s < tau.size(); s++){
        double grs = 0; // temporary location for gradient related to s-th element of tau
        if(silentFLAG == 0)Rcpp::Rcout << "  |_ gradient("<< s<< ")\n";
        
        // List three cases: 1. threshold related to item k, 2. threshold related to item l, 3. threshold non relevant to items couple (k,l)
        if(s >= (c_vec.segment(0, k).sum()) - (k) & s < c_vec.segment(0, k + 1).sum() - (k + 1)){
          // [CASE 1]: threshold related to item k
          
          if(silentFLAG == 0)Rcpp::Rcout << "  |    |_ tau item k:\n";
          unsigned int sk = s - (c_vec.segment(0, k).sum()) + (k);
          
          // i3: starting index from i2 for cat sk and sk+1
          unsigned int i3 = sk * cl;
          unsigned int i3suc = (sk+1) * cl;
          if(silentFLAG == 0)Rcpp::Rcout << "  |    |_ sk: " << sk << ". Summing over categories item l: ";
          
          // iterate over categories of item l
          for(unsigned int sl = 0; sl < cl; sl ++){
            if(silentFLAG == 0)Rcpp::Rcout << " ... cat" << sl ;
            
            // identify pairs_tab column for (sk,sl) and (sk+1, sl)
            unsigned int r = i1 + i2 + i3 + sl;
            unsigned int rsuc = i1 + i2 + i3suc + sl;
            
            // read frequences
            unsigned int n_sksl = pairs_tab(4, r);
            unsigned int n_sksucsl = pairs_tab(4, rsuc);
            
            // read probabilities
            double pi_sksl = pairs_tab(5, r);
            double pi_sksucsl = pairs_tab(5, rsuc);
            
            // identify tau_sk, tau_sl, tau_sl-1
            Eigen::VectorXd pi_thresholds = extract_thresholds(tau, c_vec, k, l, sk, sl);
            double t_sk = pi_thresholds(0); double t_sl = pi_thresholds(1); double t_sk_prev = pi_thresholds(2); double t_sl_prev = pi_thresholds(3);
            
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
          unsigned int sl = s - (c_vec.segment(0, l).sum()) + (l);
          
          if(silentFLAG == 0)Rcpp::Rcout << "  |    |_  sl: " << sl << ". Summing over categories item k: ";
          
          // iterate over categories item k
          for(unsigned int sk = 0; sk < ck; sk ++){
            
            // i3: starting index from i2 for cat sk
            unsigned int i3 = sk * cl;
            
            // identify pairs_tab column for (sk,sl) and (sk, sl + 1)
            unsigned int r = i1 + i2 + i3 + sl;
            unsigned int rsuc = i1 + i2 + i3 + sl + 1;
            
            // read frequences
            unsigned int n_sksl = pairs_tab(4, r);
            unsigned int n_skslsuc = pairs_tab(4, rsuc);
            
            // read probabilities
            double pi_sksl = pairs_tab(5, r);
            double pi_skslsuc = pairs_tab(5, rsuc);
            
            // identify tau_sk, tau_sl, tau_sl-1
            Eigen::VectorXd pi_thresholds = extract_thresholds(tau, c_vec, k, l, sk, sl);
            double t_sk = pi_thresholds(0); double t_sl = pi_thresholds(1); double t_sk_prev = pi_thresholds(2); double t_sl_prev = pi_thresholds(3);
            
            
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
        tmp_gradient(iter) += grs;
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
      for(unsigned int sk = 0; sk < ck; sk ++){
        for(unsigned int sl = 0; sl < cl; sl ++){
          if(silentFLAG == 0)Rcpp::Rcout << "  |_ sk: "<< sk << ", sl: " << sl << ": \n";
          
          // identify pairs_tab column for (sk,sl)
          unsigned int i3 = sk * cl;
          unsigned int r = i1 + i2 + i3 + sl;
          
          // read freq
          unsigned int n_sksl = pairs_tab(4, r);
          
          // read prob
          double pi_sksl = pairs_tab(5, r);
          
          // identify tau_sk, tau_sl, tau_sk-1, tau_sl-1
          Eigen::VectorXd pi_thresholds = extract_thresholds(tau, c_vec, k, l, sk, sl);
          double t_sk = pi_thresholds(0); double t_sl = pi_thresholds(1); double t_sk_prev = pi_thresholds(2); double t_sl_prev = pi_thresholds(3);
          
          
          // phi(t_sk, t_sl; rho_kl)
          double d1 = pbv_rcpp_dbvnorm0( t_sk, t_sl, rho_kl, 0);
          
          // phi(t_sk, t_sl-1; rho_kl)
          double d2 = pbv_rcpp_dbvnorm0( t_sk, t_sl_prev, rho_kl, 0);
          
          // phi(t_sk-1, t_sl; rho_kl)
          double d3 = pbv_rcpp_dbvnorm0( t_sk_prev, t_sl, rho_kl, 0);
          
          // phi(t_sk-1, t_sl-1; rho_kl)
          double d4 = pbv_rcpp_dbvnorm0( t_sk_prev, t_sl_prev, rho_kl, 0);
          
          tmp_kl += (n_sksl/(pi_sksl+1e-8)) * ( d1 - d2 - d3 + d4);
        }
      }
      if(silentFLAG == 0)Rcpp::Rcout << "  |_ tmp_kl:" << tmp_kl << "\n";
      
      ///////////////////////////////////////////////////
      // (k,l)-pair likelihood derivative wrt loadings //
      ///////////////////////////////////////////////////
      if(silentFLAG == 0)Rcpp::Rcout << "\n- Gradient wrt loadings: \n";
      
      // double loop: iterate over elements of loadings matrix
      for(unsigned int j = 0; j < p; j++){
        for(unsigned int v = 0; v < q; v++){
          if(silentFLAG == 0)Rcpp::Rcout << "  |_ visiting lambda_"<< j << v <<":\n";
          
          // elicit three cases: 1. free loading item k, 2. free loading l, 3. other
          if(j == k){
            if(A(j,v)!=0 ){
              if(silentFLAG == 0)Rcpp::Rcout << "  |   |_ item k, free loading:\n";
              Eigen::VectorXd ev(q); ev.fill(0.0); ev(v) = 1;
              double d_rho_kl = ev.transpose() * Sigma_u * lambdal;
              if(silentFLAG == 0)Rcpp::Rcout << "  |   |_ d_rho_kl:" << d_rho_kl << "\n";
              tmp_gradient(iter) += tmp_kl * d_rho_kl;
              if(silentFLAG == 0)Rcpp::Rcout << "  |   |_ Loadings:: " << iter - tau.size() << "/"<< nload <<". Tot:: "<< iter << "/"<< d << "\n";
              
              iter ++;
            }
          }else if (j == l){
            if(A(j,v)!=0 ){
              if(silentFLAG == 0)Rcpp::Rcout << "  |   |_ item l, free loading:\n";
              Eigen::VectorXd ev(q); ev.fill(0.0); ev(v) = 1;
              double d_rho_kl = lambdak.transpose() * Sigma_u * ev;
              if(silentFLAG == 0)Rcpp::Rcout << "  |   |_ d_rho_kl:" << d_rho_kl << "\n";
              if(silentFLAG == 0)Rcpp::Rcout << "  |   |_ Loadings:: " << iter - tau.size() << "/"<< nload <<". Tot:: "<< iter << "/"<< d << "\n";
              tmp_gradient(iter) += tmp_kl * d_rho_kl;
              
              iter ++;
            }
          }else if(A(j,v)!=0){
            
            if(silentFLAG == 0)Rcpp::Rcout << "  |  |_ Loadings:: " << iter - tau.size() << "/"<< nload <<". Tot:: "<< iter << "/"<< d << " [not included]\n";
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
        for(unsigned int v = 1; v < q; v++){
          for(unsigned int  t = 0; t < v; t++){
            Eigen::VectorXd ev(q); ev.fill(0.0); ev(v) = 1;
            Eigen::VectorXd et(q); et.fill(0.0); et(t) = 1;
            double trho = transformed_rhos(iter - tau.size() - nload);
            double drho = 2*exp(2*trho) * pow((exp(2*trho) + 1),-1) * ( 1 - ( exp(2*trho) - 1) * pow((exp(2*trho) + 1),-1) );
            
            // impose symmetric structure
            Eigen::MatrixXd Jvt = ev * et.transpose();
            Eigen::MatrixXd Jtv = et * ev.transpose();
            Eigen::MatrixXd Svt = Jvt + Jtv - Jvt*Jvt;
            
            double d_rho_kl = lambdak.transpose() * (Svt * drho) * lambdal;
            
            //if(silentFLAG == 0)
            tmp_gradient(iter) += tmp_kl * d_rho_kl;
            iter ++;
          }
        }
      }
      
      if(silentFLAG == 0)Rcpp::Rcout << "\n=====> gradient r-th pair:\n" << gradient << "\n";
    }
  }
  
  ll += tmp_ll;
  gradient += tmp_gradient;

}

#endif