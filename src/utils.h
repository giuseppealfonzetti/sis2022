#ifndef utils_H
#define utils_H

/* Function to print vector to rsession in a tidy way*/
void print_vec(Eigen::VectorXd vector){
    Rcpp::Rcout<<"(";
    for(int i = 0; i < vector.size(); i++){
        Rcpp::Rcout << vector(i);
        if(i != vector.size()-1) Rcpp::Rcout<<", ";
    }
    Rcpp::Rcout<<")\n";
}

/* LOADING MATRIX */
// Get loading matrix from theta
Eigen::MatrixXd get_Lam(Eigen::Map<Eigen::MatrixXd> A,
                        Eigen::Map<Eigen::VectorXd> c_vec,
                        const Eigen::VectorXd &theta
){
    unsigned int p = A.rows(); // number of items
    unsigned int q = A.cols(); // number of latents
    unsigned int d = theta.size(); // number of parameters
    unsigned int c = c_vec.sum();

    Eigen::VectorXd lambda = theta.segment(c-p, d-c+p-q*(q-1)/2);
    Eigen::MatrixXd Lam = A;
    unsigned int iter = 0;
    for(unsigned int h = 0; h < q; h++){
        for(unsigned int j = 0; j < p; j++){
            if(A(j, h) != 0.0)
            {
                Lam(j, h) = lambda(iter) ;
                iter ++;
            };
        };
    }

    return Lam;
}

/* LATENT COVARIANCE */
// Get latent correlation matrix from theta
Eigen::MatrixXd get_Sigma_u(Eigen::Map<Eigen::MatrixXd> A,
                            const Eigen::VectorXd &theta
){
    unsigned int q = A.cols(); // number of latents
    unsigned int d = theta.size(); // number of parameters

    Eigen::VectorXd transformed_rhos = theta.segment(d-q*(q-1)/2, q*(q-1)/2);
    Eigen::VectorXd rhos = ((Eigen::VectorXd(2*transformed_rhos)).array().exp() - 1)/
        ((Eigen::VectorXd(2*transformed_rhos)).array().exp() + 1);  // reparametrize rhos
    // latent variable covariance matrix
    Eigen::MatrixXd Sigma_u(q,q); Sigma_u.setIdentity();
    unsigned int iter = 0;
    for(unsigned int j = 0; j < q; j++){
        for(unsigned int h = 0; h < q; h++){
            if(j > h)
            {
                Sigma_u(j, h) = rhos(iter);
                Sigma_u(h, j) = rhos(iter);
                iter ++;
            }
        }
    }


    return Sigma_u;
}

/* THRESHOLDS */
// Get thresholds parameters from theta
Eigen::VectorXd get_tau(const Eigen::VectorXd &theta, Eigen::VectorXd c_vec)
{
    unsigned int c = c_vec.sum();
    unsigned int p = c_vec.size();
    Eigen::VectorXd tau = theta.segment(0,c-p);

    return tau;
}

/* EXTRACT PI THRESHOLDS */
// Extract from tau the trhesholds related to pi_sksl
Eigen::VectorXd extract_thresholds(const Eigen::VectorXd tau,
                                   const Eigen::VectorXd c_vec,
                                   const unsigned int k,
                                   const unsigned int l,
                                   const unsigned int sk,
                                   const unsigned int sl){

    unsigned int ck = c_vec(k);
    unsigned int cl = c_vec(l);

    // identify tau_sk, tau_sl, tau_sk-1, tau_sl-1
    unsigned int sk_tau_index = c_vec.segment(0, k).sum() - (k) + sk;   // index tau_sk in tau vector
    unsigned int sl_tau_index = c_vec.segment(0, l).sum() - (l) + sl;   // index tau_sl in tau vector
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

    Eigen::VectorXd pi_thresholds(4);
    pi_thresholds << t_sk, t_sl, t_sk_prev, t_sl_prev;
    return pi_thresholds;
}

/* INTERSECTION BETWEEN VECTORS */
// Used if stochastic sampling is stratified
std::vector<int> intersection(std::vector<int> &v1, std::vector<int> &v2){
    std::vector<int> v3;

    std::sort(v1.begin(), v1.end());
    std::sort(v2.begin(), v2.end());

    std::set_intersection(v1.begin(),v1.end(),
                          v2.begin(),v2.end(),
                          back_inserter(v3));
    return v3;
}

// [[Rcpp::export]]
Eigen::MatrixXd matSum(Eigen::MatrixXd A, Eigen::MatrixXd B){
    Eigen::MatrixXd S = Eigen::MatrixXd::Zero(A.rows(), A.cols());
    
    for(unsigned int rw = 0; rw < A.rows(); rw++){
        for(unsigned int cl = 0; cl < A.cols(); cl++){
            double a = A(rw, cl);
            double b = B(rw, cl);
            double s = a + b;
            S(rw, cl) = s;
        } 
    }
    
    return S;
}
#endif
