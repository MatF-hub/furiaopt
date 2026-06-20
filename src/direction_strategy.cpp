#include "direction_strategy.hpp"

namespace furiaoptimizer{

Eigen::VectorXd ExactHessianDirection::getDirection(const Eigen::VectorXd& gradient, const Eigen::VectorXd& x_i)
{
    //Try LLT decomposition first, requires positive definite Hessian
    Eigen::MatrixXd hessian = problem_.get().hessian_func.value()(x_i);
    llt_.compute(hessian);
    if (llt_.info() == Eigen::Success) {
        return llt_.solve(-gradient);
    }
    //If LLT fails, try LDLT decomposition, can handle both positive and negative semi definite Hessian
    ldlt_.compute(hessian);
    if (ldlt_.info() == Eigen::Success) {
        return ldlt_.solve(-gradient);
    }
    //If both decompositions fail, fallback to gradient descent direction
    return -gradient;
};

Eigen::VectorXd GaussNewtonDirection::getDirection(const Eigen::VectorXd& gradient, const Eigen::VectorXd& x_i)
{
        Eigen::MatrixXd approximate_hessian = hessian_approximation_.getApproximateHessian(x_i);
        //Try LLT decomposition first, requires positive definite Hessian
        llt_.compute(approximate_hessian);
        if (llt_.info() == Eigen::Success) {
            return llt_.solve(-gradient); // Notice that gradient == 2 * gradient_residual_vector * residual_vector
        }
        //If LLT fails, try LDLT decomposition, can handle both positive and negative semi definite Hessian
        ldlt_.compute(approximate_hessian);
        if (ldlt_.info() == Eigen::Success) {
            return ldlt_.solve(-gradient); // Notice that gradient == 2 * gradient_residual_vector * residual_vector
        }
    
    //If both decompositions fail, fallback to gradient descent direction
    return - gradient; // Notice that gradient == 2 * gradient_residual_vector * residual_vector
};

Eigen::VectorXd BFGSDirection::getDirection(const Eigen::VectorXd& g_k_plus_1, const Eigen::VectorXd& x_k_plus_1)
{

    Eigen::MatrixXd H_k = hessian_approximation_.getApproximateHessian(g_k_plus_1, x_k_plus_1);

    llt_.compute(H_k);
    if (llt_.info() == Eigen::Success) {
        return llt_.solve(-g_k_plus_1); // Notice that gradient == gradient_residual_vector * residual_vector
    }
    //If LLT fails, try LDLT decomposition, can handle both positive and negative semi definite Hessian
    ldlt_.compute(H_k);
    if (ldlt_.info() == Eigen::Success) {
        return ldlt_.solve(-g_k_plus_1); // Notice that gradient == gradient_residual_vector * residual_vector
    }

    return -g_k_plus_1; // If both decompositions fail, fallback to gradient descent direction
};

Eigen::MatrixXd GaussNewtonHessianApproximation::getApproximateHessian(const Eigen::VectorXd& x_i)
{
    Eigen::MatrixXd gradient_residual_vector = problem_.get().gradient_residual_func.value()(x_i);
    Eigen::VectorXd residual_vector = problem_.get().residual_func(x_i);

    Eigen::MatrixXd JtJ = gradient_residual_vector * gradient_residual_vector.transpose(); // n×n
    Eigen::MatrixXd approximate_hessian =  2 * Jt00000J + sigma_ * Eigen::MatrixXd::Identity(JtJ.rows(), JtJ.rows());
    return approximate_hessian;
};

Eigen::MatrixXd BFGSHessianApproximation::getApproximateHessian(const Eigen::VectorXd& g_k_plus_1, const Eigen::VectorXd& x_k_plus_1)
{
    if (!initialized_) {
        x_k_ = x_k_plus_1;
        g_k_ = g_k_plus_1;
        initialized_ = true;
        return H_k_; // Initial direction is just the negative gradient, returning identity matrix
    }

    Eigen::VectorXd s = x_k_plus_1 - x_k_;
    Eigen::VectorXd y = g_k_plus_1 - g_k_;

    //Implement powell's damping strategy to ensure positive definiteness of the Hessian approximation
    double sTHs = s.transpose()*(H_k_ * s);
    double gamma_sTHs = gamma_ * sTHs;
    double yTs = y.transpose() * s;
    double sTy = yTs;
    if (yTs <= gamma_sTHs){
        y = y + (gamma_sTHs - sTy)*(H_k_*s - y)/(sTHs-sTy);
        sTy = y.dot(s);  // recompute after damping
    }

    H_k_ = H_k_ + (y * y.transpose()) / (sTy) - (H_k_ * s * s.transpose() * H_k_) / sTHs;
    x_k_ = x_k_plus_1;
    g_k_ = g_k_plus_1;
    return H_k_;
};

}