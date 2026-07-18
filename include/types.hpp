#pragma once

#include <Eigen/Dense>
#include <functional>
//This type aliases are used to define the function signatures for the various functions used in the optimization problem.
// n = number of variables in optimization problem
// p = number of equality constraints
// q = number of inequality constraints
namespace furiaopt {

//CostFunc : R^(nx1) -> R
using CostFunc = std::function<double(const Eigen::VectorXd& x)>;
//GradientFunc : R^(nx1) -> R^(nx1)
using GradientFunc = std::function<Eigen::VectorXd(const Eigen::VectorXd& x)>;
//HessianFunc : R^(nx1) -> R^(nxn)
using HessianFunc = std::function<Eigen::MatrixXd(const Eigen::VectorXd& x)>;
//ResidualFunc: R^(nx1) -> R^(mx1), when m > n the system is overdetermined, when m < n the system is underdetermined
using ResidualFunc = std::function<Eigen::VectorXd(const Eigen::VectorXd& x)>;
//GradientResidualFunc: R^(nx1) -> R^(nxm)
using GradientResidualFunc = std::function<Eigen::MatrixXd(const Eigen::VectorXd& x)>;
//EqualityConstraintFunc: R^(nx1) -> R^(px1)
using EqualityConstraintFunc = std::function<Eigen::VectorXd(const Eigen::VectorXd& x)>;
//GradientEqualityConstraintFunc: R^(nx1) -> R^(nxp)
using GradientEqualityConstraintFunc = std::function<Eigen::MatrixXd(const Eigen::VectorXd& x)>;
//InequalityConstraintFunc: R^(nx1) -> R^(qx1)
using InequalityConstraintFunc = std::function<Eigen::VectorXd(const Eigen::VectorXd& x)>;
//GradientInequalityConstraintFunc: R^(nx1) -> R^(nxq)
using GradientInequalityConstraintFunc = std::function<Eigen::MatrixXd(const Eigen::VectorXd& x)>;
}