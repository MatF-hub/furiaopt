#pragma once

#include <Eigen/Dense>
#include <functional>

using CostFunc = std::function<double(const Eigen::VectorXd& params, const Eigen::VectorXd& x)>;
using GradientFunc = std::function<Eigen::VectorXd(const Eigen::VectorXd& params, const Eigen::VectorXd& x)>;
using HessianFunc = std::function<Eigen::MatrixXd(const Eigen::VectorXd& params, const Eigen::VectorXd& x)>;
using ResidualFunc = std::function<Eigen::VectorXd(const Eigen::VectorXd& p, const Eigen::VectorXd& x)>;
using GradientResidualFunc = std::function<Eigen::MatrixXd(const Eigen::VectorXd& p, const Eigen::VectorXd& x)>;