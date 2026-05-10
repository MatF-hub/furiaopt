#include <Eigen/Dense>
#include "solver_config.hpp"

namespace furiaoptimizer{

using CostFunc = std::function<double(const Eigen::VectorXd& params, const Eigen::VectorXd& x)>;
using GradientFunc = std::function<Eigen::VectorXd(const Eigen::VectorXd& params, const Eigen::VectorXd& x)>;
using HessianFunc = std::function<Eigen::MatrixXd(const Eigen::VectorXd& params, const Eigen::VectorXd& x)>;
class Solver{

    SolverOptions options_;

public:

    Solver(const SolverOptions& options) : options_(options) {}

    void set_options(SolverOptions options) { options_ = std::move(options); };

    Result solve(const CostFunc& f, const Eigen::VectorXd& params, const Eigen::VectorXd& x);
    Result solve(const CostFunc& f, const GradientFunc& g, const Eigen::VectorXd& params, const Eigen::VectorXd& x);
    Result solve(const CostFunc& f, const GradientFunc& g, const HessianFunc& h, const Eigen::VectorXd& params, const Eigen::VectorXd& x);

    Eigen::VectorXd compute_direction(const Eigen::VectorXd& g);
    Eigen::VectorXd compute_direction(const Eigen::VectorXd& g, const Eigen::MatrixXd& h);

    double compute_step_length(const CostFunc& f, const Eigen::VectorXd& g, const Eigen::VectorXd& params, const Eigen::VectorXd& x, const Eigen::VectorXd& direction);
};

}
