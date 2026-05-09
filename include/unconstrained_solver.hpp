#include <Eigen/Dense>


namespace furiaoptimizer{

struct SolverOptions{
    int max_iter = 100;
    double gradient_tolerance = 1e-6;
    double step_tolerance = 1e-6;
    double function_tolerance = 1e-6;
};

struct SolverSummary {
    int iterations = 0;
    double initial_cost = 0.0;
    double final_cost = 0.0;
    bool converged = false;
};

using CostFunc = std::function<double(const Eigen::VectorXd& params, const Eigen::VectorXd& x)>;

class Solver{

    SolverOptions options_;
    SolverSummary summary_;

    public:

    void SetOptions(SolverOptions options) { options_ = std::move(options); };

    void Solve(const CostFunc& f, const Eigen::VectorXd& params, Eigen::VectorXd& x);
};

}
