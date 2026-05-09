#include <Eigen/Dense>


namespace furiaoptimizer{

enum class DirectionMethod {
    GradientDescent,
    GaussNewton,
    BFGS,
    ExactNewton
};

enum class GlobalizationMethod {
    LineSearch,
    TrustRegion
};
struct SolverOptions{
    DirectionMethod direction_method = DirectionMethod::GradientDescent;
    GlobalizationMethod globalization_method = GlobalizationMethod::LineSearch;
    int max_iter = 5e4;
    double gradient_tolerance = 1e-10;
    double step_tolerance = 1e-10;
    double function_tolerance = 1e-10;
};

struct SolverSummary {
    int iterations = 0;
    double initial_cost = 0.0;
    double final_cost = 0.0;
    bool converged = false;
};

struct Result {
    Eigen::VectorXd x;
    SolverSummary summary;
};

using CostFunc = std::function<double(const Eigen::VectorXd& params, const Eigen::VectorXd& x)>;
using GradientFunc = std::function<Eigen::VectorXd(const Eigen::VectorXd& params, const Eigen::VectorXd& x)>;
using HessianFunc = std::function<Eigen::MatrixXd(const Eigen::VectorXd& params, const Eigen::VectorXd& x)>;
class Solver{

    SolverOptions options_;

public:

    Solver() = default;

    void set_options(SolverOptions options) { options_ = std::move(options); };

    Result solve(const CostFunc& f, const Eigen::VectorXd& params, const Eigen::VectorXd& x);
    Result solve(const CostFunc& f, const GradientFunc& g, const Eigen::VectorXd& params, const Eigen::VectorXd& x);
    Result solve(const CostFunc& f, const GradientFunc& g, const HessianFunc& h, const Eigen::VectorXd& params, const Eigen::VectorXd& x);

    Eigen::VectorXd compute_direction(const Eigen::VectorXd& g);
    Eigen::VectorXd compute_direction(const Eigen::VectorXd& g, const Eigen::MatrixXd& h);

    double compute_step_length(const CostFunc& f, const Eigen::VectorXd& g, const Eigen::VectorXd& params, const Eigen::VectorXd& x, const Eigen::VectorXd& direction);
};

}
