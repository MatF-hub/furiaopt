#pragma once

#include <fstream>
#include <stdexcept>
#include <string>

#include <nlohmann/json.hpp>

#include "solver_config.hpp"

namespace furiaoptimizer
{

using json = nlohmann::json;

//------------------------------------------
// Enum parsing helpers
//------------------------------------------

inline DirectionMethod parse_direction_method(
    const std::string& s)
{
    if (s == "GradientDescent")
        return DirectionMethod::GradientDescent;

    if (s == "BFGS")
        return DirectionMethod::BFGS;

    if (s == "ExactNewton")
        return DirectionMethod::ExactNewton;

    throw std::runtime_error(
        "Unknown direction_method: " + s);
}

inline GlobalizationMethod parse_globalization_method(
    const std::string& s)
{
    if (s == "LineSearch")
        return GlobalizationMethod::LineSearch;

    if (s == "TrustRegion")
        return GlobalizationMethod::TrustRegion;

    throw std::runtime_error(
        "Unknown globalization_method: " + s);
}

//------------------------------------------
// Load config file
//------------------------------------------

inline SolverOptions load_solver_options(
    const std::string& filepath)
{
    std::ifstream file(filepath);

    if (!file)
    {
        throw std::runtime_error(
            "Cannot open config file: " + filepath);
    }

    json j;
    file >> j;

    SolverOptions opt;

    opt.max_iter =
        j.value("max_iter", 100);

    opt.gradient_tolerance =
        j.value("gradient_tolerance", 1e-8);

    opt.step_tolerance =
        j.value("step_tolerance", 1e-12);

    opt.function_tolerance =
        j.value("function_tolerance", 1e-12);

    opt.direction_method =
        parse_direction_method(
            j.value("direction_method",
                    "GradientDescent"));

    opt.globalization_method =
        parse_globalization_method(
            j.value("globalization_method",
                    "LineSearch"));

    opt.log_file_folder_path =
        j.value("log_file_folder_path", "logs/");

    return opt;
}

} // namespace furiaoptimizer