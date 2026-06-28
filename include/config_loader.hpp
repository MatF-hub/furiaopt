#pragma once

#include <fstream>
#include <stdexcept>
#include <string>
#include <memory>

#include <nlohmann/json.hpp>

#include <spdlog/logger.h>
#include <spdlog/sinks/null_sink.h>
#include <spdlog/sinks/basic_file_sink.h>

#include "solver_config.hpp"

namespace furiaopt
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

inline UnconstrainedSolverOptions load_solver_options(
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

    UnconstrainedSolverOptions opt;

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

    // Create logger based on log_file_folder_path
    auto log_folder = j.value("log_file_folder_path", std::string("logs/"));
    if (!log_folder.empty()) {
        auto file_logger = spdlog::basic_logger_mt<spdlog::synchronous_factory>(
            "file_logger", log_folder + "/solver.log", true);
        file_logger->set_level(spdlog::level::info);
        file_logger->set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%l] %v");
        opt.logger = file_logger;
    } else {
        // Use null sink logger
        opt.logger = std::make_shared<spdlog::logger>("null",
            std::make_shared<spdlog::sinks::null_sink_mt>());
    }

    return opt;
}

} // namespace furiaopt