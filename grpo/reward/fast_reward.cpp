#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <string>
#include <vector>
#include <regex>
#include <cmath>
#include <algorithm>

namespace py = pybind11;

std::string extract_answer(const std::string& text) {
    std::regex gsm8k(R"(####\s*(-?[\d,]+\.?\d*))");
    std::smatch m;
    if (std::regex_search(text, m, gsm8k)) {
        std::string r = m[1].str();
        r.erase(std::remove(r.begin(), r.end(), ','), r.end());
        return r;
    }
    std::regex ans_pat(R"(answer\s+(?:is|:)\s*(-?[\d,]+\.?\d*))", std::regex::icase);
    if (std::regex_search(text, m, ans_pat)) {
        std::string r = m[1].str();
        r.erase(std::remove(r.begin(), r.end(), ','), r.end());
        return r;
    }
    std::regex num(R"(-?[\d,]+\.?\d*)");
    std::string last;
    auto it = std::sregex_iterator(text.begin(), text.end(), num);
    for (; it != std::sregex_iterator(); ++it) {
        last = (*it).str();
        last.erase(std::remove(last.begin(), last.end(), ','), last.end());
    }
    return last;
}

std::string normalize_answer(const std::string& answer) {
    if (answer.empty()) return answer;
    try {
        double val = std::stod(answer);
        long long ival = (long long)val;
        if (val == std::floor(val) && std::abs(val) < 1e15)
            return std::to_string(ival);
        char buf[64];
        snprintf(buf, sizeof(buf), "%.6f", val);
        std::string s(buf);
        s.erase(s.find_last_not_of('0') + 1);
        if (s.back() == '.') s.pop_back();
        return s;
    } catch (...) {
        std::string lower = answer;
        std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
        return lower;
    }
}

float binary_reward(const std::string& completion, const std::string& ground_truth) {
    std::string predicted = extract_answer(completion);
    std::string correct = extract_answer(ground_truth);
    if (predicted.empty() || correct.empty()) return 0;
    return (normalize_answer(predicted) == normalize_answer(correct)) ? 1 : 0;
}

std::vector<float> batch_binary_reward(
    const std::vector<std::string>& completions,
    const std::string& ground_truth
) {
    std::vector<float> rewards;
    rewards.reserve(completions.size());
    for (const auto& c : completions)
        rewards.push_back(binary_reward(c, ground_truth));
    return rewards;
}

PYBIND11_MODULE(fast_reward, m) {
    m.doc() = "Fast C++ reward functions for GRPO math training";
    m.def("extract_answer", &extract_answer);
    m.def("normalize_answer", &normalize_answer);
    m.def("binary_reward", &binary_reward);
    m.def("batch_binary_reward", &batch_binary_reward);
}
