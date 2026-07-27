#include "arg.h"
#include "common.h"
#include "log.h"
#include "llama.h"
#include "ggml.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cctype>
#include <clocale>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace fs = std::filesystem;

struct capture_state {
    fs::path out_dir;
    std::unordered_set<std::string> printed_names;
    std::unordered_map<std::string, size_t> chunks;
};

static std::string safe_name(const std::string & name) {
    std::string out;
    out.reserve(name.size());
    for (unsigned char c : name) {
        out.push_back(std::isalnum(c) || c == '-' || c == '_' ? char(c) : '_');
    }
    return out;
}

static bool is_target_name(const std::string & name) {
    static const char * targets[] = {
        "ffn_norm-7",
        // ffn_moe_topk-7 is a non-contiguous view in llama.cpp. The argsort
        // node is contiguous and its first eight columns are the same IDs.
        "ffn_moe_argsort-7",
        "ffn_moe_weights-7",
        "ffn_moe_gate-7",
        "ffn_moe_up-7",
        "ffn_moe_swiglu-7",
        "ffn_moe_down-7",
        "ffn_moe_out-7",
    };
    for (const char * target : targets) {
        if (name.find(target) != std::string::npos) {
            return true;
        }
    }
    return false;
}

static float convert_value(const uint8_t * raw, ggml_type type, size_t index) {
    switch (type) {
        case GGML_TYPE_F32:  return reinterpret_cast<const float *>(raw)[index];
        case GGML_TYPE_F16:  return ggml_fp16_to_fp32(reinterpret_cast<const ggml_fp16_t *>(raw)[index]);
        case GGML_TYPE_BF16: return ggml_bf16_to_fp32(reinterpret_cast<const ggml_bf16_t *>(raw)[index]);
        case GGML_TYPE_I32:  return float(reinterpret_cast<const int32_t *>(raw)[index]);
        case GGML_TYPE_I16:  return float(reinterpret_cast<const int16_t *>(raw)[index]);
        case GGML_TYPE_I8:   return float(reinterpret_cast<const int8_t *>(raw)[index]);
        default:              return 0.0f;
    }
}

static bool supported_type(ggml_type type) {
    return type == GGML_TYPE_F32 || type == GGML_TYPE_F16 || type == GGML_TYPE_BF16 ||
           type == GGML_TYPE_I32 || type == GGML_TYPE_I16 || type == GGML_TYPE_I8;
}

static bool capture_cb(struct ggml_tensor * t, bool ask, void * user_data) {
    auto * state = static_cast<capture_state *>(user_data);
    const std::string name = t->name;
    const bool target = is_target_name(name);

    if (ask) {
        if ((name.find("ffn_moe") != std::string::npos || name.find("ffn_norm") != std::string::npos) &&
            name.find("-7") != std::string::npos && state->printed_names.insert(name).second) {
            LOG_INF("capture-discovery: name=%s type=%s shape={%lld,%lld,%lld,%lld} elements=%lld target=%d\n",
                    name.c_str(), ggml_type_name(t->type),
                    (long long) t->ne[0], (long long) t->ne[1],
                    (long long) t->ne[2], (long long) t->ne[3],
                    (long long) ggml_nelements(t), target ? 1 : 0);
        }
        return target;
    }

    if (!target) {
        return true;
    }
    if (!ggml_is_contiguous(t)) {
        LOG_ERR("capture: target tensor is not contiguous: %s\n", name.c_str());
        return true;
    }
    if (!supported_type(t->type)) {
        LOG_ERR("capture: unsupported type %s for %s\n", ggml_type_name(t->type), name.c_str());
        return true;
    }

    const size_t elements = size_t(ggml_nelements(t));
    const size_t bytes = ggml_nbytes(t);
    std::vector<uint8_t> raw(bytes);
    ggml_backend_tensor_get(t, raw.data(), 0, bytes);
    std::vector<float> values(elements);
    for (size_t i = 0; i < elements; ++i) {
        values[i] = convert_value(raw.data(), t->type, i);
    }

    fs::create_directories(state->out_dir);
    const std::string base = safe_name(name);
    const fs::path data_path = state->out_dir / (base + ".f32");
    const fs::path meta_path = state->out_dir / (base + ".jsonl");
    {
        std::ofstream out(data_path, std::ios::binary | std::ios::app);
        out.write(reinterpret_cast<const char *>(values.data()), std::streamsize(values.size() * sizeof(float)));
        if (!out) {
            LOG_ERR("capture: failed writing %s\n", data_path.string().c_str());
            return true;
        }
    }
    const size_t chunk = state->chunks[base]++;
    {
        std::ofstream meta(meta_path, std::ios::app);
        meta << "{\"chunk\":" << chunk
             << ",\"name\":\"" << name << "\""
             << ",\"type\":\"" << ggml_type_name(t->type) << "\""
             << ",\"ne\":[" << t->ne[0] << ',' << t->ne[1] << ',' << t->ne[2] << ',' << t->ne[3] << ']'
             << ",\"elements\":" << elements << "}\n";
    }
    LOG_INF("capture: wrote %s chunk=%zu elements=%zu\n", name.c_str(), chunk, elements);
    return true;
}

static bool run(llama_context * ctx, const common_params & params) {
    const llama_model * model = llama_get_model(ctx);
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const bool add_bos = llama_vocab_get_add_bos(vocab);
    std::vector<llama_token> tokens = common_tokenize(ctx, params.prompt, add_bos, true);
    if (tokens.empty()) {
        LOG_ERR("capture: no input tokens\n");
        return false;
    }
    size_t max_tokens = 768;
    if (const char * env = std::getenv("CAPTURE_MAX_TOKENS")) {
        max_tokens = std::max<size_t>(1, std::strtoull(env, nullptr, 10));
    }
    if (tokens.size() > max_tokens) {
        tokens.resize(max_tokens);
    }
    LOG_INF("capture: evaluating %zu tokens\n", tokens.size());
    if (llama_decode(ctx, llama_batch_get_one(tokens.data(), int32_t(tokens.size())))) {
        LOG_ERR("capture: llama_decode failed\n");
        return false;
    }
    return true;
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");
    common_params params;
    common_init();
    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }
    capture_state state;
    state.out_dir = std::getenv("CAPTURE_DIR") ? std::getenv("CAPTURE_DIR") : "expert_activation_capture";

    llama_backend_init();
    llama_numa_init(params.numa);
    params.cb_eval = capture_cb;
    params.cb_eval_user_data = &state;
    params.warmup = false;

    auto llama_init = common_init_from_params(params);
    auto * model = llama_init->model();
    auto * ctx = llama_init->context();
    if (model == nullptr || ctx == nullptr) {
        LOG_ERR("capture: model/context initialization failed\n");
        return 1;
    }
    const bool ok = run(ctx, params);
    llama_perf_context_print(ctx);
    llama_backend_free();
    return ok ? 0 : 1;
}
