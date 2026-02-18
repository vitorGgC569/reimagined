#include "../include/jamba.h"
#include "../include/nsos_arena.h"
#include "../include/nsos_math.h"
#include "../include/nsos_serializer.h"
#include "../include/nsos_context.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <omp.h>

namespace nsos {

JambaModel::JambaModel(int nl, int dm, int vs, Device dev) 
    : num_layers(nl), d_model(dm), vocab_size(vs), device(dev) {
    embedding = std::make_unique<Embedding>(vs, dm);
    for(int i=0; i<nl; ++i) layers.push_back(std::make_unique<JambaBlock>(dm, i%8==0, i%2==1, i%8==4, i, nl));
    value_head = std::make_unique<BitLinear>(dm, vs);
}

void JambaModel::save(const std::string &fn) { ModelSerializer::save(this, fn); }
void JambaModel::load(const std::string &fn) { ModelSerializer::load(this, fn); }

Tensor JambaModel::forward(const Tensor &x, Context *ctx) {
    Tensor h = x;
    for(auto& l : layers) h = l->forward(h, ctx);
    return value_head->forward(h.rmsnorm());
}

Tensor JambaModel::forward(const Tensor &x) { return forward(x, nullptr); }

void JambaModel::to(Device dev) {
    device = dev;
    embedding->to(dev);
    for(auto& l : layers) l->to(dev);
    value_head->to(dev);
}

std::vector<Parameter*> JambaModel::parameters() {
    std::vector<Parameter*> p;
    auto ep = embedding->parameters(); p.insert(p.end(), ep.begin(), ep.end());
    for(auto& l : layers) { auto lp = l->parameters(); p.insert(p.end(), lp.begin(), lp.end()); }
    auto vp = value_head->parameters(); p.insert(p.end(), vp.begin(), vp.end());
    return p;
}

Tensor JambaModel::forward_ids(const std::vector<int>& ids, Context* ctx) {
    Tensor x = embedding->forward(ids);
    return forward(x, ctx);
}
Tensor JambaModel::forward_trunk(const std::vector<int>& ids, Context* ctx) {
    Tensor x = embedding->forward(ids);
    for(auto& l : layers) x = l->forward(x, ctx);
    return x.rmsnorm();
}
Tensor JambaModel::forward_embedding(const Tensor& x, Context* ctx) { return forward(x, ctx); }
Tensor JambaModel::reason(const Tensor& x, int n) { return forward(x); }
Tensor JambaModel::forward_thought(const Tensor& x, int s) { return forward(x); }
void JambaModel::run_reasoning_loop(int i) {}
void JambaModel::backward_external(const Tensor& g, Context& c) { backward(g, c); }
void JambaModel::backward_embedding(const Tensor& g, Context& c) {}
void JambaModel::backward(const Tensor& g, Context& c) { 
    Tensor dy = value_head->backward(g);
    for(int i=(int)layers.size()-1; i>=0; --i) dy = layers[i]->backward(dy, &c);
}
void JambaModel::reset_session() {}
void JambaModel::set_hamiltonian_mode(bool e) {}
void JambaModel::session_adapt(const Tensor& x, const Tensor& y) {}
Tensor JambaModel::run_simd_inference(const Tensor& x, int s) { return forward(x); }

JambaBlock::JambaBlock(int dm, bool ia, bool im, bool it, int li, int tl)
    : d_model(dm), is_attention(ia), is_moe(im), is_ttt(it), layer_idx(li), total_layers(tl) {
    if(it) ttt_layer = std::make_unique<TTTLayer>(dm, 1024);
    else if(ia) attn_layer = std::make_unique<Attention>(dm, 32);
    else mamba_layer = std::make_unique<Mamba2SSD>(dm, 128, 8, MambaConfig());
    if(im) {
        router = std::make_unique<MoERouter>(dm, 256, 8);
        num_experts = 256;
        for(int j=0; j<256; ++j) {
            expert_gate_up.push_back(std::make_unique<BitLinear>(dm, dm*8));
            expert_down.push_back(std::make_unique<BitLinear>(dm*4, dm));
        }
    } else {
        ffn_gate_up = std::make_unique<BitLinear>(dm, dm*8);
        ffn_down = std::make_unique<BitLinear>(dm*4, dm);
    }
}
JambaBlock::~JambaBlock() {}
Tensor JambaBlock::forward(const Tensor& x, Context* ctx) {
    Tensor xn = x.rmsnorm();
    Tensor h;
    if(is_ttt) h = ttt_layer->forward(xn);
    else if(is_attention) h = attn_layer->forward(xn, ctx);
    else h = mamba_layer->forward(xn, ctx);
    Tensor r1 = x.add(h);
    Tensor xn2 = r1.rmsnorm();
    if(is_moe) return r1.add(forward_moe(xn2, ctx, "L"+std::to_string(layer_idx)));
    Tensor gu = ffn_gate_up->forward(xn2);
    return r1.add(ffn_down->forward(gu));
}
Tensor JambaBlock::forward_moe(const Tensor& x, Context* ctx, const std::string& ln) { return x; }
Tensor JambaBlock::backward(const Tensor& dy, Context* ctx) { return dy; }
void JambaBlock::reset() {}
void JambaBlock::to(Device dev) {}
std::vector<Parameter*> JambaBlock::parameters() { return {}; }

Attention::Attention(int d, int n, int l) : d_model(d), n_heads(n), n_latents(l) {}
Tensor Attention::forward(const Tensor& i, Context* c) { return i; }
Tensor Attention::backward(const Tensor& d, Context* c) { return d; }
void Attention::to(Device d) {}
std::vector<Parameter*> Attention::parameters() { return {}; }

MoERouter::MoERouter(int d, int n, int k) : num_experts(n), top_k(k) {}
std::pair<Tensor, Tensor> MoERouter::forward(const Tensor& x) { return {x, x}; }
Tensor MoERouter::backward(const Tensor& g) { return g; }
void MoERouter::to(Device d) {}
std::vector<Parameter*> MoERouter::parameters() { return {}; }
float MoERouter::compute_aux_loss() { return 0.0f; }

} // namespace nsos
