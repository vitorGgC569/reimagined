#pragma once

#include <memory>
#include <string>
#include <vector>
#include "pantheon/physics/gradient_matching.hpp"
#include "pantheon/physics/information_bottleneck.hpp"
#include "pantheon/physics/topology.hpp"
#include "pantheon/physics/neural_ode.hpp"
#include "pantheon/structure/contrastive.hpp"
#include "pantheon/structure/relational.hpp"
#include "pantheon/structure/msdcrd.hpp"
#include "pantheon/structure/attention.hpp"
#include "pantheon/structure/category_theory.hpp"
#include "pantheon/response/logit_distillation.hpp"
#include "pantheon/cognition/chain_of_thought.hpp"
#include "pantheon/cognition/symbolic.hpp"
#include "pantheon/cognition/theory_of_mind.hpp"
#include "pantheon/frontier/meta.hpp"
#include "pantheon/frontier/causal.hpp"
#include "pantheon/frontier/quantum.hpp"
#include "pantheon/frontier/spiking.hpp"
#include "pantheon/frontier/physical.hpp"
#include "pantheon/frontier/privacy.hpp"
#include "pantheon/social/game_theory.hpp"
#include "pantheon/social/swarm.hpp"

// Forward declaration
namespace uhk {
    namespace runtime {
        class UniversalKernelRuntime;
    }
}

namespace pantheon {

    class PantheonEngine {
    public:
        PantheonEngine();
        ~PantheonEngine();

        void initialize();
        void shutdown();

        // KernelOpen Interaction
        void submit_dummy_task();
        float get_throughput() const;
        std::string get_status() const;

        // Part 1: Response & Structure
        float compute_contrastive_loss(const std::vector<float>& s, const std::vector<float>& t, int batch, int dim);
        float compute_relational_loss(const std::vector<float>& s, const std::vector<float>& t, int batch, int dim);
        float compute_msdcrd_loss(const std::vector<float>& s, const std::vector<float>& t, int b, int c, int h, int w, int ph, int pw);
        float compute_cam_loss(const std::vector<float>& s_f, const std::vector<float>& s_w, const std::vector<float>& t_f, const std::vector<float>& t_w, int b, int c, int h, int w, int cls, int tgt);
        float compute_ntce_loss(const std::vector<float>& s_l, const std::vector<float>& t_l, int tgt, float beta);
        float compute_optimal_transport_loss(const std::vector<float>& s_l, const std::vector<float>& t_l);

        // Part 2: Cognition
        float compute_chunk_wise_loss(const std::vector<float>& s, const std::vector<float>& t, int chunk);
        std::vector<int> adjust_granularity(const std::vector<int>& steps, float competence);
        float compute_symbolic_loss(const std::vector<float>& output, float expected_sum);

        // Part 3: Physics
        float compute_gradient_loss(const std::vector<float>& t, const std::vector<float>& s);
        float compute_ib_loss(const std::vector<float>& means, const std::vector<float>& log_vars);
        float compute_topology_loss(const std::vector<float>& s_dist, const std::vector<float>& t_dist, int n);
        std::vector<float> compute_ode_adjoint(const std::vector<float>& z, const std::vector<float>& grad, float dt);

        // Part 4: Frontier
        float update_meta_policy(float current_temp, float student_loss);
        float compute_causal_loss(const std::vector<float>& orig, const std::vector<float>& interv);
        float compute_quantum_loss(const std::vector<float>& s_state, const std::vector<float>& t_state);
        float compute_spike_loss(const std::vector<int>& s_spikes, const std::vector<int>& t_spikes);

        // Part 5: Physical
        float compute_photonic_loss(const std::vector<float>& weights, int w, int h);
        std::vector<float> inject_memristive_noise(const std::vector<float>& weights, float rate);

        // Part 6: Abstract
        float compute_functorial_loss(const std::vector<float>& map_t, const std::vector<float>& map_s);
        float compute_tom_loss(const std::vector<float>& true_b, const std::vector<float>& pred_b);

        // Part 7: Social
        float compute_nash_regret(const std::vector<float>& nash, const std::vector<float>& student);
        std::vector<float> update_swarm(const std::vector<float>& curr, const std::vector<float>& best, const std::vector<float>& vel);

        // Part 8: Integrity
        std::vector<float> compute_pate_aggregation(const std::vector<int>& votes, float epsilon);

    private:
        std::unique_ptr<uhk::runtime::UniversalKernelRuntime> kernel_runtime_;
        bool initialized_ = false;

        float* d_A = nullptr;
        float* d_B = nullptr;
        float* d_C = nullptr;
    };

}
