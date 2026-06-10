// ============================================================================
// NSOS learns Flappy Bird & Solo Pong by MULTITASK BEHAVIOR CLONING (JambaModel).
// ============================================================================
#include "../include/jamba.h"
#include "../include/trainer.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <numeric>
#include <thread>
#include <vector>

#include "flappy.h"
#include "flappy_tokenizer.h"

using namespace nsos;
using flappy::Game;
using flappy::PongGame;
using flappy::Tok;
using flappy::GameRenderContext;

namespace {

constexpr int SEQ_LEN = 12;
constexpr int MAX_FRAMES = 600;

// Evaluation policies for Flappy and Pong tasks
struct FlappyModelPolicy {
  JambaModel* model;
  mutable std::vector<int> ctx;
  bool operator()(const Game& g) const {
    ctx.push_back(Tok::TASK_FLAPPY);
    ctx.push_back(Tok::flappy_state_token(g));
    const int win_n = std::min<int>(SEQ_LEN, static_cast<int>(ctx.size()));
    std::vector<int> window(ctx.end() - win_n, ctx.end());
    model->reset_session();
    Tensor logits = model->forward_ids(window, nullptr);
    Tensor host = (logits.get_device() == Device::GPU) ? logits.cpu() : logits;
    const int vocab = host.shape.back();
    const float* row = host.data() + host.size - vocab;
    const bool flap = row[Tok::TOKEN_FLAPPY_FLAP] > row[Tok::TOKEN_FLAPPY_NOFLAP];
    ctx.push_back(Tok::flappy_action_token(flap));
    if (ctx.size() > 256) ctx.erase(ctx.begin(), ctx.begin() + 128);
    return flap;
  }
};

struct PongModelPolicy {
  JambaModel* model;
  mutable std::vector<int> ctx;
  int operator()(const PongGame& p) const {
    ctx.push_back(Tok::TASK_PONG);
    ctx.push_back(Tok::pong_state_token(p));
    const int win_n = std::min<int>(SEQ_LEN, static_cast<int>(ctx.size()));
    std::vector<int> window(ctx.end() - win_n, ctx.end());
    model->reset_session();
    Tensor logits = model->forward_ids(window, nullptr);
    Tensor host = (logits.get_device() == Device::GPU) ? logits.cpu() : logits;
    const int vocab = host.shape.back();
    const float* row = host.data() + host.size - vocab;
    
    float max_logit = row[Tok::TOKEN_PONG_STAY];
    int action = 0;
    if (row[Tok::TOKEN_PONG_UP] > max_logit) {
      max_logit = row[Tok::TOKEN_PONG_UP];
      action = 1;
    }
    if (row[Tok::TOKEN_PONG_DOWN] > max_logit) {
      max_logit = row[Tok::TOKEN_PONG_DOWN];
      action = 2;
    }
    ctx.push_back(Tok::pong_action_token(action));
    if (ctx.size() > 256) ctx.erase(ctx.begin(), ctx.begin() + 128);
    return action;
  }
};

// Play helpers for single game evaluations
template <typename Policy>
int play_flappy(Game& g, uint32_t seed, Policy policy, int max_frames = 600) {
  g.reset(seed);
  while (g.alive && g.frames < max_frames) {
    const bool flap = policy(g);
    g.step(flap);
  }
  return g.score;
}

template <typename Policy>
int play_pong(PongGame& p, uint32_t seed, Policy policy, int max_frames = 600) {
  p.reset(seed);
  while (p.alive && p.frames < max_frames) {
    const int action = policy(p);
    p.step(action);
  }
  return p.score;
}

// Live play watchdog using Win32 GDI split screen (1280 width)
void play_multitask_gui(JambaModel& model) {
  Game fg;
  PongGame pg;
  
  uint32_t f_seed = 70000;
  uint32_t p_seed = 80000;
  
  fg.reset(f_seed++);
  pg.reset(p_seed++);

  FlappyModelPolicy f_policy{&model, {}};
  PongModelPolicy p_policy{&model, {}};

  int f_best = 0;
  int p_best = 0;

  GameRenderContext renderCtx;
  HWND hwnd = create_game_window(fg, pg, "NSOS Multitask", f_best, p_best, renderCtx);
  if (!hwnd) return;

  while (true) {
    // Process messages
    MSG msg;
    while (PeekMessageW(&msg, NULL, 0, 0, PM_REMOVE)) {
      if (msg.message == WM_QUIT) {
        DestroyWindow(hwnd);
        return;
      }
      TranslateMessage(&msg);
      DispatchMessageW(&msg);
    }

    // Step Flappy Bird
    if (fg.alive) {
      const bool flap = f_policy(fg);
      fg.step(flap);
    } else {
      f_best = std::max(f_best, fg.score);
      fg.reset(f_seed++);
      f_policy.ctx.clear();
    }

    // Step Pong
    if (pg.alive) {
      const int action = p_policy(pg);
      pg.step(action);
    } else {
      p_best = std::max(p_best, pg.score);
      pg.reset(p_seed++);
      p_policy.ctx.clear();
    }

    // Update screen paint
    renderCtx.flappy_best = f_best;
    renderCtx.pong_best = p_best;
    InvalidateRect(hwnd, NULL, FALSE);
    UpdateWindow(hwnd);

    // Smooth ~33 FPS
    std::this_thread::sleep_for(std::chrono::milliseconds(30));
  }
}

}  // namespace

int main() {
  std::setvbuf(stdout, nullptr, _IONBF, 0);

  std::printf("=====================================================================\n");
  std::printf(" NSOS Multitask Learning: Flappy Bird & Solo Pong (JambaModel)\n");
  std::printf("=====================================================================\n");

  // ---- baselines ----
  long f_tot = 0;
  for (int i = 0; i < 30; ++i) {
    Game g;
    f_tot += play_flappy(g, 1000 + i, [](const Game& gg) { return gg.heuristic(); }, 10000);
  }
  float f_heur = static_cast<float>(f_tot) / 30;

  long p_tot = 0;
  for (int i = 0; i < 30; ++i) {
    PongGame p;
    p_tot += play_pong(p, 2000 + i, [](const PongGame& pg) { return pg.heuristic(); }, 10000);
  }
  float p_heur = static_cast<float>(p_tot) / 30;

  std::printf(" baselines: Flappy Heuristic avg=%.2f   Pong Heuristic avg=%.2f\n\n", f_heur, p_heur);

  // ---- generate training data: heuristic games -> [task, state, action] token stream ----
  std::vector<int> dataset;
  dataset.reserve(900000);
  Game fg;
  PongGame pg;
  for (int ep = 0; ep < 250; ++ep) {
    // Flappy episode
    fg.reset(10000 + static_cast<uint32_t>(ep));
    while (fg.alive && fg.frames < MAX_FRAMES) {
      dataset.push_back(Tok::TASK_FLAPPY);
      dataset.push_back(Tok::flappy_state_token(fg));
      const bool flap = fg.heuristic();
      dataset.push_back(Tok::flappy_action_token(flap));
      fg.step(flap);
    }
    // Pong episode
    pg.reset(20000 + static_cast<uint32_t>(ep));
    while (pg.alive && pg.frames < MAX_FRAMES) {
      dataset.push_back(Tok::TASK_PONG);
      dataset.push_back(Tok::pong_state_token(pg));
      const int action = pg.heuristic();
      dataset.push_back(Tok::pong_action_token(action));
      pg.step(action);
    }
  }
  std::printf(" training tokens: %zu  (vocab=%d)\n", dataset.size(), Tok::VOCAB);

  // ---- build + train the real NSOS model ----
  JambaModel model(2, 96, Tok::VOCAB, Device::CPU);
  Trainer trainer(&model, 3e-3f);
  trainer.weight_decay = 0.0f;
  trainer.max_grad_norm = 2.0f;
  trainer.warmup_steps = 10;
  std::printf(" training a 2-layer d=96 NSOS model...\n");
  const auto t0 = std::chrono::steady_clock::now();
  trainer.train_loop(dataset, /*epochs*/ 6, /*batch*/ 16, SEQ_LEN,
                     [&](int step, float loss) {
                       if (step == 1 || step % 100 == 0)
                         std::printf("   training... step %4d   loss %.4f\n", step, loss);
                     }, /*max_steps*/ 8000);
  const double train_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  std::printf(" trained in %.1fs\n\n", train_s);

  // ---- WATCH the trained model play live, right after training ----
  if (!std::getenv("FLAPPY_NOWATCH")) {
    std::printf(" >>> Now watch the NSOS multitask model PLAY. Close window to stop. <<<\n");
    std::this_thread::sleep_for(std::chrono::milliseconds(1500));
    play_multitask_gui(model);
  }

  // ---- did it learn the policy? action-prediction accuracy on UNSEEN games ----
  long agree_f = 0, total_f = 0;
  FlappyModelPolicy f_policy_eval{&model, {}};
  for (int ep = 0; ep < 20; ++ep) {
    Game g; g.reset(50000 + ep);
    f_policy_eval.ctx.clear();
    while (g.alive && g.frames < 600) {
      const bool model_flap = f_policy_eval(g);
      const bool heur_flap = g.heuristic();
      agree_f += (model_flap == heur_flap) ? 1 : 0;
      ++total_f;
      f_policy_eval.ctx.back() = Tok::flappy_action_token(heur_flap);
      g.step(heur_flap);
    }
  }

  long agree_p = 0, total_p = 0;
  PongModelPolicy p_policy_eval{&model, {}};
  for (int ep = 0; ep < 20; ++ep) {
    PongGame p; p.reset(60000 + ep);
    p_policy_eval.ctx.clear();
    while (p.alive && p.frames < 600) {
      const int model_act = p_policy_eval(p);
      const int heur_act = p.heuristic();
      agree_p += (model_act == heur_act) ? 1 : 0;
      ++total_p;
      p_policy_eval.ctx.back() = Tok::pong_action_token(heur_act);
      p.step(heur_act);
    }
  }

  std::printf(" Action-prediction accuracy (teacher-forced):\n");
  std::printf("   Flappy Bird: %.1f%%\n", 100.0f * agree_f / std::max<long>(total_f, 1));
  std::printf("   Solo Pong:   %.1f%%\n\n", 100.0f * agree_p / std::max<long>(total_p, 1));

  // ---- aggregate self-play score over fresh games ----
  long sf_tot = 0; int sf_best = 0;
  for (int i = 0; i < 10; ++i) {
    Game g;
    f_policy_eval.ctx.clear();
    int sc = play_flappy(g, 70000 + i, f_policy_eval, 10000);
    sf_tot += sc; sf_best = std::max(sf_best, sc);
  }

  long sp_tot = 0; int sp_best = 0;
  for (int i = 0; i < 10; ++i) {
    PongGame p;
    p_policy_eval.ctx.clear();
    int sc = play_pong(p, 80000 + i, p_policy_eval, 10000);
    sp_tot += sc; sp_best = std::max(sp_best, sc);
  }

  std::printf("=====================================================================\n");
  std::printf(" Multitask AI Self-Play Performance (Split Screen):\n");
  std::printf("   Flappy Bird: avg=%.2f  best=%d  |  Heuristic baseline %.2f\n",
              static_cast<float>(sf_tot) / 10, sf_best, f_heur);
  std::printf("   Solo Pong:   avg=%.2f  best=%d  |  Heuristic baseline %.2f\n",
              static_cast<float>(sp_tot) / 10, sp_best, p_heur);
  std::printf("=====================================================================\n");

  return 0;
}
