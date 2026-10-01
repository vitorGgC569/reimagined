#pragma once
// Multitask-specific tokenizer. Exposes a unified, non-overlapping token space
// for both Flappy Bird and Solo Pong tasks.
#include <algorithm>
#include <cmath>
#include "flappy.h"

namespace flappy {

struct Tok {
  // Task indicator tokens
  static constexpr int TASK_FLAPPY = 0;
  static constexpr int TASK_PONG = 1;

  // Flappy states (256 states)
  static constexpr int NB_OFF_FLAPPY = 32;
  static constexpr int NB_VEL_FLAPPY = 8;
  static constexpr int NSTATE_FLAPPY = NB_OFF_FLAPPY * NB_VEL_FLAPPY; // 256
  
  // Pong states (512 states)
  static constexpr int NB_OFF_PONG = 16;
  static constexpr int NB_X_PONG = 8;
  static constexpr int NB_DIR_PONG = 4; // vx > 0 vs vx < 0, vy > 0 vs vy < 0
  static constexpr int NSTATE_PONG = NB_OFF_PONG * NB_X_PONG * NB_DIR_PONG; // 512

  // Offsets in vocab
  static constexpr int STATE_FLAPPY_START = 2;
  static constexpr int STATE_PONG_START = 2 + NSTATE_FLAPPY; // 258

  // Action tokens
  static constexpr int FLAPPY_ACTION_START = 2 + NSTATE_FLAPPY + NSTATE_PONG; // 770
  static constexpr int TOKEN_FLAPPY_FLAP = FLAPPY_ACTION_START; // 770
  static constexpr int TOKEN_FLAPPY_NOFLAP = FLAPPY_ACTION_START + 1; // 771

  static constexpr int PONG_ACTION_START = FLAPPY_ACTION_START + 2; // 772
  static constexpr int TOKEN_PONG_STAY = PONG_ACTION_START; // 772
  static constexpr int TOKEN_PONG_UP = PONG_ACTION_START + 1; // 773
  static constexpr int TOKEN_PONG_DOWN = PONG_ACTION_START + 2; // 774

  static constexpr int VOCAB = PONG_ACTION_START + 3; // 775

  static int clampi(int v, int lo, int hi) { return v < lo ? lo : (v > hi ? hi : v); }

  static int flappy_state_token(const Game& g) {
    const float off = g.bird_y - g.gap_y;
    int ob = static_cast<int>((off + 0.5f) * NB_OFF_FLAPPY);
    ob = clampi(ob, 0, NB_OFF_FLAPPY - 1);
    const float vn = (g.bird_v + Game::VMAX) / (2.0f * Game::VMAX);
    int vb = clampi(static_cast<int>(vn * NB_VEL_FLAPPY), 0, NB_VEL_FLAPPY - 1);
    return STATE_FLAPPY_START + ob * NB_VEL_FLAPPY + vb;
  }

  static int flappy_action_token(bool flap) {
    return flap ? TOKEN_FLAPPY_FLAP : TOKEN_FLAPPY_NOFLAP;
  }

  static int pong_state_token(const PongGame& p) {
    const float off = p.ball_y - p.paddle_y; // [-1, 1]
    int ob = static_cast<int>((off + 0.5f) * NB_OFF_PONG);
    ob = clampi(ob, 0, NB_OFF_PONG - 1);

    int xb = static_cast<int>(p.ball_x * NB_X_PONG);
    xb = clampi(xb, 0, NB_X_PONG - 1);

    int db = 0;
    if (p.ball_vx > 0) db += 2;
    if (p.ball_vy > 0) db += 1;

    return STATE_PONG_START + ob * NB_X_PONG * NB_DIR_PONG + xb * NB_DIR_PONG + db;
  }

  static int pong_action_token(int act) {
    return TOKEN_PONG_STAY + clampi(act, 0, 2);
  }
};

}  // namespace flappy
