#pragma once
// Self-contained Flappy-Bird & Pong sim + Win32 GDI dual-pane renderer.
#include <cmath>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace flappy {

// y in [0,1] (0 = top, 1 = bottom).
struct Game {
  float bird_y = 0.5f, bird_v = 0.0f;
  float gap_y = 0.5f, gap_half = 0.18f;
  float pipe_x = 1.0f;            // horizontal distance of the next pipe (1->0)
  int score = 0, frames = 0;
  bool alive = true;
  std::mt19937 rng;

  static constexpr float GRAVITY = 0.0012f;
  static constexpr float FLAP = -0.012f;
  static constexpr float VMAX = 0.020f;
  static constexpr float PIPE_SPEED = 0.018f;   // base speed

  void reset(uint32_t seed) {
    rng.seed(seed);
    bird_y = 0.5f; bird_v = 0.0f; pipe_x = 1.0f; score = 0; frames = 0; alive = true;
    new_gap();
  }
  void new_gap() {
    std::uniform_real_distribution<float> u(0.30f, 0.70f);
    gap_y = u(rng);
  }
  // advance one frame given the action; returns alive (with difficulty scaling)
  bool step(bool flap) {
    if (!alive) return false;
    bird_v = flap ? FLAP : (bird_v + GRAVITY);
    if (bird_v > VMAX) bird_v = VMAX;
    if (bird_v < -VMAX) bird_v = -VMAX;
    bird_y += bird_v;

    // Difficulty scaling: speed increases, gap narrows over time
    float pipe_speed = PIPE_SPEED + static_cast<float>(score) * 0.0008f;
    float current_gap_half = std::max(0.12f, 0.18f - static_cast<float>(score) * 0.002f);

    pipe_x -= pipe_speed;
    if (pipe_x <= 0.0f) {                       // pipe reaches the bird column
      if (std::fabs(bird_y - gap_y) > current_gap_half) { alive = false; }
      else { ++score; pipe_x = 1.0f; new_gap(); }
    }
    if (bird_y < 0.0f || bird_y > 1.0f) alive = false;
    ++frames;
    return alive;
  }
  // Lookahead heuristic policy
  bool heuristic() const {
    const float projected = bird_y + bird_v * 3.0f;
    return projected > gap_y;
  }
};

// Pong Game Struct
struct PongGame {
  float paddle_y = 0.5f;
  float ball_x = 0.5f, ball_y = 0.5f;
  float ball_vx = 0.012f, ball_vy = 0.008f;
  int score = 0, frames = 0;
  bool alive = true;
  std::mt19937 rng;

  static constexpr float PADDLE_SPEED = 0.018f;
  static constexpr float BALL_INIT_VX = 0.012f;
  static constexpr float BALL_INIT_VY = 0.008f;

  void reset(uint32_t seed) {
    rng.seed(seed);
    paddle_y = 0.5f;
    ball_x = 0.5f;
    ball_y = 0.5f;
    
    std::uniform_real_distribution<float> u(-0.5f, 0.5f);
    ball_vx = BALL_INIT_VX;
    ball_vy = BALL_INIT_VY * (u(rng) > 0 ? 1.0f : -1.0f);
    score = 0;
    frames = 0;
    alive = true;
  }

  // Action: 0 = STAY, 1 = UP, 2 = DOWN
  bool step(int action) {
    if (!alive) return false;

    // Difficulty scaling: speed increases, paddle size shrinks
    float speed_mult = 1.0f + static_cast<float>(score) * 0.03f;
    float current_vx = (ball_vx > 0 ? 1.0f : -1.0f) * BALL_INIT_VX * speed_mult;
    float current_vy = (ball_vy > 0 ? 1.0f : -1.0f) * BALL_INIT_VY * speed_mult;
    float paddle_half = std::max(0.08f, 0.15f - static_cast<float>(score) * 0.003f);

    // Paddle movement
    if (action == 1) paddle_y -= PADDLE_SPEED;
    if (action == 2) paddle_y += PADDLE_SPEED;
    if (paddle_y < 0.0f) paddle_y = 0.0f;
    if (paddle_y > 1.0f) paddle_y = 1.0f;

    // Ball movement
    ball_x += current_vx;
    ball_y += current_vy;

    // Collisions
    if (ball_y <= 0.0f) {
      ball_y = 0.0f;
      ball_vy = std::fabs(ball_vy);
    }
    if (ball_y >= 1.0f) {
      ball_y = 1.0f;
      ball_vy = -std::fabs(ball_vy);
    }
    if (ball_x >= 1.0f) {
      ball_x = 1.0f;
      ball_vx = -std::fabs(ball_vx);
    }

    // Paddle hit check
    if (ball_x <= 0.05f) {
      if (std::fabs(ball_y - paddle_y) <= paddle_half) {
        ball_x = 0.05f;
        ball_vx = std::fabs(ball_vx);
        ++score;
        // slightly randomize vertical speed to make it organic
        std::uniform_real_distribution<float> ur(-0.002f, 0.002f);
        ball_vy += ur(rng);
      } else {
        alive = false;
      }
    }

    ++frames;
    return alive;
  }

  // Heuristic policy
  int heuristic() const {
    if (paddle_y < ball_y - 0.03f) return 2; // DOWN
    if (paddle_y > ball_y + 0.03f) return 1; // UP
    return 0; // STAY
  }
};

// ---- Win32 GDI Renderer ----

#pragma comment(lib, "user32.lib")
#pragma comment(lib, "gdi32.lib")

struct GameRenderContext {
  const Game* flappy_g = nullptr;
  const PongGame* pong_g = nullptr;
  const char* who = "";
  int flappy_best = 0;
  int pong_best = 0;
};

// Double-buffered Split-Screen Render (1280x680)
inline void RenderGameGDI(HDC hdc, HWND hwnd, const Game& fg, const PongGame& pg, const char* who, int f_best, int p_best) {
  RECT rect;
  GetClientRect(hwnd, &rect);
  int width = rect.right - rect.left;
  int height = rect.bottom - rect.top;
  int half_w = width / 2;

  HDC memDC = CreateCompatibleDC(hdc);
  HBITMAP memBitmap = CreateCompatibleBitmap(hdc, width, height);
  HBITMAP oldBitmap = (HBITMAP)SelectObject(memDC, memBitmap);

  // -------------------------------------------------------------
  // LEFT PANEL: Flappy Bird (x in [0, half_w])
  // -------------------------------------------------------------
  // Draw sky blue gradient
  for (int y = 0; y < height; ++y) {
    float t = static_cast<float>(y) / height;
    BYTE r = static_cast<BYTE>(20 + t * 60);
    BYTE g = static_cast<BYTE>(140 + t * 60);
    BYTE b = 255;
    RECT rowRect = { 0, y, half_w, y + 1 };
    HBRUSH rowBrush = CreateSolidBrush(RGB(r, g, b));
    FillRect(memDC, &rowRect, rowBrush);
    DeleteObject(rowBrush);
  }

  // Draw some subtle green ground at the bottom
  int ground_h = height / 8;
  int playable_h = height - ground_h;
  RECT groundRect = { 0, height - ground_h, half_w, height };
  HBRUSH groundBrush = CreateSolidBrush(RGB(100, 200, 80));
  FillRect(memDC, &groundRect, groundBrush);
  DeleteObject(groundBrush);

  // Draw grass details
  HPEN grassPen = CreatePen(PS_SOLID, 3, RGB(70, 160, 50));
  HPEN oldPen = (HPEN)SelectObject(memDC, grassPen);
  for (int x = 0; x < half_w; x += 30) {
    MoveToEx(memDC, x, height - ground_h, NULL);
    LineTo(memDC, x + 10, height - ground_h - 10);
  }
  SelectObject(memDC, oldPen);
  DeleteObject(grassPen);

  // Scale variables for Flappy
  float bird_x = 0.159f;
  int bird_px_x = static_cast<int>(bird_x * half_w);
  int bird_px_y = static_cast<int>(fg.bird_y * playable_h);
  int bird_radius = static_cast<int>(0.035f * height);
  if (bird_radius < 12) bird_radius = 12;

  // Draw Flappy Pipes
  float pipe_screen_x = bird_x + fg.pipe_x * (1.0f - bird_x);
  int pipe_px_x = static_cast<int>(pipe_screen_x * half_w);
  int pipe_width = static_cast<int>(0.10f * half_w);
  if (pipe_width < 35) pipe_width = 35;

  float current_gap_half = std::max(0.12f, 0.18f - static_cast<float>(fg.score) * 0.002f);
  int gap_top_py = static_cast<int>((fg.gap_y - current_gap_half) * playable_h);
  int gap_bot_py = static_cast<int>((fg.gap_y + current_gap_half) * playable_h);

  HBRUSH pipeBrush = CreateSolidBrush(RGB(40, 190, 70));
  HBRUSH pipeBorderBrush = CreateSolidBrush(RGB(10, 100, 30));
  HPEN pipePen = CreatePen(PS_SOLID, 2, RGB(10, 100, 30));
  oldPen = (HPEN)SelectObject(memDC, pipePen);

  // Draw upper pipe
  RECT upperPipe = { pipe_px_x, 0, pipe_px_x + pipe_width, gap_top_py };
  FillRect(memDC, &upperPipe, pipeBrush);
  FrameRect(memDC, &upperPipe, pipeBorderBrush);

  // Draw upper pipe lip
  int lip_h = 15;
  RECT upperLip = { pipe_px_x - 4, gap_top_py - lip_h, pipe_px_x + pipe_width + 4, gap_top_py };
  FillRect(memDC, &upperLip, pipeBrush);
  FrameRect(memDC, &upperLip, pipeBorderBrush);

  // Draw lower pipe
  RECT lowerPipe = { pipe_px_x, gap_bot_py, pipe_px_x + pipe_width, playable_h };
  FillRect(memDC, &lowerPipe, pipeBrush);
  FrameRect(memDC, &lowerPipe, pipeBorderBrush);

  // Draw lower pipe lip
  RECT lowerLip = { pipe_px_x - 4, gap_bot_py, pipe_px_x + pipe_width + 4, gap_bot_py + lip_h };
  FillRect(memDC, &lowerLip, pipeBrush);
  FrameRect(memDC, &lowerLip, pipeBorderBrush);

  SelectObject(memDC, oldPen);
  DeleteObject(pipePen);
  DeleteObject(pipeBrush);
  DeleteObject(pipeBorderBrush);

  // Draw Bird
  HBRUSH birdBrush = fg.alive ? CreateSolidBrush(RGB(255, 230, 0)) : CreateSolidBrush(RGB(220, 80, 50));
  HBRUSH borderBrush = CreateSolidBrush(RGB(180, 100, 0));
  HPEN birdPen = CreatePen(PS_SOLID, 2, RGB(180, 100, 0));
  oldPen = (HPEN)SelectObject(memDC, birdPen);
  
  Ellipse(memDC, bird_px_x - bird_radius, bird_px_y - bird_radius, bird_px_x + bird_radius, bird_px_y + bird_radius);

  // Eye
  HBRUSH whiteBrush = CreateSolidBrush(RGB(255, 255, 255));
  SelectObject(memDC, whiteBrush);
  Ellipse(memDC, bird_px_x + bird_radius / 3, bird_px_y - bird_radius / 2, bird_px_x + bird_radius * 5 / 6, bird_px_y - bird_radius / 8);

  // Pupil
  HBRUSH blackBrush = CreateSolidBrush(RGB(0, 0, 0));
  SelectObject(memDC, blackBrush);
  Ellipse(memDC, bird_px_x + bird_radius * 3 / 5, bird_px_y - bird_radius * 3 / 8, bird_px_x + bird_radius * 5 / 6, bird_px_y - bird_radius / 4);

  // Beak
  HBRUSH beakBrush = CreateSolidBrush(RGB(255, 130, 0));
  SelectObject(memDC, beakBrush);
  POINT beak[3];
  beak[0].x = bird_px_x + bird_radius - 2; beak[0].y = bird_px_y - 2;
  beak[1].x = bird_px_x + bird_radius + bird_radius / 2; beak[1].y = bird_px_y + bird_radius / 4;
  beak[2].x = bird_px_x + bird_radius - 2; beak[2].y = bird_px_y + bird_radius / 2;
  Polygon(memDC, beak, 3);

  // Wing
  HBRUSH wingBrush = CreateSolidBrush(RGB(255, 255, 180));
  SelectObject(memDC, wingBrush);
  int flap_offset = static_cast<int>(fg.bird_v * 200.0f);
  Ellipse(memDC, bird_px_x - bird_radius * 4 / 5, bird_px_y - bird_radius / 3 + flap_offset, bird_px_x - bird_radius / 5, bird_px_y + bird_radius / 3 + flap_offset);

  SelectObject(memDC, oldPen);
  DeleteObject(birdPen); DeleteObject(birdBrush); DeleteObject(borderBrush);
  DeleteObject(whiteBrush); DeleteObject(blackBrush); DeleteObject(beakBrush); DeleteObject(wingBrush);

  // Display Flappy HUD
  SetBkMode(memDC, TRANSPARENT);
  HFONT hFont = CreateFontW(26, 0, 0, 0, FW_BOLD, FALSE, FALSE, FALSE, DEFAULT_CHARSET, OUT_OUTLINE_PRECIS,
                           CLIP_DEFAULT_PRECIS, CLEARTYPE_QUALITY, VARIABLE_PITCH | FF_SWISS, L"Segoe UI");
  HFONT oldFont = (HFONT)SelectObject(memDC, hFont);

  std::wstring f_hud = L"Flappy Bird - Score: " + std::to_wstring(fg.score) + L"   Best: " + std::to_wstring(f_best);
  SetTextColor(memDC, RGB(0, 0, 0));
  TextOutW(memDC, 22, 22, f_hud.c_str(), static_cast<int>(f_hud.length()));
  SetTextColor(memDC, RGB(255, 255, 255));
  TextOutW(memDC, 20, 20, f_hud.c_str(), static_cast<int>(f_hud.length()));

  if (!fg.alive) {
    SetTextColor(memDC, RGB(240, 50, 50));
    std::wstring goText = L"GAME OVER";
    TextOutW(memDC, half_w / 2 - 60, height / 2 - 20, goText.c_str(), static_cast<int>(goText.length()));
  }

  // -------------------------------------------------------------
  // RIGHT PANEL: Solo Pong (x in [half_w, width])
  // -------------------------------------------------------------
  // Draw dark retro background
  for (int y = 0; y < height; ++y) {
    float t = static_cast<float>(y) / height;
    BYTE r = static_cast<BYTE>(10 + t * 10);
    BYTE g = static_cast<BYTE>(10 + t * 15);
    BYTE b = static_cast<BYTE>(20 + t * 30);
    RECT rowRect = { half_w, y, width, y + 1 };
    HBRUSH rowBrush = CreateSolidBrush(RGB(r, g, b));
    FillRect(memDC, &rowRect, rowBrush);
    DeleteObject(rowBrush);
  }

  // Draw dividing dashed center line inside Pong arena
  HPEN dashPen = CreatePen(PS_DASH, 2, RGB(60, 60, 100));
  oldPen = (HPEN)SelectObject(memDC, dashPen);
  MoveToEx(memDC, half_w + (width - half_w)/2, 0, NULL);
  LineTo(memDC, half_w + (width - half_w)/2, height);
  SelectObject(memDC, oldPen);
  DeleteObject(dashPen);

  // Scale variables for Pong
  int pong_arena_w = width - half_w;
  float paddle_x = 0.05f;
  int pad_px_x = half_w + static_cast<int>(paddle_x * pong_arena_w);
  int pad_px_y = static_cast<int>(pg.paddle_y * height);
  float paddle_half = std::max(0.08f, 0.15f - static_cast<float>(pg.score) * 0.003f);
  int pad_h = static_cast<int>(paddle_half * 2.0f * height);
  int pad_w = 12;

  int ball_px_x = half_w + static_cast<int>(pg.ball_x * pong_arena_w);
  int ball_px_y = static_cast<int>(pg.ball_y * height);
  int ball_radius = 8;

  // Draw neon paddle (Cian)
  HBRUSH padBrush = pg.alive ? CreateSolidBrush(RGB(0, 240, 255)) : CreateSolidBrush(RGB(120, 120, 120));
  RECT padRect = { pad_px_x, pad_px_y - pad_h/2, pad_px_x + pad_w, pad_px_y + pad_h/2 };
  FillRect(memDC, &padRect, padBrush);
  DeleteObject(padBrush);

  // Draw neon ball (Red / Magenta)
  HBRUSH ballBrush = pg.alive ? CreateSolidBrush(RGB(255, 0, 128)) : CreateSolidBrush(RGB(120, 120, 120));
  SelectObject(memDC, ballBrush);
  Ellipse(memDC, ball_px_x - ball_radius, ball_px_y - ball_radius, ball_px_x + ball_radius, ball_px_y + ball_radius);
  DeleteObject(ballBrush);

  // Display Pong HUD
  std::wstring p_hud = L"Solo Pong - Score: " + std::to_wstring(pg.score) + L"   Best: " + std::to_wstring(p_best);
  SetTextColor(memDC, RGB(0, 0, 0));
  TextOutW(memDC, half_w + 22, 22, p_hud.c_str(), static_cast<int>(p_hud.length()));
  SetTextColor(memDC, RGB(0, 240, 255));
  TextOutW(memDC, half_w + 20, 20, p_hud.c_str(), static_cast<int>(p_hud.length()));

  if (!pg.alive) {
    SetTextColor(memDC, RGB(255, 80, 80));
    std::wstring goText = L"GAME OVER";
    TextOutW(memDC, half_w + pong_arena_w / 2 - 60, height / 2 - 20, goText.c_str(), static_cast<int>(goText.length()));
  }

  // -------------------------------------------------------------
  // SHARED: Dividing Line and Metadata HUD
  // -------------------------------------------------------------
  // Draw vertical split dividing line (solid gray)
  HPEN splitPen = CreatePen(PS_SOLID, 4, RGB(120, 120, 120));
  oldPen = (HPEN)SelectObject(memDC, splitPen);
  MoveToEx(memDC, half_w, 0, NULL);
  LineTo(memDC, half_w, height);
  SelectObject(memDC, oldPen);
  DeleteObject(splitPen);

  // Display multitasking player tag
  if (who && who[0]) {
    std::string wStr(who);
    std::wstring wWStr(wStr.begin(), wStr.end());
    std::wstring tagText = L"Multitask Model: " + wWStr;
    
    // Draw shadow
    SetTextColor(memDC, RGB(0, 0, 0));
    TextOutW(memDC, width / 2 - 128, height - 48, tagText.c_str(), static_cast<int>(tagText.length()));
    
    // Draw main (White)
    SetTextColor(memDC, RGB(255, 255, 255));
    TextOutW(memDC, width / 2 - 130, height - 50, tagText.c_str(), static_cast<int>(tagText.length()));
  }

  // Blit the entire back buffer to screen
  BitBlt(hdc, 0, 0, width, height, memDC, 0, 0, SRCCOPY);

  SelectObject(memDC, oldBitmap);
  SelectObject(memDC, oldFont);
  DeleteObject(memBitmap);
  DeleteDC(memDC);
  DeleteObject(hFont);
}

// Window Procedure
inline LRESULT CALLBACK GameWndProc(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam) {
  switch (uMsg) {
    case WM_PAINT: {
      PAINTSTRUCT ps;
      HDC hdc = BeginPaint(hwnd, &ps);
      auto* ctx = reinterpret_cast<GameRenderContext*>(GetWindowLongPtrW(hwnd, GWLP_USERDATA));
      if (ctx && ctx->flappy_g && ctx->pong_g) {
        RenderGameGDI(hdc, hwnd, *(ctx->flappy_g), *(ctx->pong_g), ctx->who, ctx->flappy_best, ctx->pong_best);
      }
      EndPaint(hwnd, &ps);
      return 0;
    }
    case WM_CLOSE:
      DestroyWindow(hwnd);
      PostQuitMessage(0);
      return 0;
    case WM_DESTROY:
      return 0;
  }
  return DefWindowProcW(hwnd, uMsg, wParam, lParam);
}

// Helper to register and create split window (1280 width)
inline HWND create_game_window(const Game& fg, const PongGame& pg, const char* who, int f_best, int p_best, GameRenderContext& ctx) {
  static bool registered = false;
  HINSTANCE hInst = GetModuleHandleW(NULL);
  if (!registered) {
    WNDCLASSW wc = {0};
    wc.lpfnWndProc = GameWndProc;
    wc.hInstance = hInst;
    wc.hbrBackground = NULL;
    wc.lpszClassName = L"FlappyBirdGameClass";
    wc.hCursor = LoadCursor(NULL, IDC_ARROW);
    RegisterClassW(&wc);
    registered = true;
  }

  ctx.flappy_g = &fg;
  ctx.pong_g = &pg;
  ctx.who = who;
  ctx.flappy_best = f_best;
  ctx.pong_best = p_best;

  HWND hwnd = CreateWindowExW(
    0,
    L"FlappyBirdGameClass",
    L"NSOS Multitask AI - Flappy Bird & Solo Pong Split Screen",
    WS_OVERLAPPEDWINDOW & ~WS_THICKFRAME & ~WS_MAXIMIZEBOX,
    CW_USEDEFAULT, CW_USEDEFAULT, 1280, 680,
    NULL, NULL, hInst, NULL
  );

  if (hwnd) {
    SetWindowLongPtrW(hwnd, GWLP_USERDATA, reinterpret_cast<LONG_PTR>(&ctx));
    ShowWindow(hwnd, SW_SHOW);
    UpdateWindow(hwnd);
  }
  return hwnd;
}

}  // namespace flappy
