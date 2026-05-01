#include "../include/tensor.h"
#include <iostream>
#include <vector>

using namespace nsos;

#ifdef _WIN32
#include <winsock2.h>
#include <ws2tcpip.h>
#pragma comment(lib, "Ws2_32.lib")
typedef int socklen_t;
#define CLOSE_SOCKET closesocket
#else
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>
#define CLOSE_SOCKET close
#define SOCKET int
#define INVALID_SOCKET -1
#define SOCKET_ERROR -1
#endif

#include "jamba.h"
#include "tokenizer.h"

// Simple HTTP Server for Inference
int main() {
#ifdef _WIN32
  WSADATA wsaData;
  int iResult = WSAStartup(MAKEWORD(2, 2), &wsaData);
  if (iResult != 0) {
    std::cout << "WSAStartup failed: " << iResult << std::endl;
    return 1;
  }
#endif

  SOCKET server_fd = socket(AF_INET, SOCK_STREAM, 0);
  if (server_fd == INVALID_SOCKET) {
    perror("Socket failed");
    return 1;
  }

  sockaddr_in address;
  address.sin_family = AF_INET;
  address.sin_addr.s_addr = INADDR_ANY;
  address.sin_port = htons(8080);

  if (bind(server_fd, (sockaddr *)&address, sizeof(address)) == SOCKET_ERROR) {
    perror("Bind failed");
    return 1;
  }

  if (listen(server_fd, 3) == SOCKET_ERROR) {
    perror("Listen failed");
    return 1;
  }

  std::cout << "NSOS Inference Server running on port 8080..." << std::endl;

  JambaModel model(4, 64);
  Tokenizer tokenizer;

  while (true) {
    SOCKET new_socket = accept(server_fd, nullptr, nullptr);
    if (new_socket == INVALID_SOCKET)
      continue;

    // V101 FIX: Safe buffer handling
    char buffer[1024] = {0};
    int bytes_recv = recv(new_socket, buffer, sizeof(buffer) - 1, 0);
    if (bytes_recv <= 0) {
      closesocket(new_socket);
      continue;
    }
    buffer[bytes_recv] = '\0'; // Ensure null termination

    // Parse "POST /generate ... body: prompt"
    // Dummy parsing: Assume body is the prompt
    std::string req(buffer);
    std::string prompt = "Hello"; // Default

    // Basic Logic: Run model
    std::vector<int> tokens = tokenizer.encode(prompt);

    // Convert to Tensor
    // Assuming Batch=1
    Tensor input = Tensor::zeros({(int)tokens.size(), 64}, Device::CPU);
    // Fill dummy embedding (should be real embedding lookup)

    // Run Forward (Latent Thought)
    Tensor latent = model.forward_thought(input, 1);

    // Decode (Stub: Just map back to token 0 for demo as we lack Head in Server
    // context) In real server, we need the Head layer here too. Assuming we
    // just reply with a fixed string proving execution.

    std::string response_text = "NSOS Processed: " + prompt;
    std::string response_body = "{\"text\": \"" + response_text + "\"}";

    std::string response =
        "HTTP/1.1 200 OK\nContent-Type: application/json\nContent-Length: " +
        std::to_string(response_body.length()) + "\n\n" + response_body;

    send(new_socket, response.c_str(), response.length(), 0);
    CLOSE_SOCKET(new_socket);
  }

#ifdef _WIN32
  WSACleanup();
#endif
  return 0;
}
